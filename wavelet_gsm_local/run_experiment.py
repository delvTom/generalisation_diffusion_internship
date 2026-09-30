"""Fit per-subband local GSMs and evaluate coefficient denoising on MNIST.

Inputs: official MNIST train/test files, optional ``--download`` and image
size 28 or 64. Outputs: fitted GSM parameters, diagnostics and risk tables in
``--out-dir``. Example CPU check::

    python -m wavelet_gsm_local.run_experiment --smoke-test --device cpu --download
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wavelet-gsm-local")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_gsm_local.estimators import (
    METHODS,
    gaussian_posterior_center,
    gsm_local_posterior_center,
    gsm_self_posterior_center,
    gsm_local_posterior_center_tensor,
    gsm_self_posterior_center_tensor,
    predict_all,
    predict_all_tensor,
)
from wavelet_gsm_local.gsm import (
    GSMParameters,
    fit_two_state_gsm_torch,
    load_gsm,
    responsibilities,
    save_gsm,
)
from wavelet_gsm_local.patches import (
    CENTER_INDEX,
    PATCH_OFFSETS,
    PatchSelection,
    extract_patches,
    sample_valid_centers,
)
from wavelet_gsm_local.statistics import (
    EnergyStatistics,
    compute_energy_statistics,
    gsm_fourth_order_excess,
)
from wavelet_gsm_local.wavelets import (
    BAND_NAMES,
    analyze_haar_level2,
    audit_transform,
    load_mnist,
)

METHOD_LABELS = {
    "wiener_empirical_local": "Wiener empirical local",
    "wiener_matched_local": "Wiener matched local",
    "wiener_matched_self": "Wiener matched self",
    "gsm_self": "GSM self",
    "gsm_local": "GSM local",
}
METHOD_COLORS = {
    "wiener_empirical_local": "#111111",
    "wiener_matched_local": "#377eb8",
    "wiener_matched_self": "#984ea3",
    "gsm_self": "#ff7f00",
    "gsm_local": "#4daf4a",
}
LATENT_TIMES = (0.0, 0.25, 0.50, 0.75, 0.95, 1.0)
CI_COMPARISONS = (
    ("wiener_matched_self", "gsm_self", "marginal_non_gaussianity"),
    ("gsm_self", "gsm_local", "context_beyond_self"),
    ("wiener_matched_local", "gsm_local", "local_non_gaussianity"),
    ("wiener_empirical_local", "gsm_local", "practical_local_comparison"),
)


def parse_times(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("times must lie in [0,1]")
    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    default_times = ",".join(f"{value:.2f}" for value in np.linspace(0.0, 1.0, 21))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--image-size", type=int, choices=(28, 64), default=28)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--times", type=parse_times, default=parse_times(default_times))
    parser.add_argument("--patches-per-image", type=int, default=5)
    parser.add_argument("--num-train-images", type=int, default=60_000)
    parser.add_argument("--num-test-images", type=int, default=10_000)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--max-em-iterations", type=int, default=200)
    parser.add_argument("--em-tolerance", type=float, default=1e-7)
    parser.add_argument("--covariance-epsilon", type=float, default=1e-6)
    parser.add_argument("--scale-floor", type=float, default=1e-4)
    parser.add_argument("--white-noise-samples", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reuse-fits", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(
        args.patches_per_image,
        args.num_train_images,
        args.num_test_images,
        args.num_restarts,
        args.max_em_iterations,
        args.white_noise_samples,
    ) < 1:
        parser.error("all counts must be positive")
    if min(args.em_tolerance, args.covariance_epsilon, args.scale_floor) <= 0.0:
        parser.error("numerical tolerances and floors must be positive")
    if args.smoke_test:
        args.num_train_images = min(args.num_train_images, 512)
        args.num_test_images = min(args.num_test_images, 256)
        args.patches_per_image = min(args.patches_per_image, 2)
        args.num_restarts = min(args.num_restarts, 2)
        args.max_em_iterations = min(args.max_em_iterations, 30)
        args.white_noise_samples = min(args.white_noise_samples, 512)
        args.times = (0.0, 0.5, 1.0)
    if args.out_dir is None:
        if args.smoke_test:
            args.out_dir = Path(f"/tmp/wavelet_gsm_local_smoke_{args.image_size}")
        else:
            args.out_dir = Path(
                f"wavelet_gsm_local/results/mnist{args.image_size}_haar2"
            )
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def band_level(band: str) -> str:
    return f"level_{band[-1]}"


def selection_to_arrays(prefix: str, selection: PatchSelection) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_image": selection.image_indices,
        f"{prefix}_row": selection.center_rows,
        f"{prefix}_column": selection.center_columns,
    }


def statistics_rows(
    band: str, statistics: EnergyStatistics
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (delta_1, delta_2) in enumerate(PATCH_OFFSETS):
        rows.append(
            {
                "band": band,
                "offset_index": index,
                "delta_row": int(delta_1),
                "delta_column": int(delta_2),
                "signed_correlation": float(statistics.signed_correlation[index]),
                "correlation_of_squares": float(statistics.square_correlation[index]),
                "fourth_order_excess": float(statistics.fourth_order_excess[index]),
            }
        )
    off_center = np.arange(9) != CENTER_INDEX
    summary = {
        "band": band,
        "mean_abs_signed_correlation_off_center": float(
            np.mean(np.abs(statistics.signed_correlation[off_center]))
        ),
        "mean_correlation_of_squares_off_center": float(
            np.mean(statistics.square_correlation[off_center])
        ),
        "mean_kappa_off_center": float(
            np.mean(statistics.fourth_order_excess[off_center])
        ),
        "median_kappa_off_center": float(
            np.median(statistics.fourth_order_excess[off_center])
        ),
        "max_kappa_off_center": float(
            np.max(statistics.fourth_order_excess[off_center])
        ),
    }
    return rows, summary


def plot_statistic_heatmaps(
    values: dict[str, np.ndarray], title: str, colorbar_label: str, path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 5.7), constrained_layout=True)
    maximum = max(float(np.max(np.abs(value))) for value in values.values())
    maximum = max(maximum, 1e-15)
    for axis, band in zip(axes.flat, BAND_NAMES, strict=True):
        matrix = values[band].reshape(3, 3)
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axis.set_title(band)
        axis.set_xticks((0, 1, 2), ("-1", "0", "+1"))
        axis.set_yticks((0, 1, 2), ("-1", "0", "+1"))
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    figure.colorbar(image, ax=axes, shrink=0.82, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_kappa_fit(rows: list[dict[str, object]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for band in BAND_NAMES:
        selected = [row for row in rows if row["band"] == band]
        axis.scatter(
            [row["kappa_empirical"] for row in selected],
            [row["kappa_gsm"] for row in selected],
            label=band,
            s=35,
        )
    limits = axis.get_xlim()
    lower = min(limits[0], axis.get_ylim()[0])
    upper = max(limits[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Empirical fourth-order excess")
    axis.set_ylabel("GSM-predicted fourth-order excess")
    axis.set_title("Does the two-state GSM reproduce fourth-order dependence?")
    axis.legend(ncol=2)
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_mse(
    metrics: list[dict[str, object]],
    path: Path,
    per_band_path: Path,
    *,
    image_size: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    for axis, scope in zip(axes, ("global", "level_1", "level_2"), strict=True):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in metrics
                    if row["split"] == "test"
                    and row["scope"] == scope
                    and row["method"] == method
                ),
                key=lambda row: float(row["time"]),
            )
            axis.plot(
                [row["time"] for row in selected],
                [row["mse"] for row in selected],
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=2.5,
            )
        axis.set_title(scope.replace("_", " ").title())
        axis.set_xlabel("t (0=noise, 1=clean)")
        axis.set_ylabel("Test MSE on center coefficient")
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=8)
    figure.suptitle(f"Local wavelet denoising on MNIST-{image_size}")
    figure.savefig(path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), constrained_layout=True)
    for axis, band in zip(axes.flat, BAND_NAMES, strict=True):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in metrics
                    if row["split"] == "test"
                    and row["scope"] == band
                    and row["method"] == method
                ),
                key=lambda row: float(row["time"]),
            )
            axis.plot(
                [row["time"] for row in selected],
                [row["mse"] for row in selected],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(band)
        axis.set_xlabel("t")
        axis.set_ylabel("Test MSE")
        axis.grid(alpha=0.2)
    axes[0, 2].legend(fontsize=7)
    figure.suptitle("Test denoising risk by wavelet subband")
    figure.savefig(per_band_path, dpi=180)
    plt.close(figure)


def plot_context_gain(summary: list[dict[str, object]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    for axis, scope in zip(axes, ("global", "level_1", "level_2"), strict=True):
        selected = sorted(
            (row for row in summary if row["scope"] == scope),
            key=lambda row: float(row["time"]),
        )
        gains = [
            np.nan if row["gain_context"] == "N/A" else float(row["gain_context"])
            for row in selected
        ]
        axis.plot(
            [row["time"] for row in selected],
            gains,
            color=METHOD_COLORS["gsm_local"],
            marker="o",
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(scope.replace("_", " ").title())
        axis.set_xlabel("t")
        axis.set_ylabel("(MSE GSM self - MSE GSM local) / MSE GSM self")
        axis.grid(alpha=0.2)
    figure.suptitle("Relative gain supplied by the eight neighboring coefficients")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_generalization_gap(metrics: list[dict[str, object]], path: Path) -> None:
    lookup = {
        (row["scope"], row["time"], row["method"], row["split"]): float(row["mse"])
        for row in metrics
    }
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    times = sorted({float(row["time"]) for row in metrics})
    for method in METHODS:
        train_mse = [lookup[("global", time, method, "train")] for time in times]
        test_mse = [lookup[("global", time, method, "test")] for time in times]
        axes[0].plot(
            times,
            test_mse,
            label=f"{METHOD_LABELS[method]} test",
            color=METHOD_COLORS[method],
        )
        axes[0].plot(
            times,
            train_mse,
            label=f"{METHOD_LABELS[method]} train",
            color=METHOD_COLORS[method],
            linestyle="--",
            alpha=0.75,
        )
        gap = [
            lookup[("global", time, method, "test")]
            - lookup[("global", time, method, "train")]
            for time in times
        ]
        axes[1].plot(times, gap, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Global MSE")
    axes[0].set_title("Train and test risks")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Test MSE - train MSE")
    axes[1].set_title("Generalization gap")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    figure.suptitle("Train uses the exact patches employed to fit each subband model")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_latent_histograms(
    latent_samples: dict[tuple[str, float], np.ndarray], path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 41)
    for axis, band in zip(axes.flat, BAND_NAMES, strict=True):
        for time in LATENT_TIMES:
            values = latent_samples[(band, time)]
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            axis.plot(centers, hist, label=f"t={time:g}")
        axis.set_title(band)
        axis.set_xlabel("P(high-scale state | local observation)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
    axes[0, 2].legend(fontsize=7)
    figure.suptitle("Evolution of the local latent-state posterior")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_latent_examples(examples: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    categories = ("low", "ambiguous", "high")
    figure, axes = plt.subplots(6, 3, figsize=(7.5, 14.0), constrained_layout=True)
    for row, band in enumerate(BAND_NAMES):
        maximum = max(float(np.abs(examples[band][name]).max()) for name in categories)
        maximum = max(maximum, 1e-12)
        for column, category in enumerate(categories):
            axis = axes[row, column]
            axis.imshow(
                examples[band][category], cmap="coolwarm", vmin=-maximum, vmax=maximum
            )
            if row == 0:
                axis.set_title(category.capitalize())
            if column == 0:
                axis.set_ylabel(band)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("Clean wavelet patches selected by their fitted GSM responsibility")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def paired_interval(differences: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def entropy_binary(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def gain_summary_row(
    scope: str,
    time: float,
    risks: dict[str, float],
    *,
    tolerance: float = 1e-14,
) -> dict[str, object]:
    def relative_gain(baseline: str, candidate: str) -> float | str:
        denominator = risks[baseline]
        if denominator <= tolerance:
            return "N/A"
        return (denominator - risks[candidate]) / denominator

    return {
        "scope": scope,
        "time": float(time),
        "gain_context": relative_gain("gsm_self", "gsm_local"),
        "gain_self_vs_wiener_matched_self": relative_gain(
            "wiener_matched_self", "gsm_self"
        ),
        "gain_local_vs_wiener_matched_local": relative_gain(
            "wiener_matched_local", "gsm_local"
        ),
        "gain_local_vs_wiener_empirical_local": relative_gain(
            "wiener_empirical_local", "gsm_local"
        ),
    }


def run_model_sanity_checks(
    models: dict[str, GSMParameters],
    statistics_by_band: dict[str, EnergyStatistics],
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Numerically verify endpoint, responsibility, and collapsed-state identities."""
    generator = np.random.default_rng(seed)
    results: dict[str, dict[str, float]] = {}
    for band in BAND_NAMES:
        model = models[band]
        statistics = statistics_by_band[band]
        observations = generator.standard_normal((64, 9))
        predictions_t0 = predict_all(
            observations, 0.0, statistics.mean, statistics.covariance, model
        )
        predictions_t1 = predict_all(
            observations, 1.0, statistics.mean, statistics.covariance, model
        )
        _, gamma_local_t0 = gsm_local_posterior_center(
            observations, 0.0, model, return_responsibilities=True
        )
        _, gamma_self_t0 = gsm_self_posterior_center(
            observations[:, CENTER_INDEX], 0.0, model, return_responsibilities=True
        )
        _, gamma_local_mid = gsm_local_posterior_center(
            observations, 0.5, model, return_responsibilities=True
        )
        collapsed = GSMParameters(
            mean=model.mean,
            covariance=model.covariance,
            mixing_weights=np.asarray([0.4, 0.6]),
            scales=np.ones(2),
            log_likelihood=0.0,
            iterations=0,
            converged=True,
            scale_floor_hits=0,
            likelihood_decreases=0,
            condition_number=model.condition_number,
            restart=0,
        )
        collapsed_prediction = gsm_local_posterior_center(
            observations, 0.5, collapsed
        )
        matched_prediction = gaussian_posterior_center(
            observations, 0.5, model.mean, model.covariance
        )
        expected_mean = statistics.mean[CENTER_INDEX]
        values = {
            "weight_sum_error": abs(float(model.mixing_weights.sum()) - 1.0),
            "mean_scale_error": abs(float(np.dot(model.mixing_weights, model.scales)) - 1.0),
            "gamma_local_t0_prior_error": float(
                np.max(np.abs(gamma_local_t0 - model.mixing_weights))
            ),
            "gamma_self_t0_prior_error": float(
                np.max(np.abs(gamma_self_t0 - model.mixing_weights))
            ),
            "gamma_sum_error": float(
                np.max(np.abs(gamma_local_mid.sum(axis=1) - 1.0))
            ),
            "posterior_mean_t0_error": max(
                float(np.max(np.abs(prediction - expected_mean)))
                for prediction in predictions_t0.values()
            ),
            "posterior_mean_t1_error": max(
                float(np.max(np.abs(prediction - observations[:, CENTER_INDEX])))
                for prediction in predictions_t1.values()
            ),
            "collapsed_gsm_wiener_error": float(
                np.max(np.abs(collapsed_prediction - matched_prediction))
            ),
            "minimum_covariance_eigenvalue": float(
                np.linalg.eigvalsh(model.covariance).min()
            ),
        }
        error_keys = [key for key in values if key.endswith("error")]
        if max(values[key] for key in error_keys) > 1e-9:
            raise RuntimeError(f"Posterior sanity check failed for {band}: {values}")
        if values["minimum_covariance_eigenvalue"] <= 0.0:
            raise RuntimeError(f"Non-positive GSM covariance for {band}.")
        results[band] = values
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    compute_dtype = torch.float32 if args.compute_dtype == "float32" else torch.float64
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    start = perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "normalization": "MNIST float pixels in [0,1]",
        "source_image_size": 28,
        "resize": (
            "none"
            if args.image_size == 28
            else "torchvision PIL bilinear resize with antialiasing"
        ),
        "wavelet": "haar",
        "levels": 2,
        "mode": "periodization",
        "patch_boundary": "valid centers only; no wrapping and no padding",
        "patch_order": PATCH_OFFSETS.tolist(),
        "center_index": CENTER_INDEX,
        "path": "Y_t = t Y_0 + (1-t) Z",
        "methods": list(METHODS),
        "device_resolved": str(device),
        "bulk_compute_dtype": str(compute_dtype),
        "parameter_and_reduction_dtype": "torch.float64",
        "ci": "paired image-level normal approximation, mean +/- 1.96 SE",
        "aggregation_weight": "number of valid 3x3 center positions in each subband",
    }
    save_json(args.out_dir / "config.json", config)

    print(f"device: {device}, bulk compute dtype: {compute_dtype}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Loading MNIST at {args.image_size}x{args.image_size}...")
    train = load_mnist(
        args.data_root,
        "train",
        image_size=args.image_size,
        max_images=args.num_train_images,
        download=args.download,
    )
    test = load_mnist(
        args.data_root,
        "test",
        image_size=args.image_size,
        max_images=args.num_test_images,
        download=args.download,
    )
    transform_audit = audit_transform(
        train.images, noise_samples=args.white_noise_samples, seed=args.seed
    )
    save_json(args.out_dir / "transform_audit.json", transform_audit)
    if transform_audit["max_relative_reconstruction_error"] > 2e-6:
        raise RuntimeError("Haar reconstruction audit failed.")
    if transform_audit["max_relative_parseval_error"] > 2e-6:
        raise RuntimeError("Haar Parseval audit failed.")
    train_wavelets = analyze_haar_level2(train.images)
    test_wavelets = analyze_haar_level2(test.images)

    selections: dict[tuple[str, str], PatchSelection] = {}
    selection_arrays: dict[str, np.ndarray] = {}
    for band_index, band in enumerate(BAND_NAMES):
        shape = train_wavelets.bands[band].shape[1:]
        train_selection = sample_valid_centers(
            len(train.images),
            shape,
            args.patches_per_image,
            seed=args.seed + 100 * band_index + 1,
        )
        test_selection = sample_valid_centers(
            len(test.images),
            shape,
            args.patches_per_image,
            seed=args.seed + 100 * band_index + 2,
        )
        selections[("train", band)] = train_selection
        selections[("test", band)] = test_selection
        selection_arrays.update(selection_to_arrays(f"train_{band}", train_selection))
        selection_arrays.update(selection_to_arrays(f"test_{band}", test_selection))
    np.savez_compressed(args.out_dir / "patch_selections.npz", **selection_arrays)

    energy_rows: list[dict[str, object]] = []
    energy_summary_rows: list[dict[str, object]] = []
    gsm_rows: list[dict[str, object]] = []
    restart_rows: list[dict[str, object]] = []
    kappa_rows: list[dict[str, object]] = []
    statistics_by_band: dict[str, EnergyStatistics] = {}
    models: dict[str, GSMParameters] = {}
    latent_examples: dict[str, dict[str, np.ndarray]] = {}

    for band_index, band in enumerate(BAND_NAMES):
        print(f"\n=== {band}: clean statistics and GSM fit ===", flush=True)
        patches = extract_patches(
            train_wavelets.bands[band], selections[("train", band)]
        ).astype(np.float64)
        statistics = compute_energy_statistics(patches)
        statistics_by_band[band] = statistics
        rows, summary = statistics_rows(band, statistics)
        energy_rows.extend(rows)
        energy_summary_rows.append(summary)
        model_path = model_dir / f"gsm_{band}.npz"
        if args.reuse_fits and model_path.exists():
            print(f"Reusing {model_path}", flush=True)
            model = load_gsm(str(model_path))
            histories: list[dict[str, float | int | bool]] = []
        else:
            model, histories = fit_two_state_gsm_torch(
                patches,
                device=device,
                compute_dtype=compute_dtype,
                num_restarts=args.num_restarts,
                max_iterations=args.max_em_iterations,
                tolerance=args.em_tolerance,
                covariance_epsilon=args.covariance_epsilon,
                scale_floor=args.scale_floor,
                seed=args.seed + 1000 + band_index * args.num_restarts,
            )
            save_gsm(str(model_path), model)
        models[band] = model
        for history in histories:
            restart_rows.append({"band": band, **history})
        covariance_difference = np.linalg.norm(model.covariance - statistics.covariance)
        covariance_difference /= max(np.linalg.norm(statistics.covariance), 1e-15)
        gsm_rows.append(
            {
                "band": band,
                "pi_low": float(model.mixing_weights[0]),
                "pi_high": float(model.mixing_weights[1]),
                "s_low": float(model.scales[0]),
                "s_high": float(model.scales[1]),
                "var_s": model.scale_variance,
                "log_likelihood": model.log_likelihood,
                "log_likelihood_per_patch": model.log_likelihood / len(patches),
                "iterations": model.iterations,
                "converged": model.converged,
                "scale_floor_hits": model.scale_floor_hits,
                "scale_floor_hit": model.scale_floor_hits > 0,
                "state_collapse": model.state_collapse,
                "condition_number_sigma": model.condition_number,
                "relative_sigma_empirical_difference": float(covariance_difference),
            }
        )
        predicted_kappa = gsm_fourth_order_excess(
            model.covariance, model.mixing_weights, model.scales
        )
        for offset_index, (delta_1, delta_2) in enumerate(PATCH_OFFSETS):
            kappa_rows.append(
                {
                    "band": band,
                    "offset_index": offset_index,
                    "delta_row": int(delta_1),
                    "delta_column": int(delta_2),
                    "kappa_empirical": float(statistics.fourth_order_excess[offset_index]),
                    "kappa_gsm": float(predicted_kappa[offset_index]),
                    "difference": float(
                        predicted_kappa[offset_index]
                        - statistics.fourth_order_excess[offset_index]
                    ),
                }
            )
        gamma_clean, _ = responsibilities(
            patches,
            model.mean,
            model.covariance,
            model.mixing_weights,
            model.scales,
        )
        high = gamma_clean[:, 1]
        latent_examples[band] = {
            "low": patches[int(np.argmin(high))].reshape(3, 3),
            "ambiguous": patches[int(np.argmin(np.abs(high - 0.5)))].reshape(3, 3),
            "high": patches[int(np.argmax(high))].reshape(3, 3),
        }
        write_csv(args.out_dir / "energy_statistics.csv", energy_rows)
        write_csv(args.out_dir / "energy_summary.csv", energy_summary_rows)
        write_csv(args.out_dir / "gsm_parameters.csv", gsm_rows)
        write_csv(args.out_dir / "gsm_restarts.csv", restart_rows)
        write_csv(args.out_dir / "kappa_empirical_vs_gsm.csv", kappa_rows)
        print(
            f"pi={model.mixing_weights}, scales={model.scales}, "
            f"LL/N={model.log_likelihood / len(patches):.5f}, "
            f"floor_hits={model.scale_floor_hits}",
            flush=True,
        )
        del patches, gamma_clean
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    plot_statistic_heatmaps(
        {band: statistics_by_band[band].signed_correlation for band in BAND_NAMES},
        "Signed dependence between center and neighboring coefficients",
        "Corr(X_center, X_offset)",
        args.out_dir / "signed_correlation_heatmaps.png",
    )
    plot_statistic_heatmaps(
        {band: statistics_by_band[band].square_correlation for band in BAND_NAMES},
        "Dependence between squared wavelet coefficients",
        "Corr(X_center^2, X_offset^2)",
        args.out_dir / "square_correlation_heatmaps.png",
    )
    plot_statistic_heatmaps(
        {band: statistics_by_band[band].fourth_order_excess for band in BAND_NAMES},
        "Fourth-order excess beyond a Gaussian with matched covariance",
        "kappa(center, offset)",
        args.out_dir / "fourth_order_excess_heatmaps.png",
    )
    plot_kappa_fit(kappa_rows, args.out_dir / "kappa_empirical_vs_gsm.png")
    plot_latent_examples(latent_examples, args.out_dir / "clean_latent_patch_examples.png")
    sanity_checks = run_model_sanity_checks(
        models, statistics_by_band, seed=args.seed + 20_000
    )
    save_json(args.out_dir / "sanity_checks.json", sanity_checks)

    print("\n=== Denoising evaluation ===", flush=True)
    clean_patches: dict[tuple[str, str], torch.Tensor] = {}
    noise_patches: dict[tuple[str, str], torch.Tensor] = {}
    for split_index, (split, wavelets, num_images) in enumerate(
        (("train", train_wavelets, len(train.images)), ("test", test_wavelets, len(test.images)))
    ):
        for band_index, band in enumerate(BAND_NAMES):
            selection = selections[(split, band)]
            clean_patches[(split, band)] = torch.as_tensor(
                extract_patches(wavelets.bands[band], selection),
                device=device,
                dtype=compute_dtype,
            )
            generator = np.random.default_rng(
                args.seed + 10_000 + split_index * 1000 + band_index
            )
            noise_field = generator.standard_normal(
                (num_images, *selection.band_shape), dtype=np.float32
            )
            noise_patches[(split, band)] = torch.as_tensor(
                extract_patches(noise_field, selection),
                device=device,
                dtype=compute_dtype,
            )
    statistics_tensors = {
        band: (
            torch.as_tensor(
                statistics_by_band[band].mean, device=device, dtype=compute_dtype
            ),
            torch.as_tensor(
                statistics_by_band[band].covariance,
                device=device,
                dtype=compute_dtype,
            ),
        )
        for band in BAND_NAMES
    }

    metrics: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    latent_rows: list[dict[str, object]] = []
    latent_samples: dict[tuple[str, float], np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for time in args.times:
        print(f"t={time:.2f}", flush=True)
        per_image_errors: dict[tuple[str, str, str], np.ndarray] = {}
        aggregate_sums: dict[tuple[str, str, str], np.ndarray] = {}
        aggregate_weights: dict[str, float] = {"level_1": 0.0, "level_2": 0.0, "global": 0.0}

        for band in BAND_NAMES:
            band_height, band_width = selections[("test", band)].band_shape
            band_weight = float((band_height - 2) * (band_width - 2))
            scopes = (band_level(band), "global")
            for scope in scopes:
                aggregate_weights[scope] += band_weight
            for split, num_images in (("train", len(train.images)), ("test", len(test.images))):
                clean = clean_patches[(split, band)]
                noisy = time * clean + (1.0 - time) * noise_patches[(split, band)]
                empirical_mean, empirical_covariance = statistics_tensors[band]
                predictions = predict_all_tensor(
                    noisy,
                    time,
                    empirical_mean,
                    empirical_covariance,
                    models[band],
                )
                target = clean[:, CENTER_INDEX]
                for method, prediction in predictions.items():
                    squared_error = (prediction - target) ** 2
                    image_error = (
                        squared_error.reshape(num_images, args.patches_per_image)
                        .mean(dim=1)
                        .to(torch.float64)
                        .cpu()
                        .numpy()
                    )
                    per_image_errors[(split, band, method)] = image_error
                    metrics.append(
                        {
                            "split": split,
                            "scope": band,
                            "time": float(time),
                            "method": method,
                            "mse": float(squared_error.mean(dtype=torch.float64).item()),
                            "num_images": num_images,
                            "num_patches": len(clean),
                        }
                    )
                    for scope in scopes:
                        key = split, scope, method
                        if key not in aggregate_sums:
                            aggregate_sums[key] = np.zeros(num_images, dtype=np.float64)
                        aggregate_sums[key] += band_weight * image_error

            test_errors = {
                method: per_image_errors[("test", band, method)] for method in METHODS
            }
            for baseline, candidate, contrast in CI_COMPARISONS:
                mean, lower, upper = paired_interval(
                    test_errors[baseline] - test_errors[candidate]
                )
                confidence_rows.append(
                    {
                        "scope": band,
                        "time": float(time),
                        "contrast": contrast,
                        "baseline": baseline,
                        "candidate": candidate,
                        "mean_risk_reduction": mean,
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "num_images": len(test.images),
                    }
                )
            summary_rows.append(
                gain_summary_row(
                    band,
                    float(time),
                    {method: float(test_errors[method].mean()) for method in METHODS},
                )
            )

            if any(abs(time - latent_time) < 1e-12 for latent_time in LATENT_TIMES):
                clean = clean_patches[("test", band)]
                noisy = time * clean + (1.0 - time) * noise_patches[("test", band)]
                _, gamma_local = gsm_local_posterior_center_tensor(
                    noisy, time, models[band], return_responsibilities=True
                )
                _, gamma_self = gsm_self_posterior_center_tensor(
                    noisy[:, CENTER_INDEX], time, models[band], return_responsibilities=True
                )
                for posterior_name, gamma in (("local", gamma_local), ("self", gamma_self)):
                    high = gamma[:, 1].to(torch.float64)
                    latent_rows.append(
                        {
                            "band": band,
                            "time": float(time),
                            "posterior": posterior_name,
                            "mean_gamma_high": float(high.mean().item()),
                            "variance_gamma_high": float(high.var(unbiased=False).item()),
                            "mean_entropy": float(
                                entropy_binary(high.cpu().numpy()).mean()
                            ),
                        }
                    )
                latent_samples[(band, float(time))] = (
                    gamma_local[: min(20_000, len(gamma_local)), 1]
                    .to(torch.float64)
                    .cpu()
                    .numpy()
                )

        for scope in ("level_1", "level_2", "global"):
            for split, num_images in (("train", len(train.images)), ("test", len(test.images))):
                for method in METHODS:
                    values = aggregate_sums[(split, scope, method)] / aggregate_weights[scope]
                    per_image_errors[(split, scope, method)] = values
                    metrics.append(
                        {
                            "split": split,
                            "scope": scope,
                            "time": float(time),
                            "method": method,
                            "mse": float(values.mean()),
                            "num_images": num_images,
                            "num_patches": "weighted subband aggregate",
                        }
                    )
            test_errors = {
                method: per_image_errors[("test", scope, method)] for method in METHODS
            }
            for baseline, candidate, contrast in CI_COMPARISONS:
                mean, lower, upper = paired_interval(
                    test_errors[baseline] - test_errors[candidate]
                )
                confidence_rows.append(
                    {
                        "scope": scope,
                        "time": float(time),
                        "contrast": contrast,
                        "baseline": baseline,
                        "candidate": candidate,
                        "mean_risk_reduction": mean,
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "num_images": len(test.images),
                    }
                )
            summary_rows.append(
                gain_summary_row(
                    scope,
                    float(time),
                    {method: float(test_errors[method].mean()) for method in METHODS},
                )
            )
        write_csv(args.out_dir / "denoising_metrics.csv", metrics)
        write_csv(args.out_dir / "confidence_intervals.csv", confidence_rows)
        write_csv(args.out_dir / "latent_posterior_statistics.csv", latent_rows)
        write_csv(args.out_dir / "summary.csv", summary_rows)

    plot_mse(
        metrics,
        args.out_dir / "test_mse_global_and_levels.png",
        args.out_dir / "test_mse_by_subband.png",
        image_size=args.image_size,
    )
    plot_context_gain(summary_rows, args.out_dir / "context_gain.png")
    plot_generalization_gap(metrics, args.out_dir / "generalization_gap.png")
    if all((band, time) in latent_samples for band in BAND_NAMES for time in LATENT_TIMES):
        plot_latent_histograms(
            latent_samples, args.out_dir / "latent_posterior_histograms.png"
        )

    selected_time = min(args.times, key=lambda value: abs(value - 0.75))
    global_summary = next(
        row
        for row in summary_rows
        if row["scope"] == "global" and row["time"] == selected_time
    )
    report = f"""# Local wavelet GSM experiment

## Protocol

- MNIST at {args.image_size}x{args.image_size} in `[0,1]`.
- Resize: {config['resize']}.
- Orthogonal Haar DWT, two levels, periodization.
- Valid 3x3 neighborhoods only: no patch wrapping or padding.
- {args.patches_per_image} fixed patches per image and subband.
- Flow-Matching corruption `Y_t = t Y_0 + (1-t) Z`; no velocity field or generation.
- The GSM mean is fixed to the empirical patch mean.

## Numerical audit

- Maximum relative reconstruction error: {transform_audit['max_relative_reconstruction_error']:.3e}
- Maximum relative Parseval error: {transform_audit['max_relative_parseval_error']:.3e}
- White-noise mean coefficient variance: {transform_audit['white_noise_variance_mean']:.6f}
- White-noise off-diagonal covariance RMS: {transform_audit['white_noise_off_diagonal_rms']:.3e}

## GSM fits

See `gsm_parameters.csv`. A scale-floor hit is reported rather than hidden, and a
near equality of the two scales is marked as state collapse. `Wiener matched self`
is included so that marginal non-Gaussianity is not confounded with access to the
eight neighboring coefficients.

## Denoising snapshot

At the predeclared grid time closest to `t=0.75` (`t={selected_time:g}`), the global
relative gains are:

- GSM self vs matched self Wiener: {global_summary['gain_self_vs_wiener_matched_self']}
- GSM local vs GSM self (context): {global_summary['gain_context']}
- GSM local vs matched local Wiener: {global_summary['gain_local_vs_wiener_matched_local']}
- GSM local vs empirical local Wiener: {global_summary['gain_local_vs_wiener_empirical_local']}

Positive values indicate lower test risk for the GSM method. Full curves, paired
image-level confidence intervals, fourth-order diagnostics, and latent-state
statistics are stored alongside this report.

## Limitations

MNIST contains many nearly zero background patches, which can make low/high scale
separation unusually easy. It is also not spatially stationary: pooling positions
can partly encode center-versus-periphery variation. Positive fourth-order excess
does not prove that this two-state GSM is the correct data model. A small denoising
train-test gap would not establish absence of memorization in a future generator.
"""
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    save_json(
        args.out_dir / "run_summary.json",
        {
            "elapsed_seconds": perf_counter() - start,
            "num_train_images": len(train.images),
            "num_test_images": len(test.images),
            "num_fit_patches_per_band": len(train.images) * args.patches_per_image,
            "num_test_patches_per_band": len(test.images) * args.patches_per_image,
        },
    )
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak CUDA tensor memory: {peak_gib:.2f} GiB", flush=True)
    print(f"\nResults written to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
