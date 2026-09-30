"""Reconstruct complete MNIST images from previously fitted local GSMs.

Inputs: a completed ``run_experiment`` directory and official MNIST test
images. Outputs: unclipped image MSE, paired gains and reconstruction plots.
This stage does not refit the detail-band GSMs. Example::

    python -m wavelet_gsm_local.run_image_denoising --source-dir wavelet_gsm_local/results/mnist64_haar2 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wavelet-gsm-image")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_gsm_local.estimators import gaussian_self_posterior_center
from wavelet_gsm_local.gsm import GSMParameters, load_gsm
from wavelet_gsm_local.image_denoising import (
    IMAGE_METHODS,
    gaussian_vector_posterior_tensor,
    predict_detail_band_tensor,
)
from wavelet_gsm_local.patches import CENTER_INDEX, PatchSelection, extract_patches
from wavelet_gsm_local.statistics import EnergyStatistics, compute_energy_statistics
from wavelet_gsm_local.wavelets import (
    BAND_NAMES,
    HaarLevel2Batch,
    analyze_haar_level2,
    load_mnist,
    synthesize_haar_level2,
)


METHOD_LABELS = {
    "wiener_matched_self": "Wiener matched self",
    "wiener_matched_local": "Wiener matched local",
    "wiener_empirical_local": "Wiener empirical local",
    "gsm_self": "GSM self",
    "gsm_local": "GSM local",
}
METHOD_COLORS = {
    "wiener_matched_self": "#9467bd",
    "wiener_matched_local": "#377eb8",
    "wiener_empirical_local": "#111111",
    "gsm_self": "#ff7f00",
    "gsm_local": "#4daf4a",
}
MAIN_TIMES = (0.60, 0.70, 0.80, 0.85, 0.90)
MAGNITUDE_DIAGNOSTIC_TIMES = (0.70, 0.80, 0.85, 0.90)
MAGNITUDE_METHODS = (
    "wiener_matched_local",
    "gsm_self",
    "gsm_local",
)
DEFAULT_MAGNITUDE_BIN_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf"))
CONTRASTS = (
    (
        "wiener_matched_self",
        "wiener_matched_local",
        "gaussian_context",
        "Gaussian context: matched self to matched local",
    ),
    ("gsm_self", "gsm_local", "gsm_context", "GSM context: self to local"),
    (
        "wiener_matched_local",
        "gsm_local",
        "non_gaussianity",
        "Non-Gaussianity: matched local Wiener to local GSM",
    ),
    (
        "wiener_empirical_local",
        "wiener_matched_local",
        "structured_covariance",
        "Covariance estimate: empirical to GSM-matched Wiener",
    ),
)


def parse_times(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("main plotting times must lie strictly in (0,1)")
    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("wavelet_gsm_local/results/mnist28_haar2"),
        help="Completed coefficient-level experiment containing models and selections.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--times",
        type=parse_times,
        default=MAIN_TIMES,
        help="Comma-separated main times; t=0 and t=1 are added only for audits.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-visual-images", type=int, default=5)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_visual_images < 1:
        parser.error("batch-size and num-visual-images must be positive")
    if args.max_test_images is not None and args.max_test_images < 1:
        parser.error("max-test-images must be positive")
    if args.out_dir is None:
        args.out_dir = args.source_dir / "full_image_denoising"
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_source_config(source_dir: Path) -> dict[str, object]:
    path = source_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source configuration: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"wavelet": "haar", "levels": 2, "mode": "periodization"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Source configuration mismatch for {key}: {config.get(key)!r}")
    accepted_normalizations = {
        "native MNIST float pixels in [0,1]",
        "MNIST float pixels in [0,1]",
    }
    if config.get("normalization") not in accepted_normalizations:
        raise ValueError(
            "Source configuration uses an unsupported normalization: "
            f"{config.get('normalization')!r}"
        )
    image_size = int(config.get("image_size", 28))
    if image_size not in (28, 64):
        raise ValueError(f"Unsupported source image size: {image_size}")
    return config


def load_selection(
    arrays: np.lib.npyio.NpzFile,
    prefix: str,
    band_shape: tuple[int, int],
    *,
    max_images: int | None = None,
) -> PatchSelection:
    image_indices = arrays[f"{prefix}_image"].copy()
    rows = arrays[f"{prefix}_row"].copy()
    columns = arrays[f"{prefix}_column"].copy()
    if max_images is not None:
        keep = image_indices < max_images
        image_indices = image_indices[keep]
        rows = rows[keep]
        columns = columns[keep]
    return PatchSelection(image_indices, rows, columns, band_shape)


def load_models(source_dir: Path) -> dict[str, GSMParameters]:
    models: dict[str, GSMParameters] = {}
    for band in BAND_NAMES:
        path = source_dir / "models" / f"gsm_{band}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fitted model {path}. Complete the coefficient experiment first."
            )
        models[band] = load_gsm(str(path))
    return models


def empirical_patch_statistics(
    wavelets: HaarLevel2Batch,
    selections_path: Path,
    split: str,
    *,
    max_images: int | None = None,
) -> dict[str, EnergyStatistics]:
    results: dict[str, EnergyStatistics] = {}
    with np.load(selections_path) as arrays:
        for band in BAND_NAMES:
            band_values = wavelets.bands[band]
            selection = load_selection(
                arrays,
                f"{split}_{band}",
                band_values.shape[1:],
                max_images=max_images,
            )
            results[band] = compute_energy_statistics(
                extract_patches(band_values, selection)
            )
    return results


def fit_approximation_moments(approximation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = approximation.reshape(len(approximation), -1).astype(np.float64)
    mean = flat.mean(axis=0)
    centered = flat - mean
    covariance = centered.T @ centered / len(centered)
    return mean, 0.5 * (covariance + covariance.T)


def denoise_detail_band_batched(
    noisy_band: np.ndarray,
    time: float,
    statistics: EnergyStatistics,
    parameters: GSMParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, np.ndarray]:
    results = {
        method: np.empty(noisy_band.shape, dtype=np.float32) for method in IMAGE_METHODS
    }
    empirical_mean = torch.as_tensor(statistics.mean, device=device, dtype=dtype)
    empirical_covariance = torch.as_tensor(
        statistics.covariance, device=device, dtype=dtype
    )
    noisy_tensor = torch.as_tensor(
        np.ascontiguousarray(noisy_band), device=device, dtype=dtype
    )
    for start in range(0, len(noisy_band), batch_size):
        stop = min(start + batch_size, len(noisy_band))
        predictions = predict_detail_band_tensor(
            noisy_tensor[start:stop],
            time,
            empirical_mean,
            empirical_covariance,
            parameters,
        )
        for method, prediction in predictions.items():
            results[method][start:stop] = prediction.float().cpu().numpy()
    return results


def border_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def plot_reconstructions(
    clean: np.ndarray,
    noisy: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    indices: np.ndarray,
    time: float,
    path: Path,
) -> None:
    columns = ("clean", "noisy", *IMAGE_METHODS)
    titles = ("Clean", "Noisy", *(METHOD_LABELS[method] for method in IMAGE_METHODS))
    figure, axes = plt.subplots(
        len(indices), len(columns), figsize=(2.05 * len(columns), 2.0 * len(indices)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(indices), len(columns))
    for row, image_index in enumerate(indices):
        images = (
            clean[image_index],
            noisy[image_index],
            *(reconstructions[method][image_index] for method in IMAGE_METHODS),
        )
        for column, (axis, image) in enumerate(zip(axes[row], images, strict=True)):
            axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
            if row == 0:
                axis.set_title(titles[column], fontsize=9)
            if column == 0:
                axis.set_ylabel(f"test #{image_index}")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"Full-image wavelet denoising at t={time:g}\n"
        "Display clipped to [0,1]; every reported metric uses raw values",
        fontsize=12,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_image_mse(rows: list[dict[str, object]], times: tuple[float, ...], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    for method in IMAGE_METHODS:
        selected = [
            row for row in rows if row["method"] == method and row["time"] in times
        ]
        selected.sort(key=lambda row: float(row["time"]))
        axis.plot(
            [row["time"] for row in selected],
            [row["mse"] for row in selected],
            marker="o",
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("t (0 = noise, 1 = clean)")
    axis.set_ylabel("Raw test MSE per image pixel")
    axis.set_title("Complete MNIST reconstruction from denoised wavelet coefficients")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_region_mse(
    rows: list[dict[str, object]], times: tuple[float, ...], path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for axis, region in zip(axes, ("interior", "border"), strict=True):
        for method in IMAGE_METHODS:
            selected = [
                row
                for row in rows
                if row["scope"] == "all_detail_bands"
                and row["region"] == region
                and row["method"] == method
                and row["time"] in times
            ]
            selected.sort(key=lambda row: float(row["time"]))
            axis.plot(
                [row["time"] for row in selected],
                [row["mse"] for row in selected],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(region.capitalize())
        axis.set_xlabel("t")
        axis.set_ylabel("MSE per detail coefficient")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Where does the full-image detail-band error come from?")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_relative_gains(
    rows: list[dict[str, object]], times: tuple[float, ...], path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for axis, (_, _, key, title) in zip(axes.flat, CONTRASTS, strict=True):
        selected = [row for row in rows if row["contrast"] == key and row["time"] in times]
        selected.sort(key=lambda row: float(row["time"]))
        x = np.asarray([float(row["time"]) for row in selected])
        mean = np.asarray([float(row["relative_reduction"]) for row in selected])
        lower = np.asarray([float(row["relative_ci95_low"]) for row in selected])
        upper = np.asarray([float(row["relative_ci95_high"]) for row in selected])
        axis.plot(x, mean, marker="o", color="#0072b2")
        axis.fill_between(x, lower, upper, color="#56b4e9", alpha=0.28)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("t")
        axis.set_ylabel("Relative paired MSE reduction")
        axis.grid(alpha=0.25)
    figure.suptitle("Positive values mean that the destination estimator has lower image MSE")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_covariance_diagnostic(rows: list[dict[str, object]], path: Path) -> None:
    positions = np.arange(len(BAND_NAMES))
    empirical = [
        next(float(row["relative_frobenius_error"]) for row in rows if row["band"] == band and row["covariance"] == "empirical_train")
        for band in BAND_NAMES
    ]
    gsm = [
        next(float(row["relative_frobenius_error"]) for row in rows if row["band"] == band and row["covariance"] == "gsm_fitted")
        for band in BAND_NAMES
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    width = 0.38
    axis.bar(positions - width / 2, empirical, width, label="Empirical train covariance")
    axis.bar(positions + width / 2, gsm, width, label="Fitted GSM covariance")
    axis.set_xticks(positions, BAND_NAMES)
    axis.set_ylabel("Relative Frobenius error to held-out test covariance")
    axis.set_title("Covariance diagnostic only: the test split is never used for fitting")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def magnitude_bin_labels(edges: np.ndarray) -> list[str]:
    labels: list[str] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        upper_text = "inf" if np.isinf(upper) else f"{upper:g}"
        labels.append(f"[{lower:g}, {upper_text})")
    return labels


def prepare_magnitude_bins(
    train_wavelets: HaarLevel2Batch,
    test_wavelets: HaarLevel2Batch,
) -> tuple[dict[str, float], np.ndarray, dict[str, np.ndarray], np.ndarray, str]:
    """Normalize clean magnitudes per train subband and assign fixed bins."""
    standard_deviations: dict[str, float] = {}
    for band in BAND_NAMES:
        scale = float(np.std(train_wavelets.bands[band], dtype=np.float64))
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"Invalid train-only standard deviation for {band}: {scale}")
        standard_deviations[band] = scale

    edges = np.asarray(DEFAULT_MAGNITUDE_BIN_EDGES, dtype=np.float64)

    def assign(current_edges: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        assignments: dict[str, np.ndarray] = {}
        counts = np.zeros(len(current_edges) - 1, dtype=np.int64)
        for band in BAND_NAMES:
            normalized = (
                np.abs(test_wavelets.bands[band].astype(np.float64, copy=False))
                / standard_deviations[band]
            )
            indices = np.searchsorted(
                current_edges[1:-1], normalized, side="right"
            ).astype(np.uint8)
            assignments[band] = indices
            counts += np.bincount(
                indices.reshape(-1), minlength=len(current_edges) - 1
            )
        return assignments, counts

    assignments, counts = assign(edges)
    strategy = "fixed"
    if np.any(counts == 0):
        # This fallback is deliberately deterministic and only used when the
        # explicit scientific bins cannot be estimated on the selected test set.
        samples: list[np.ndarray] = []
        for band in BAND_NAMES:
            values = (
                np.abs(test_wavelets.bands[band].astype(np.float64, copy=False))
                / standard_deviations[band]
            ).reshape(-1)
            stride = max(1, int(np.ceil(len(values) / 200_000)))
            samples.append(values[::stride])
        pooled = np.concatenate(samples)
        quantiles = np.quantile(pooled, (0.50, 0.75, 0.90, 0.97, 0.995))
        internal = np.unique(quantiles[quantiles > 0.0])
        edges = np.concatenate(([0.0], internal, [np.inf]))
        assignments, counts = assign(edges)
        strategy = "quantile_fallback_due_to_empty_fixed_bin"
    return standard_deviations, edges, assignments, counts, strategy


def format_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def plot_conditional_magnitude_mse(
    rows: list[dict[str, object]],
    times: tuple[float, ...],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for axis, time in zip(axes.flat, times, strict=True):
        time_rows = [row for row in rows if float(row["time"]) == time]
        counts = [
            int(row["count"])
            for row in time_rows
            if row["method"] == MAGNITUDE_METHODS[0]
        ]
        labels = [
            str(row["bin_label"])
            for row in time_rows
            if row["method"] == MAGNITUDE_METHODS[0]
        ]
        positions = np.arange(len(labels))
        for method in MAGNITUDE_METHODS:
            selected = [row for row in time_rows if row["method"] == method]
            axis.plot(
                positions,
                [float(row["mse"]) for row in selected],
                marker="o",
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_xticks(
            positions,
            [f"{label}\nn={format_count(count)}" for label, count in zip(labels, counts)],
            fontsize=8,
        )
        axis.set_title(f"t = {time:g}")
        axis.set_xlabel(r"Clean normalized magnitude  $|Y_{0,q}| / \sigma_b$")
        axis.set_ylabel("Conditional coefficient MSE (log scale)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "MNIST-64 detail coefficients: denoising risk conditioned on clean magnitude\n"
        r"Each $\sigma_b$ is estimated from the training split only"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_conditional_magnitude_gain(
    rows: list[dict[str, object]],
    times: tuple[float, ...],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for axis, time in zip(axes.flat, times, strict=True):
        selected = [row for row in rows if float(row["time"]) == time]
        positions = np.arange(len(selected))
        values = np.asarray([float(row["relative_gain"]) for row in selected])
        colors = np.where(values >= 0.0, "#4daf4a", "#d95f02")
        axis.bar(positions, values, color=colors)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(
            positions,
            [
                f"{row['bin_label']}\nn={format_count(int(row['count']))}"
                for row in selected
            ],
            fontsize=8,
        )
        axis.set_title(f"t = {time:g}")
        axis.set_xlabel(r"Clean normalized magnitude  $|Y_{0,q}| / \sigma_b$")
        axis.set_ylabel(r"$(R_W - R_{GSM}) / R_W$")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Relative gain of GSM local over matched local Wiener\n"
        "Positive bars favor GSM local; counts expose rare magnitude tails"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    start_time = perf_counter()
    source_config = load_source_config(args.source_dir)
    device = resolve_device(args.device)
    compute_dtype = torch.float32 if args.compute_dtype == "float32" else torch.float64
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_train_count = int(source_config["num_train_images"])
    source_test_count = int(source_config["num_test_images"])
    image_size = int(source_config.get("image_size", 28))
    test_count = source_test_count
    if args.max_test_images is not None:
        test_count = min(test_count, args.max_test_images)
    data_root = args.data_root or Path(str(source_config["data_root"]))
    noise_seed = args.noise_seed
    if noise_seed is None:
        noise_seed = int(source_config["seed"]) + 30_000

    print(f"device: {device}, detail compute dtype: {compute_dtype}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(
        f"Loading the same MNIST-{image_size} splits as the coefficient experiment...",
        flush=True,
    )
    train = load_mnist(
        data_root,
        "train",
        image_size=image_size,
        max_images=source_train_count,
        download=args.download,
    )
    test = load_mnist(
        data_root,
        "test",
        image_size=image_size,
        max_images=test_count,
        download=args.download,
    )
    train_wavelets = analyze_haar_level2(train.images)
    test_wavelets = analyze_haar_level2(test.images)
    models = load_models(args.source_dir)

    magnitude_diagnostic_enabled = image_size == 64
    magnitude_standard_deviations: dict[str, float] = {}
    magnitude_edges = np.asarray(DEFAULT_MAGNITUDE_BIN_EDGES, dtype=np.float64)
    magnitude_assignments: dict[str, np.ndarray] = {}
    magnitude_counts = np.zeros(len(magnitude_edges) - 1, dtype=np.int64)
    magnitude_bin_strategy = "disabled_for_non_64_resolution"
    if magnitude_diagnostic_enabled:
        (
            magnitude_standard_deviations,
            magnitude_edges,
            magnitude_assignments,
            magnitude_counts,
            magnitude_bin_strategy,
        ) = prepare_magnitude_bins(train_wavelets, test_wavelets)
        print(
            "Magnitude diagnostic bins: "
            f"{magnitude_bin_labels(magnitude_edges)}; "
            f"counts={magnitude_counts.tolist()}; strategy={magnitude_bin_strategy}",
            flush=True,
        )

    selections_path = args.source_dir / "patch_selections.npz"
    if not selections_path.exists():
        raise FileNotFoundError(f"Missing saved patch selections: {selections_path}")
    train_statistics = empirical_patch_statistics(
        train_wavelets, selections_path, "train"
    )
    test_statistics = empirical_patch_statistics(
        test_wavelets, selections_path, "test", max_images=len(test.images)
    )
    mean_checks = {
        band: float(np.max(np.abs(train_statistics[band].mean - models[band].mean)))
        for band in BAND_NAMES
    }
    if max(mean_checks.values()) > 1e-10:
        raise RuntimeError(
            "Recomputed empirical means do not match the fitted GSM models; "
            "the source selections or dataset differ."
        )

    covariance_rows: list[dict[str, object]] = []
    for band in BAND_NAMES:
        test_covariance = test_statistics[band].covariance
        denominator = max(float(np.linalg.norm(test_covariance)), 1e-15)
        for name, covariance in (
            ("empirical_train", train_statistics[band].covariance),
            ("gsm_fitted", models[band].covariance),
        ):
            covariance_rows.append(
                {
                    "band": band,
                    "covariance": name,
                    "relative_frobenius_error": float(
                        np.linalg.norm(covariance - test_covariance) / denominator
                    ),
                }
            )
    write_csv(args.out_dir / "covariance_heldout_diagnostic.csv", covariance_rows)
    plot_covariance_diagnostic(
        covariance_rows, args.out_dir / "covariance_heldout_diagnostic.png"
    )

    approximation_mean, approximation_covariance = fit_approximation_moments(
        train_wavelets.approximation
    )
    np.savez_compressed(
        args.out_dir / "input_moments.npz",
        approximation_mean=approximation_mean,
        approximation_covariance=approximation_covariance,
        **{
            f"{band}_{name}": value
            for band in BAND_NAMES
            for name, value in (
                ("empirical_mean", train_statistics[band].mean),
                ("empirical_covariance", train_statistics[band].covariance),
                ("test_covariance", test_statistics[band].covariance),
            )
        },
    )

    generator = np.random.default_rng(noise_seed)
    pixel_noise = generator.standard_normal(test.images.shape, dtype=np.float32)
    noise_wavelets = analyze_haar_level2(pixel_noise)
    visual_generator = np.random.default_rng(noise_seed + 1)
    visual_indices = np.sort(
        visual_generator.choice(
            len(test.images), size=min(args.num_visual_images, len(test.images)), replace=False
        )
    )
    np.save(args.out_dir / "visual_test_indices.npy", visual_indices)

    # Verify once that mixing in pixels and mixing transformed coefficients agree.
    audit_count = min(32, len(test.images))
    audit_time = 0.73
    direct_audit = analyze_haar_level2(
        audit_time * test.images[:audit_count]
        + (1.0 - audit_time) * pixel_noise[:audit_count]
    )
    linearity_errors = [
        np.max(
            np.abs(
                direct_audit.approximation
                - (
                    audit_time * test_wavelets.approximation[:audit_count]
                    + (1.0 - audit_time) * noise_wavelets.approximation[:audit_count]
                )
            )
        )
    ]
    for band in BAND_NAMES:
        linearity_errors.append(
            np.max(
                np.abs(
                    direct_audit.bands[band]
                    - (
                        audit_time * test_wavelets.bands[band][:audit_count]
                        + (1.0 - audit_time) * noise_wavelets.bands[band][:audit_count]
                    )
                )
            )
        )

    diagnostic_times = MAGNITUDE_DIAGNOSTIC_TIMES if magnitude_diagnostic_enabled else ()
    all_times = tuple(sorted(set((0.0, 1.0, *args.times, *diagnostic_times))))
    approximation_mean_tensor = torch.as_tensor(
        approximation_mean, device=device, dtype=torch.float64
    )
    approximation_covariance_tensor = torch.as_tensor(
        approximation_covariance, device=device, dtype=torch.float64
    )
    metrics_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    jitter_rows: list[dict[str, object]] = []
    magnitude_rows: list[dict[str, object]] = []
    magnitude_sse = {
        (time, method): np.zeros(len(magnitude_edges) - 1, dtype=np.float64)
        for time in diagnostic_times
        for method in MAGNITUDE_METHODS
    }
    per_image_metrics: dict[str, np.ndarray] = {}
    sanity: dict[str, object] = {
        "max_dwt_linearity_error": float(max(linearity_errors)),
        "max_model_vs_empirical_mean_error": mean_checks,
        "endpoint_t0_max_method_difference": 0.0,
        "endpoint_t1_max_clean_reconstruction_error": {},
        "max_parseval_absolute_sse_error": 0.0,
        "max_parseval_relative_sse_error_main_times": 0.0,
        "max_border_rule_error": {
            "wiener_matched_local_vs_self": 0.0,
            "wiener_empirical_local_vs_empirical_self": 0.0,
            "gsm_local_vs_self": 0.0,
        },
        "interior_and_border_counts_per_band": {},
        "a2_is_shared_exactly": True,
    }

    clean_approximation = test_wavelets.approximation.reshape(len(test.images), -1)
    noise_approximation = noise_wavelets.approximation.reshape(len(test.images), -1)
    approximation_shape = test_wavelets.approximation.shape[1:]
    approximation_dimension = int(np.prod(approximation_shape))
    for time in all_times:
        time_start = perf_counter()
        print(f"\n=== Full-image denoising at t={time:g} ===", flush=True)
        noisy_approximation = (
            time * clean_approximation + (1.0 - time) * noise_approximation
        )
        approximation_result = gaussian_vector_posterior_tensor(
            torch.as_tensor(noisy_approximation, device=device, dtype=torch.float64),
            time,
            approximation_mean_tensor,
            approximation_covariance_tensor,
        )
        denoised_approximation = approximation_result.mean.cpu().numpy().reshape(
            len(test.images), *approximation_shape
        )
        jitter_rows.append({"time": time, "a2_jitter": approximation_result.jitter})
        if approximation_result.jitter > 0.0:
            print(f"A2 Cholesky jitter used: {approximation_result.jitter:.3e}", flush=True)

        method_bands: dict[str, dict[str, np.ndarray]] = {
            method: {} for method in IMAGE_METHODS
        }
        region_sse = {
            method: {
                "interior": np.zeros(len(test.images), dtype=np.float64),
                "border": np.zeros(len(test.images), dtype=np.float64),
            }
            for method in IMAGE_METHODS
        }
        region_counts = {"interior": 0, "border": 0}

        for band in BAND_NAMES:
            clean_band = test_wavelets.bands[band]
            noisy_band = time * clean_band + (1.0 - time) * noise_wavelets.bands[band]
            predictions = denoise_detail_band_batched(
                noisy_band,
                time,
                train_statistics[band],
                models[band],
                device=device,
                dtype=compute_dtype,
                batch_size=args.batch_size,
            )
            if time in diagnostic_times:
                bin_indices = magnitude_assignments[band].reshape(-1)
                for method in MAGNITUDE_METHODS:
                    error = (
                        predictions[method].astype(np.float64, copy=False)
                        - clean_band
                    )
                    magnitude_sse[(time, method)] += np.bincount(
                        bin_indices,
                        weights=np.square(error).reshape(-1),
                        minlength=len(magnitude_edges) - 1,
                    )
            height, width = clean_band.shape[1:]
            interior_count = (height - 2) * (width - 2)
            current_border_count = height * width - interior_count
            sanity["interior_and_border_counts_per_band"][band] = {
                "interior": interior_count,
                "border": current_border_count,
            }
            region_counts["interior"] += interior_count
            region_counts["border"] += current_border_count
            mask = border_mask((height, width))

            empirical_self = gaussian_self_posterior_center(
                noisy_band.reshape(-1),
                time,
                train_statistics[band].mean[CENTER_INDEX],
                train_statistics[band].covariance[CENTER_INDEX, CENTER_INDEX],
            ).reshape(noisy_band.shape)
            border_errors = sanity["max_border_rule_error"]
            border_errors["wiener_matched_local_vs_self"] = max(
                border_errors["wiener_matched_local_vs_self"],
                float(
                    np.max(
                        np.abs(
                            predictions["wiener_matched_local"][:, mask]
                            - predictions["wiener_matched_self"][:, mask]
                        )
                    )
                ),
            )
            border_errors["wiener_empirical_local_vs_empirical_self"] = max(
                border_errors["wiener_empirical_local_vs_empirical_self"],
                float(
                    np.max(
                        np.abs(
                            predictions["wiener_empirical_local"][:, mask]
                            - empirical_self[:, mask]
                        )
                    )
                ),
            )
            border_errors["gsm_local_vs_self"] = max(
                border_errors["gsm_local_vs_self"],
                float(
                    np.max(
                        np.abs(
                            predictions["gsm_local"][:, mask]
                            - predictions["gsm_self"][:, mask]
                        )
                    )
                ),
            )

            for method, prediction in predictions.items():
                method_bands[method][band] = prediction
                error = prediction.astype(np.float64) - clean_band
                region_sse[method]["interior"] += np.sum(
                    error[:, 1:-1, 1:-1] ** 2, axis=(1, 2)
                )
                region_sse[method]["border"] += np.sum(
                    error[:, mask] ** 2, axis=1
                )

        if time in diagnostic_times:
            labels = magnitude_bin_labels(magnitude_edges)
            for method in MAGNITUDE_METHODS:
                for bin_index, (label, count) in enumerate(
                    zip(labels, magnitude_counts, strict=True)
                ):
                    mse = (
                        magnitude_sse[(time, method)][bin_index] / count
                        if count > 0
                        else float("nan")
                    )
                    magnitude_rows.append(
                        {
                            "time": time,
                            "bin_index": bin_index,
                            "bin_lower": float(magnitude_edges[bin_index]),
                            "bin_upper": float(magnitude_edges[bin_index + 1]),
                            "bin_label": label,
                            "count": int(count),
                            "method": method,
                            "mse": float(mse),
                        }
                    )

        reconstructions: dict[str, np.ndarray] = {}
        image_errors: dict[str, np.ndarray] = {}
        for method in IMAGE_METHODS:
            coefficients = HaarLevel2Batch(
                approximation=denoised_approximation,
                bands=method_bands[method],
            )
            reconstruction = synthesize_haar_level2(coefficients).astype(
                np.float64, copy=False
            )
            reconstructions[method] = reconstruction
            squared_error = (reconstruction - test.images) ** 2
            per_image = squared_error.mean(axis=(1, 2))
            image_errors[method] = per_image
            per_image_metrics[f"t_{time:g}_{method}"] = per_image
            mean, lower, upper = confidence_interval(per_image)
            metrics_rows.append(
                {
                    "time": time,
                    "method": method,
                    "mse": mean,
                    "mean_ci95_low": lower,
                    "mean_ci95_high": upper,
                    "num_images": len(test.images),
                }
            )

            coefficient_sse = np.sum(
                (denoised_approximation - test_wavelets.approximation) ** 2,
                axis=(1, 2),
            )
            for band in BAND_NAMES:
                coefficient_sse += np.sum(
                    (method_bands[method][band] - test_wavelets.bands[band]) ** 2,
                    axis=(1, 2),
                )
            pixel_sse = squared_error.sum(axis=(1, 2))
            absolute_parseval = np.abs(pixel_sse - coefficient_sse)
            sanity["max_parseval_absolute_sse_error"] = max(
                sanity["max_parseval_absolute_sse_error"],
                float(absolute_parseval.max()),
            )
            if time in args.times:
                relative_parseval = absolute_parseval / np.maximum(pixel_sse, 1e-15)
                sanity["max_parseval_relative_sse_error_main_times"] = max(
                    sanity["max_parseval_relative_sse_error_main_times"],
                    float(relative_parseval.max()),
                )

            for region in ("interior", "border"):
                values = region_sse[method][region] / region_counts[region]
                region_rows.append(
                    {
                        "time": time,
                        "scope": "all_detail_bands",
                        "region": region,
                        "method": method,
                        "mse": float(values.mean()),
                        "num_coefficients_per_image": region_counts[region],
                    }
                )

        for baseline, candidate, key, _ in CONTRASTS:
            difference = image_errors[baseline] - image_errors[candidate]
            mean, lower, upper = confidence_interval(difference)
            baseline_mean = float(image_errors[baseline].mean())
            if baseline_mean <= 1e-15:
                relative = relative_lower = relative_upper = float("nan")
            else:
                relative = mean / baseline_mean
                relative_lower = lower / baseline_mean
                relative_upper = upper / baseline_mean
            comparison_rows.append(
                {
                    "time": time,
                    "contrast": key,
                    "baseline": baseline,
                    "candidate": candidate,
                    "mean_mse_reduction": mean,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "relative_reduction": relative,
                    "relative_ci95_low": relative_lower,
                    "relative_ci95_high": relative_upper,
                    "num_images": len(test.images),
                }
            )

        if time == 0.0:
            reference = reconstructions[IMAGE_METHODS[0]]
            sanity["endpoint_t0_max_method_difference"] = max(
                float(np.max(np.abs(reconstruction - reference)))
                for reconstruction in reconstructions.values()
            )
        if time == 1.0:
            sanity["endpoint_t1_max_clean_reconstruction_error"] = {
                method: float(np.max(np.abs(reconstruction - test.images)))
                for method, reconstruction in reconstructions.items()
            }
        if time in args.times:
            noisy_images = time * test.images + (1.0 - time) * pixel_noise
            plot_reconstructions(
                test.images,
                noisy_images,
                reconstructions,
                visual_indices,
                time,
                args.out_dir / f"reconstructions_t{int(round(100 * time)):03d}.png",
            )
        print(f"completed in {perf_counter() - time_start:.1f}s", flush=True)

        write_csv(args.out_dir / "image_mse_metrics.csv", metrics_rows)
        write_csv(args.out_dir / "interior_border_metrics.csv", region_rows)
        write_csv(args.out_dir / "paired_comparisons.csv", comparison_rows)
        write_csv(args.out_dir / "a2_jitter_log.csv", jitter_rows)
        if magnitude_diagnostic_enabled:
            write_csv(
                args.out_dir / "conditional_mse_by_clean_magnitude.csv",
                magnitude_rows,
            )

    np.savez_compressed(args.out_dir / "per_image_mse.npz", **per_image_metrics)
    plot_image_mse(metrics_rows, args.times, args.out_dir / "image_mse_vs_t.png")
    plot_region_mse(
        region_rows, args.times, args.out_dir / "interior_border_mse_vs_t.png"
    )
    plot_relative_gains(
        comparison_rows, args.times, args.out_dir / "relative_image_mse_gains.png"
    )
    magnitude_gain_rows: list[dict[str, object]] = []
    if magnitude_diagnostic_enabled:
        magnitude_lookup = {
            (float(row["time"]), int(row["bin_index"]), str(row["method"])): float(
                row["mse"]
            )
            for row in magnitude_rows
        }
        labels = magnitude_bin_labels(magnitude_edges)
        for time in diagnostic_times:
            for bin_index, (label, count) in enumerate(
                zip(labels, magnitude_counts, strict=True)
            ):
                wiener_risk = magnitude_lookup[
                    (time, bin_index, "wiener_matched_local")
                ]
                gsm_risk = magnitude_lookup[(time, bin_index, "gsm_local")]
                gain = (
                    (wiener_risk - gsm_risk) / wiener_risk
                    if wiener_risk > 0.0
                    else float("nan")
                )
                magnitude_gain_rows.append(
                    {
                        "time": time,
                        "bin_index": bin_index,
                        "bin_lower": float(magnitude_edges[bin_index]),
                        "bin_upper": float(magnitude_edges[bin_index + 1]),
                        "bin_label": label,
                        "count": int(count),
                        "wiener_matched_local_mse": wiener_risk,
                        "gsm_local_mse": gsm_risk,
                        "relative_gain": float(gain),
                    }
                )
        write_csv(
            args.out_dir / "gsm_local_gain_by_clean_magnitude.csv",
            magnitude_gain_rows,
        )
        plot_conditional_magnitude_mse(
            magnitude_rows,
            diagnostic_times,
            args.out_dir / "conditional_mse_by_clean_magnitude.png",
        )
        plot_conditional_magnitude_gain(
            magnitude_gain_rows,
            diagnostic_times,
            args.out_dir / "gsm_local_gain_by_clean_magnitude.png",
        )
        save_json(
            args.out_dir / "magnitude_diagnostic_config.json",
            {
                "definition": "a_q = abs(Y_0,q) / sigma_b",
                "sigma_b": "population standard deviation over all training coefficients in subband b",
                "train_only_standard_deviations": magnitude_standard_deviations,
                "bin_edges": [
                    None if np.isinf(edge) else float(edge) for edge in magnitude_edges
                ],
                "bin_edge_note": "null is positive infinity",
                "bin_labels": magnitude_bin_labels(magnitude_edges),
                "bin_counts": magnitude_counts.tolist(),
                "bin_counts_by_subband": {
                    band: np.bincount(
                        magnitude_assignments[band].reshape(-1),
                        minlength=len(magnitude_edges) - 1,
                    ).tolist()
                    for band in BAND_NAMES
                },
                "bin_strategy": magnitude_bin_strategy,
                "times": list(diagnostic_times),
                "methods": list(MAGNITUDE_METHODS),
                "aggregation": "all test images, all positions, and all six detail subbands",
                "border_rule": "same self-only fallback as the complete-image protocol",
            },
        )

    if sanity["max_dwt_linearity_error"] > 2e-6:
        raise RuntimeError(f"DWT linearity audit failed: {sanity['max_dwt_linearity_error']}")
    if sanity["endpoint_t0_max_method_difference"] > 2e-5:
        raise RuntimeError("The five methods do not share their expected t=0 prior mean.")
    if max(sanity["endpoint_t1_max_clean_reconstruction_error"].values()) > 2e-5:
        raise RuntimeError("At least one method fails to reconstruct the clean image at t=1.")
    if sanity["max_parseval_relative_sse_error_main_times"] > 2e-5:
        raise RuntimeError("The image/coefficient error Parseval audit failed.")
    if max(sanity["max_border_rule_error"].values()) > 2e-6:
        raise RuntimeError("At least one local method violates its self-only border rule.")

    elapsed = perf_counter() - start_time
    config = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "data_root": str(data_root),
        "times_plotted": list(args.times),
        "times_audited": list(all_times),
        "path": "X_t = t X_0 + (1-t) Z with one Z per test image shared over t",
        "methods": list(IMAGE_METHODS),
        "magnitude_diagnostic_enabled": magnitude_diagnostic_enabled,
        "magnitude_diagnostic_times": list(diagnostic_times),
        "image_size": image_size,
        "a2": (
            f"shared {approximation_dimension}D empirical affine Wiener, "
            "float64, no default ridge"
        ),
        "detail_border": "self posterior only; no padding or wrapping",
        "detail_interior": "one 3x3 posterior prediction per center; no overlap averaging",
        "normalization": "MNIST float pixels in [0,1]",
        "resize": source_config.get("resize", "none"),
        "metric": "raw unclipped image MSE",
        "display": "clipped to [0,1]",
        "noise_seed": noise_seed,
        "visual_indices": visual_indices.tolist(),
        "num_train_images": len(train.images),
        "num_test_images": len(test.images),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "batch_size": args.batch_size,
        "source_config": str(args.source_dir / "config.json"),
    }
    save_json(args.out_dir / "config.json", config)
    save_json(args.out_dir / "sanity_checks.json", sanity)
    save_json(
        args.out_dir / "run_summary.json",
        {
            "elapsed_seconds": elapsed,
            "peak_cuda_tensor_memory_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        },
    )
    print(f"\nResults written to {args.out_dir}", flush=True)
    print(f"total elapsed: {elapsed:.1f}s", flush=True)
    if device.type == "cuda":
        print(
            f"peak CUDA tensor memory: "
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
