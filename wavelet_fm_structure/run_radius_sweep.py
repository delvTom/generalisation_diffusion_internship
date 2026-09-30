"""Study the spatial radius required by exact masked empirical FM optima."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_fm_structure.closed_form import ScalarMaskedEmpiricalVelocity
from wavelet_fm_structure.masks import (
    RadiusMaskFamily,
    TargetCoefficient,
    build_central_targets,
    build_radius_masks,
    max_nonaliasing_radius,
)
from wavelet_fm_structure.protocol import DEFAULT_PROTOCOL
from wavelet_fm_structure.radius_study import (
    approximate_physical_radius_pixels,
    evaluate_clean_risk,
    sample_flow_matching_queries,
    select_smallest_near_optimal_radius,
)
from wavelet_score.data import load_mnist_32
from wavelet_score.transforms import WaveletTransform2D

FAMILIES: tuple[RadiusMaskFamily, ...] = (
    "same_band",
    "orientation_augmented",
    "scale_augmented",
    "full",
)
FAMILY_TITLES = {
    "same_band": "Same subband",
    "orientation_augmented": "Orientation augmented",
    "scale_augmented": "Scale augmented",
    "full": "Full structured mask",
}
STRATEGIES = ("fixed r=1", "maximal radius", "adaptive radius")


def parse_times(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("times must lie strictly inside (0,1)")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_fm_structure/results/radius_sweep_mnist32"),
    )
    parser.add_argument("--times", type=parse_times, default=DEFAULT_PROTOCOL.times)
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument(
        "--num-evaluation-queries",
        type=int,
        default=DEFAULT_PROTOCOL.num_evaluation_queries,
    )
    parser.add_argument("--reference-chunk-size", type=int, default=2048)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.max_reference_images is not None and args.max_reference_images < 2:
        parser.error("max-reference-images must be at least two")
    if min(
        args.num_evaluation_queries,
        args.reference_chunk_size,
        args.query_batch_size,
    ) < 1:
        parser.error("query and chunk sizes must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    """Resolve the requested compute device, preferring CUDA in auto mode."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _rows_for(
    rows: list[dict[str, object]],
    *,
    subband: str | None = None,
    time: float | None = None,
    family: str | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (subband is None or row["subband"] == subband)
        and (time is None or row["t"] == time)
        and (family is None or row["family"] == family)
    ]


def _risk_scale(rows: list[dict[str, object]]) -> tuple[float, float]:
    values = np.asarray(
        [float(row["normalized_clean_risk"]) for row in rows], dtype=np.float64
    )
    positive = values[values > 0.0]
    linear_threshold = max(
        1e-14,
        float(np.min(positive)) * 0.1 if len(positive) else 1e-14,
    )
    upper = max(1.0, float(np.max(values)) * 1.08)
    return linear_threshold, upper


def plot_risk_by_radius(
    rows: list[dict[str, object]],
    subbands: list[str],
    times: tuple[float, ...],
    out_dir: Path,
) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(times)))
    linear_threshold, upper = _risk_scale(rows)
    for family in FAMILIES:
        figure, axes = plt.subplots(2, 5, figsize=(19, 8), sharex=True, sharey=True)
        for axis, subband in zip(axes.flat, subbands, strict=True):
            for color, time in zip(colors, times, strict=True):
                selected = sorted(
                    _rows_for(rows, subband=subband, time=time, family=family),
                    key=lambda row: int(row["radius"]),
                )
                axis.plot(
                    [int(row["radius"]) for row in selected],
                    [float(row["normalized_clean_risk"]) for row in selected],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.5,
                    color=color,
                    label=f"t={time:.2f}",
                )
            axis.set_title(subband)
            axis.set_xticks(range(8))
            axis.set_xlim(-0.2, 7.2)
            axis.set_ylim(0.0, upper)
            axis.set_yscale("symlog", linthresh=linear_threshold)
            axis.grid(True, which="both", alpha=0.22)
        figure.supxlabel("Chebyshev radius r in the target subband grid")
        figure.supylabel(
            r"Normalized clean risk $\mathbb{E}[(\widehat Y_{0,i}-Y_{0,i})^2]"
            r"/\mathrm{Var}(Y_{0,i})$"
        )
        figure.suptitle(
            f"Clean-coefficient risk versus spatial radius: {FAMILY_TITLES[family]}"
        )
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncols=len(times), bbox_to_anchor=(0.5, 0.94))
        figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.89))
        figure.savefig(out_dir / f"clean_risk_vs_radius_{family}.png", dpi=180)
        plt.close(figure)


def plot_selected_radius_heatmap(
    rows: list[dict[str, object]],
    subbands: list[str],
    times: tuple[float, ...],
    out_path: Path,
) -> None:
    values = np.empty((len(subbands), len(times)), dtype=np.int64)
    for row_index, subband in enumerate(subbands):
        for column_index, time in enumerate(times):
            selected = _rows_for(rows, subband=subband, time=time, family="full")
            values[row_index, column_index] = int(selected[0]["selected_radius"])

    boundaries = np.arange(-0.5, 8.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=256)
    figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", norm=norm)
    axis.set(
        xticks=np.arange(len(times)),
        xticklabels=[f"{time:.2f}" for time in times],
        yticks=np.arange(len(subbands)),
        yticklabels=subbands,
        xlabel="Flow time t (noise to data)",
        ylabel="Central target subband (coarse to fine)",
        title="Smallest full-mask radius within 1% of maximal-radius clean risk",
    )
    for row_index in range(len(subbands)):
        for column_index in range(len(times)):
            value = int(values[row_index, column_index])
            text_color = "white" if value <= 2 or value >= 6 else "black"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, ticks=range(8))
    colorbar.set_label(r"Selected radius $r^*(t)$")
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _strategy_rows(
    rows: list[dict[str, object]],
    subband: str,
    times: tuple[float, ...],
) -> dict[str, list[dict[str, object]]]:
    result = {strategy: [] for strategy in STRATEGIES}
    for time in times:
        candidates = _rows_for(rows, subband=subband, time=time, family="full")
        by_radius = {int(row["radius"]): row for row in candidates}
        maximum = max(by_radius)
        adaptive = int(candidates[0]["selected_radius"])
        result["fixed r=1"].append(by_radius[min(1, maximum)])
        result["maximal radius"].append(by_radius[maximum])
        result["adaptive radius"].append(by_radius[adaptive])
    return result


def plot_full_strategy_comparison(
    rows: list[dict[str, object]],
    subbands: list[str],
    times: tuple[float, ...],
    out_dir: Path,
) -> None:
    styles = {
        "fixed r=1": ("#377eb8", "o"),
        "maximal radius": ("#e41a1c", "s"),
        "adaptive radius": ("#4daf4a", "D"),
    }
    linear_threshold, risk_upper = _risk_scale(
        _rows_for(rows, family="full")
    )
    specifications = (
        (
            "normalized_clean_risk",
            "Normalized clean risk",
            "Full-mask strategies: clean-coefficient risk",
            "full_strategies_clean_risk.png",
            "symlog",
        ),
        (
            "mask_size",
            "Number of visible wavelet coefficients",
            "Full-mask strategies: mask cardinality",
            "full_strategies_mask_size.png",
            "log",
        ),
    )
    for field, ylabel, title, filename, scale in specifications:
        figure, axes = plt.subplots(2, 5, figsize=(19, 8), sharex=True, sharey=True)
        for axis, subband in zip(axes.flat, subbands, strict=True):
            strategy_rows = _strategy_rows(rows, subband, times)
            for strategy in STRATEGIES:
                color, marker = styles[strategy]
                axis.plot(
                    times,
                    [float(row[field]) for row in strategy_rows[strategy]],
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4,
                    label=strategy,
                )
            axis.set_title(subband)
            axis.set_xticks(times)
            axis.tick_params(axis="x", rotation=45)
            if scale == "symlog":
                axis.set_yscale("symlog", linthresh=linear_threshold)
                axis.set_ylim(0.0, risk_upper)
            else:
                axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.22)
        figure.supxlabel("Flow time t (noise to data)")
        figure.supylabel(ylabel)
        figure.suptitle(title)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.94))
        figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.89))
        figure.savefig(out_dir / filename, dpi=180)
        plt.close(figure)


def write_metrics(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metrics table.")
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device}, dtype=float64", flush=True)

    split = load_mnist_32(
        args.data_root,
        "train",
        max_samples=args.max_reference_images,
        batch_size=1024,
        download=args.download,
    )
    global_mean = float(np.mean(split.images, dtype=np.float64))
    global_std = float(np.std(split.images, dtype=np.float64))
    if not np.isfinite(global_std) or global_std <= 0.0:
        raise RuntimeError("MNIST global standard deviation is invalid.")
    normalized_images = np.ascontiguousarray(
        (split.images.astype(np.float64) - global_mean) / global_std
    )
    transform = WaveletTransform2D(
        DEFAULT_PROTOCOL.wavelet_name,
        DEFAULT_PROTOCOL.decomposition_level,
        DEFAULT_PROTOCOL.mode,
    )
    reference_Y0, layout = transform.analyze(normalized_images)
    del normalized_images
    targets = build_central_targets(layout)
    target_masks = {
        target.flat_index: {
            radius: build_radius_masks(layout, target, radius)
            for radius in range(max_nonaliasing_radius(target) + 1)
        }
        for target in targets
    }
    clean_variances = {
        target.flat_index: float(
            np.var(reference_Y0[:, target.flat_index], dtype=np.float64)
        )
        for target in targets
    }
    if any(variance <= 0.0 for variance in clean_variances.values()):
        raise RuntimeError("A target clean coefficient has zero empirical variance.")

    queries = sample_flow_matching_queries(
        reference_Y0, args.num_evaluation_queries, args.seed
    )
    estimator = ScalarMaskedEmpiricalVelocity(
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        device=device,
    ).fit(reference_Y0)

    rows: list[dict[str, object]] = []
    largest_identity_error = 0.0
    start_time = perf_counter()
    for time in args.times:
        print(f"\n=== t={time:.2f} ===", flush=True)
        query_Yt = (1.0 - time) * queries.noise + time * queries.clean_Y0
        for target in targets:
            print(
                f"{target.subband}: radii 0..{max_nonaliasing_radius(target)}",
                flush=True,
            )
            target_velocity = queries.velocity[:, target.flat_index]
            noisy_coefficient = query_Yt[:, target.flat_index]
            clean_coefficient = queries.clean_Y0[:, target.flat_index]
            target_rows: list[dict[str, object]] = []
            prediction_cache: dict[tuple[int, ...], np.ndarray] = {}
            for radius, masks in target_masks[target.flat_index].items():
                for family in FAMILIES:
                    mask = masks[family]
                    prediction = prediction_cache.get(mask.flat_indices)
                    if prediction is None:
                        prediction = estimator.predict(
                            query_Yt,
                            time,
                            target.flat_index,
                            mask.flat_indices,
                        )
                        prediction_cache[mask.flat_indices] = prediction
                    risk = evaluate_clean_risk(
                        prediction,
                        target_velocity,
                        noisy_coefficient,
                        clean_coefficient,
                        time,
                        clean_variances[target.flat_index],
                    )
                    largest_identity_error = max(
                        largest_identity_error, risk.identity_max_abs_error
                    )
                    if risk.identity_max_abs_error > 1e-10:
                        raise FloatingPointError(
                            "Clean/velocity error identity failed: "
                            f"{risk.identity_max_abs_error:.3e}."
                        )
                    target_rows.append(
                        {
                            "subband": target.subband,
                            "target_flat_index": target.flat_index,
                            "target_k1": target.k1,
                            "target_k2": target.k2,
                            "pywt_level": target.metadata.pywt_level,
                            "t": time,
                            "family": family,
                            "radius": radius,
                            "physical_radius_pixels_approx": (
                                approximate_physical_radius_pixels(
                                    radius, target.metadata.pywt_level
                                )
                            ),
                            "mask_size": len(mask.flat_indices),
                            "clean_mse": risk.clean_mse,
                            "clean_variance": risk.clean_variance,
                            "normalized_clean_risk": risk.normalized_clean_risk,
                            "selected_radius": "",
                            "identity_max_abs_error": risk.identity_max_abs_error,
                            "seed": args.seed,
                            "num_reference_images": len(reference_Y0),
                            "num_evaluation_queries": args.num_evaluation_queries,
                        }
                    )

            full_risks = {
                int(row["radius"]): float(row["normalized_clean_risk"])
                for row in target_rows
                if row["family"] == "full"
            }
            selected_radius = select_smallest_near_optimal_radius(full_risks)
            for row in target_rows:
                row["selected_radius"] = selected_radius
            rows.extend(target_rows)
            print(f"  selected full radius={selected_radius}", flush=True)

    write_metrics(rows, args.out_dir / "radius_sweep_metrics.csv")
    subbands = [target.subband for target in targets]
    plot_risk_by_radius(rows, subbands, args.times, args.out_dir)
    plot_selected_radius_heatmap(
        rows, subbands, args.times, args.out_dir / "selected_radius_heatmap.png"
    )
    plot_full_strategy_comparison(rows, subbands, args.times, args.out_dir)

    elapsed_seconds = perf_counter() - start_time
    config = {
        "dataset": "MNIST training split only",
        "normalization": {"global_mean": global_mean, "global_std": global_std},
        "wavelet": DEFAULT_PROTOCOL.wavelet_name,
        "level": DEFAULT_PROTOCOL.decomposition_level,
        "mode": DEFAULT_PROTOCOL.mode,
        "dtype": "float64",
        "device": str(device),
        "path": "Y_t=(1-t)xi+tY_0",
        "target_velocity": "V=Y_0-xi",
        "metric": "E[(Y0_hat_i-Y0_i)^2]/Var_reference(Y0_i), ddof=0",
        "physical_radius_approximation": "radius * 2**pywt_level pixels",
        "radius_selection": (
            "smallest r with R_full(r,t) <= 1.01 * R_full(r_max,t) "
            "+ float64 roundoff tolerance"
        ),
        "times": args.times,
        "seed": args.seed,
        "num_reference_images": len(reference_Y0),
        "num_evaluation_queries": args.num_evaluation_queries,
        "clean_indices_sampled_with_replacement": True,
        "reference_chunk_size": args.reference_chunk_size,
        "query_batch_size": args.query_batch_size,
        "largest_identity_max_abs_error": largest_identity_error,
        "elapsed_seconds": elapsed_seconds,
        "targets": {
            target.subband: {
                "flat_index": target.flat_index,
                "position": [target.k1, target.k2],
                "pywt_level": target.metadata.pywt_level,
                "subband_shape": list(target.metadata.subband_shape),
                "max_radius": max_nonaliasing_radius(target),
                "clean_variance": clean_variances[target.flat_index],
            }
            for target in targets
        },
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"elapsed_seconds={elapsed_seconds:.2f}", flush=True)
    print(f"saved radius study to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
