"""Measure subband predictability and information gains under masked FM.

Inputs: MNIST-32 images and a standard flow-matching noise path.
Outputs: per-subband MSE and information-gain plots, SNR plot, CSV, and config.
Example: python -m wavelet_fm_structure.run_subband_information --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_fm_structure.closed_form import ScalarMaskedEmpiricalVelocity
from wavelet_fm_structure.masks import (
    MaskName,
    build_subband_target_groups,
    build_target_masks,
)
from wavelet_fm_structure.plot_subband_snr import (
    compute_subband_signal_variances,
    plot_subband_snr,
)
from wavelet_fm_structure.protocol import DEFAULT_PROTOCOL
from wavelet_score.data import load_mnist_32
from wavelet_score.transforms import WaveletTransform2D

MASK_ORDER: tuple[MaskName, ...] = ("self", "orientation", "local", "scale", "all")
GAIN_MASKS: tuple[MaskName, ...] = ("orientation", "local", "scale", "all")


def parse_times(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("times must lie strictly between zero and one")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(
            "wavelet_fm_structure/results/experiment_a_mnist32_subband_aggregated"
        ),
    )
    parser.add_argument("--times", type=parse_times, default=DEFAULT_PROTOCOL.times)
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument(
        "--num-evaluation-queries", type=int,
        default=1024,
    )
    parser.add_argument("--exhaustive-max-size", type=int, default=64)
    parser.add_argument("--sampled-grid-size", type=int, default=4)
    parser.add_argument("--reference-chunk-size", type=int, default=2048)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.max_reference_images is not None and args.max_reference_images < 1:
        parser.error("max-reference-images must be positive")
    if min(
        args.num_evaluation_queries,
        args.reference_chunk_size,
        args.query_batch_size,
        args.exhaustive_max_size,
        args.sampled_grid_size,
    ) < 1:
        parser.error("query and chunk sizes must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def plot_heatmap(
    values: np.ndarray,
    row_names: list[str],
    times: tuple[float, ...],
    title: str,
    colorbar_label: str,
    out_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", interpolation="nearest")
    axis.set(
        xticks=np.arange(len(times)),
        xticklabels=[f"{time:.2f}" for time in times],
        yticks=np.arange(len(row_names)),
        yticklabels=row_names,
        xlabel="Flow time t (noise to data)",
        ylabel="Target wavelet subband (coarse to fine)",
        title=title,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def plot_gain_heatmaps(
    gain_values: dict[str, np.ndarray],
    row_names: list[str],
    times: tuple[float, ...],
    out_path: Path,
) -> None:
    finite = np.concatenate([
        values[np.isfinite(values)] for values in gain_values.values()
    ])
    extent = max(0.05, float(np.max(np.abs(finite))))
    norm = colors.TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    last_image = None
    for axis, mask_name in zip(axes.flat, GAIN_MASKS, strict=True):
        values = gain_values[mask_name]
        last_image = axis.imshow(
            values, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest"
        )
        axis.set(
            xticks=np.arange(len(times)),
            xticklabels=[f"{time:.2f}" for time in times],
            yticks=np.arange(len(row_names)),
            yticklabels=row_names,
            xlabel="Flow time t (noise to data)",
            ylabel="Target subband",
            title=f"self + {mask_name}" if mask_name != "all" else "all structured groups",
        )
        for row, column in zip(*np.where(~np.isfinite(values)), strict=True):
            axis.text(column, row, "N/A", ha="center", va="center", fontsize=7)
    assert last_image is not None
    colorbar = figure.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.9)
    colorbar.set_label(r"Relative information gain $(R_{self}-R_S)/R_{self}$")
    figure.suptitle(
        "Subband-averaged information from wavelet receptive groups\n"
        "All 4x4/8x8 positions; spatial 4x4 grid for 16x16 bands"
    )
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


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
    target_groups = build_subband_target_groups(
        layout,
        exhaustive_max_size=args.exhaustive_max_size,
        sampled_grid_size=args.sampled_grid_size,
    )
    masks_by_target = {
        target.flat_index: build_target_masks(layout, target)
        for targets in target_groups.values()
        for target in targets
    }
    print(
        "target coordinates by subband: "
        + ", ".join(f"{name}={len(targets)}" for name, targets in target_groups.items()),
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    clean_indices = rng.integers(0, len(reference_Y0), size=args.num_evaluation_queries)
    query_Y0 = reference_Y0[clean_indices]
    query_noise = rng.standard_normal(query_Y0.shape).astype(np.float64)
    conditional_velocity = query_Y0 - query_noise

    estimator = ScalarMaskedEmpiricalVelocity(
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        device=device,
    ).fit(reference_Y0)
    rows: list[dict[str, object]] = []
    start_time = perf_counter()
    for time in args.times:
        print(f"\n=== t={time:.2f} ===", flush=True)
        query_Yt = (1.0 - time) * query_noise + time * query_Y0
        for subband, targets in target_groups.items():
            print(f"{subband}: evaluating {len(targets)} targets", flush=True)
            target_energy_sum = 0.0
            squared_error_sums = {name: 0.0 for name in MASK_ORDER}
            coordinate_risks: dict[MaskName, list[float]] = {
                name: [] for name in MASK_ORDER
            }
            visible_sizes: dict[MaskName, list[int]] = {
                name: [] for name in MASK_ORDER
            }
            for target_number, target in enumerate(targets, start=1):
                target_velocity = conditional_velocity[:, target.flat_index]
                target_energy_sum += float(np.sum(target_velocity**2))
                prediction_cache: dict[tuple[int, ...], np.ndarray] = {}
                for mask_name in MASK_ORDER:
                    mask = masks_by_target[target.flat_index][mask_name]
                    prediction = prediction_cache.get(mask.flat_indices)
                    if prediction is None:
                        prediction = estimator.predict(
                            query_Yt, time, target.flat_index, mask.flat_indices
                        )
                        prediction_cache[mask.flat_indices] = prediction
                    squared_error = (target_velocity - prediction) ** 2
                    squared_error_sums[mask_name] += float(np.sum(squared_error))
                    coordinate_risks[mask_name].append(float(np.mean(squared_error)))
                    visible_sizes[mask_name].append(len(mask.flat_indices))
                if target_number % 16 == 0 or target_number == len(targets):
                    print(f"  targets {target_number:02d}/{len(targets):02d}", flush=True)

            num_risk_values = len(targets) * args.num_evaluation_queries
            target_energy = target_energy_sum / num_risk_values
            if not np.isfinite(target_energy) or target_energy <= 0.0:
                raise FloatingPointError(f"Invalid target energy for {subband}.")
            target_rows: list[dict[str, object]] = []
            for mask_name in MASK_ORDER:
                risk = squared_error_sums[mask_name] / num_risk_values
                sizes = visible_sizes[mask_name]
                row: dict[str, object] = {
                    "experiment": "A_subband_aggregated_information",
                    "time": time,
                    "subband": subband,
                    "target_flat_index": "",
                    "target_k1": "",
                    "target_k2": "",
                    "num_target_coordinates": len(targets),
                    "mask": mask_name,
                    "interaction_class": "",
                    "num_visible_coordinates": float(np.mean(sizes)),
                    "min_visible_coordinates": min(sizes),
                    "max_visible_coordinates": max(sizes),
                    "mse": risk,
                    "normalized_mse": risk / target_energy,
                    "coordinate_mse_std": float(np.std(coordinate_risks[mask_name])),
                    "relative_gain": 0.0,
                    "gap_to_closed_form_all": "",
                    "kernel_width_multiplier": "",
                    "ridge": "",
                    "seed": args.seed,
                    "num_reference_images": len(reference_Y0),
                    "num_fit_queries": 0,
                    "num_validation_queries": 0,
                    "num_evaluation_queries": args.num_evaluation_queries,
                }
                target_rows.append(row)
            self_risk = float(target_rows[0]["mse"])
            for row in target_rows:
                if subband == "cA3" and row["mask"] == "orientation":
                    row["relative_gain"] = float("nan")
                else:
                    row["relative_gain"] = (self_risk - float(row["mse"])) / self_risk
            rows.extend(target_rows)
            print(
                "  " + ", ".join(
                    f"{row['mask']}={float(row['normalized_mse']):.4f}"
                    for row in target_rows
                ),
                flush=True,
            )

    fieldnames = list(rows[0])
    with (args.out_dir / "experiment_a_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    subband_names = list(target_groups)
    row_lookup = {
        (str(row["subband"]), float(row["time"]), str(row["mask"])): row for row in rows
    }
    predictability = np.array([
        [
            float(row_lookup[(subband, time, "all")]["normalized_mse"])
            for time in args.times
        ]
        for subband in subband_names
    ])
    gains = {
        mask_name: np.array([
            [
                float(row_lookup[(subband, time, mask_name)]["relative_gain"])
                for time in args.times
            ]
            for subband in subband_names
        ])
        for mask_name in GAIN_MASKS
    }
    plot_heatmap(
        predictability,
        subband_names,
        args.times,
        "Subband-averaged predictability of wavelet velocities\n"
        "All 4x4/8x8 positions; spatial 4x4 grid for 16x16 bands",
        r"Normalized risk $R_b(all,t) / \mathbb{E}_{n,i\in b}[V_{n,i}^2]$",
        args.out_dir / "predictability_by_subband_and_time.png",
    )
    plot_gain_heatmaps(
        gains, subband_names, args.times, args.out_dir / "mask_information_gains.png"
    )
    signal_variances = compute_subband_signal_variances(reference_Y0, layout)
    plot_subband_snr(
        signal_variances, args.out_dir / "effective_snr_by_subband.png"
    )

    config = {
        "dataset": "MNIST training split only",
        "normalization": {"global_mean": global_mean, "global_std": global_std},
        "wavelet": DEFAULT_PROTOCOL.wavelet_name,
        "level": DEFAULT_PROTOCOL.decomposition_level,
        "mode": DEFAULT_PROTOCOL.mode,
        "dtype": "float64",
        "path": "Y_t=(1-t)xi+tY_0",
        "times": args.times,
        "device": str(device),
        "seed": args.seed,
        "num_reference_images": len(reference_Y0),
        "num_evaluation_queries": args.num_evaluation_queries,
        "clean_indices_sampled_with_replacement": True,
        "reference_chunk_size": args.reference_chunk_size,
        "query_batch_size": args.query_batch_size,
        "target_selection": {
            "rule": "exhaustive small bands, cell-centered regular grid otherwise",
            "exhaustive_max_size": args.exhaustive_max_size,
            "sampled_grid_size": args.sampled_grid_size,
            "num_target_coordinates": {
                name: len(targets) for name, targets in target_groups.items()
            },
        },
        "subband_signal_variances": signal_variances,
        "elapsed_seconds": perf_counter() - start_time,
        "targets": {
            name: [
                {
                    "flat_index": target.flat_index,
                    "position": [target.k1, target.k2],
                }
                for target in targets
            ]
            for name, targets in target_groups.items()
        },
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"saved experiment A to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
