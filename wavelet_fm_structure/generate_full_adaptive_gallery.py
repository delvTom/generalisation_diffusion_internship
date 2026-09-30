"""Generate a larger full-r*(t) MNIST gallery with training-set neighbors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_fm_structure.adaptive_radius import (
    FullRadiusSchedule,
    build_radius_group_families,
)
from wavelet_fm_structure.run_radius_mask_generation import (
    display_images,
    load_calibration_config,
    parse_float_list,
    resolve_device,
    snapshot_indices,
)
from wavelet_score.data import load_mnist_32
from wavelet_score.masked_empirical_fm import MaskedEmpiricalWaveletVelocity
from wavelet_score.transforms import WaveletTransform2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--radius-metrics",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/radius_sweep_mnist32/"
            "radius_sweep_metrics.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/radius_mask_generation_mnist32"
        ),
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument(
        "--trajectory-times",
        type=parse_float_list,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument("--reference-chunk-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=10)
    parser.add_argument("--group-chunk-size", type=int, default=32)
    parser.add_argument("--neighbor-chunk-size", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=1730)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    positive = (
        args.num_samples,
        args.num_trajectories,
        args.num_steps,
        args.reference_chunk_size,
        args.query_batch_size,
        args.group_chunk_size,
        args.neighbor_chunk_size,
    )
    if any(value < 1 for value in positive):
        parser.error("Sample, step, and chunk sizes must be positive.")
    if args.num_trajectories > args.num_samples:
        parser.error("num-trajectories cannot exceed num-samples.")
    if args.max_reference_images is not None and args.max_reference_images < 2:
        parser.error("max-reference-images must be at least two.")
    if any(not 0.0 <= time <= 1.0 for time in args.trajectory_times):
        parser.error("trajectory-times must lie in [0,1].")
    return args


@torch.no_grad()
def nearest_reference_indices(
    queries: np.ndarray,
    references: torch.Tensor,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact Euclidean neighbors without materializing [B,N_ref,d]."""
    if queries.ndim != 2 or queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must have shapes [B,d] and [N,d].")
    if len(queries) == 0 or len(references) == 0 or chunk_size < 1:
        raise ValueError("Queries, references, and chunk size must be non-empty.")
    query = torch.as_tensor(
        np.ascontiguousarray(queries),
        device=references.device,
        dtype=references.dtype,
    )
    query_norm = query.square().sum(dim=1)
    best_distance = torch.full_like(query_norm, torch.inf)
    best_index = torch.zeros(len(query), device=references.device, dtype=torch.long)
    for start in range(0, len(references), chunk_size):
        stop = min(start + chunk_size, len(references))
        candidates = references[start:stop]
        squared_distance = (
            query_norm[:, None]
            + candidates.square().sum(dim=1)[None, :]
            - 2.0 * query @ candidates.T
        ).clamp_min_(0.0)
        chunk_distance, chunk_index = squared_distance.min(dim=1)
        improved = chunk_distance < best_distance
        best_distance = torch.where(improved, chunk_distance, best_distance)
        best_index = torch.where(improved, chunk_index + start, best_index)
    return best_index.cpu().numpy(), best_distance.cpu().numpy()


def plot_gallery(
    generated: np.ndarray,
    neighbors: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_labels: np.ndarray,
    pixel_mse: np.ndarray,
    out_path: Path,
) -> None:
    pairs_per_row = 4
    rows = math.ceil(len(generated) / pairs_per_row)
    figure, axes = plt.subplots(
        rows,
        2 * pairs_per_row,
        figsize=(2.0 * 2 * pairs_per_row, 2.35 * rows),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for sample_index, image in enumerate(generated):
        row = sample_index // pairs_per_row
        pair = sample_index % pairs_per_row
        generated_axis = axes[row, 2 * pair]
        neighbor_axis = axes[row, 2 * pair + 1]
        generated_axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        neighbor_axis.imshow(neighbors[sample_index], cmap="gray", vmin=0.0, vmax=1.0)
        generated_axis.set_title(f"sample {sample_index}\ngenerated", fontsize=9)
        neighbor_axis.set_title(
            f"nearest train: {neighbor_labels[sample_index]}\n"
            f"idx={neighbor_indices[sample_index]}, MSE={pixel_mse[sample_index]:.2e}",
            fontsize=8,
        )
    figure.suptitle(
        "Full adaptive r*(t): generated samples and exact pixel-L2 neighbors",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(
    trajectory_images: dict[int, np.ndarray],
    state_indices: tuple[int, ...],
    num_steps: int,
    neighbors: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_labels: np.ndarray,
    pixel_mse: np.ndarray,
    out_path: Path,
) -> None:
    num_samples = len(neighbors)
    figure, axes = plt.subplots(
        num_samples,
        len(state_indices) + 1,
        figsize=(1.75 * (len(state_indices) + 1), 1.72 * num_samples),
        squeeze=False,
    )
    for row in range(num_samples):
        for column, state_index in enumerate(state_indices):
            axis = axes[row, column]
            axis.imshow(
                trajectory_images[state_index][row], cmap="gray", vmin=0.0, vmax=1.0
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t={state_index / num_steps:.2f}")
            if column == 0:
                axis.set_ylabel(
                    f"sample {row}", rotation=0, ha="right", va="center"
                )
        neighbor_axis = axes[row, -1]
        neighbor_axis.imshow(neighbors[row], cmap="gray", vmin=0.0, vmax=1.0)
        neighbor_axis.set_xticks([])
        neighbor_axis.set_yticks([])
        if row == 0:
            neighbor_axis.set_title("nearest train")
        neighbor_axis.set_xlabel(
            f"label={neighbor_labels[row]}, idx={neighbor_indices[row]}\n"
            f"MSE={pixel_mse[row]:.2e}",
            fontsize=7,
        )
    figure.suptitle(
        "Full adaptive r*(t): Euler trajectories and final nearest neighbors",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.06, 0.0, 1.0, 0.98))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_neighbor_metrics(rows: list[dict[str, object]], out_dir: Path) -> None:
    csv_path = out_dir / "full_adaptive_rstar_neighbor_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "full_adaptive_rstar_neighbor_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    numpy_dtype = np.float32 if args.dtype == "float32" else np.float64

    calibration = load_calibration_config(args.radius_metrics)
    normalization = calibration["normalization"]
    if not isinstance(normalization, dict):
        raise TypeError("Calibration normalization metadata must be a dictionary.")
    global_mean = float(normalization["global_mean"])
    global_std = float(normalization["global_std"])
    if global_std <= 0.0:
        raise ValueError("The calibration standard deviation must be positive.")

    train = load_mnist_32(
        args.data_root,
        "train",
        batch_size=1024,
        max_samples=args.max_reference_images,
        download=args.download,
    )
    normalized_images = np.ascontiguousarray(
        (train.images.astype(numpy_dtype) - global_mean) / global_std
    )
    transform = WaveletTransform2D(
        str(calibration["wavelet"]),
        int(calibration["level"]),
        str(calibration["mode"]),
    )
    reference_Y0, layout = transform.analyze(normalized_images)
    del normalized_images
    schedule = FullRadiusSchedule.from_metrics_csv(args.radius_metrics, layout)

    group_cache = {}

    def full_groups(time: float):
        radii = schedule.radii_at(time)
        signature = tuple(radii[subband] for subband in schedule.subbands)
        if signature not in group_cache:
            group_cache[signature] = build_radius_group_families(layout, radii)["full"]
        return group_cache[signature]

    estimator = MaskedEmpiricalWaveletVelocity(
        family="independent",
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        group_chunk_size=args.group_chunk_size,
        device=device,
        dtype=torch_dtype,
        endpoint_epsilon=1e-8,
    ).fit_with_groups(reference_Y0, layout, full_groups(0.0))

    rng = np.random.default_rng(args.seed)
    initial_Y = rng.standard_normal(
        (args.num_samples, layout.num_coefficients)
    ).astype(numpy_dtype)
    state = initial_Y.copy()
    state_indices = snapshot_indices(args.trajectory_times, args.num_steps)
    trajectory_Y = {
        0: state[: args.num_trajectories].copy()
    } if 0 in state_indices else {}
    step_size = 1.0 / args.num_steps
    step_timings = []
    print(
        f"device={device} dtype={args.dtype} references={len(reference_Y0)} "
        f"samples={args.num_samples} steps={args.num_steps}",
        flush=True,
    )
    for step in range(args.num_steps):
        time = step / args.num_steps
        started = perf_counter()
        estimator.set_groups(full_groups(time))
        state += step_size * estimator.predict_velocity(state, time)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite Euler state at t={time}.")
        state_index = step + 1
        if state_index in state_indices:
            trajectory_Y[state_index] = state[: args.num_trajectories].copy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        step_timings.append(elapsed)
        print(
            f"step {state_index:03d}/{args.num_steps} t={time:.4f} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    neighbor_indices, squared_distances = nearest_reference_indices(
        state,
        estimator.reference_tensor,
        args.neighbor_chunk_size,
    )
    normalized_mse = squared_distances / layout.num_coefficients
    pixel_mse = normalized_mse * global_std**2
    neighbor_labels = train.labels[neighbor_indices]
    neighbor_images = train.images[neighbor_indices]

    generated_normalized = transform.synthesize(state, layout)
    generated_unclipped = generated_normalized * global_std + global_mean
    generated_display = np.clip(generated_unclipped, 0.0, 1.0)
    trajectory_images = {
        index: display_images(
            transform.synthesize(values, layout), global_mean, global_std
        )
        for index, values in trajectory_Y.items()
    }

    plot_gallery(
        generated_display,
        neighbor_images,
        neighbor_indices,
        neighbor_labels,
        pixel_mse,
        args.out_dir / "full_adaptive_rstar_gallery.png",
    )
    plot_trajectories(
        trajectory_images,
        state_indices,
        args.num_steps,
        neighbor_images[: args.num_trajectories],
        neighbor_indices[: args.num_trajectories],
        neighbor_labels[: args.num_trajectories],
        pixel_mse[: args.num_trajectories],
        args.out_dir / "full_adaptive_rstar_trajectories.png",
    )

    rows = [
        {
            "sample_index": sample_index,
            "nearest_train_index": int(neighbor_indices[sample_index]),
            "nearest_train_label": int(neighbor_labels[sample_index]),
            "normalized_pixel_mse": float(normalized_mse[sample_index]),
            "pixel_mse_0_1_unclipped": float(pixel_mse[sample_index]),
            "pixel_l2_0_1_unclipped": float(
                math.sqrt(pixel_mse[sample_index] * layout.num_coefficients)
            ),
        }
        for sample_index in range(args.num_samples)
    ]
    write_neighbor_metrics(rows, args.out_dir)
    np.savez_compressed(
        args.out_dir / "full_adaptive_rstar_gallery_data.npz",
        initial_wavelet_noise=initial_Y,
        final_wavelet_state=state,
        generated_images_unclipped=generated_unclipped,
        generated_images_display=generated_display,
        trajectory_state_indices=np.asarray(state_indices, dtype=np.int64),
        trajectory_wavelet_states=np.stack(
            [trajectory_Y[index] for index in state_indices], axis=1
        ),
        nearest_train_indices=neighbor_indices,
        nearest_train_labels=neighbor_labels,
        nearest_train_images=neighbor_images,
        pixel_mse_0_1_unclipped=pixel_mse,
    )
    config = {
        "dataset": "MNIST-32 training empirical distribution",
        "num_reference_images": len(reference_Y0),
        "family": "full",
        "radius_strategy": "full-family calibrated r*(t), nearest calibrated time",
        "radius_metrics": str(args.radius_metrics),
        "wavelet": layout.wavelet_name,
        "level": layout.decomposition_level,
        "mode": layout.mode,
        "path": "Y_t=(1-t)xi+tY_0",
        "sampler": "explicit left Euler",
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "num_trajectories": args.num_trajectories,
        "trajectory_times_requested": list(args.trajectory_times),
        "trajectory_state_indices": list(state_indices),
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "neighbor_metric": (
            "exact squared L2 on unclipped globally normalized pixels; "
            "computed equivalently in the orthonormal wavelet basis"
        ),
        "mean_step_seconds": float(np.mean(step_timings)),
        "total_generation_seconds": float(np.sum(step_timings)),
    }
    (args.out_dir / "full_adaptive_rstar_gallery_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(
        f"nearest pixel MSE: mean={pixel_mse.mean():.3e}, "
        f"min={pixel_mse.min():.3e}, max={pixel_mse.max():.3e}",
        flush=True,
    )
    print(f"saved gallery to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
