"""Calibrate RGB wavelet radii against a pretrained CIFAR-10 FM U-Net.

Inputs: CIFAR-10 reference and disjoint calibration queries, plus a compatible
FM U-Net checkpoint. Outputs: radius map, teacher-MSE curves and shared-noise
generation grids under the selected result directory. Example::

    python -m wavelet_fm_structure.run_cifar10_fm_calibrated_radius --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_fm_wiener_operator,
    load_fm_unet,
    load_or_compute_wiener_svd,
    predict_fm_unet_velocity,
    predict_wiener_velocity,
)
from local_diffusion.configuration import load_config  # noqa: E402
from local_diffusion.data import build_dataset  # noqa: E402
from local_diffusion.data.datasets import DatasetBundle  # noqa: E402
from wavelet_fm_structure.adaptive_radius import (  # noqa: E402
    RADIUS_FAMILIES,
    FullRadiusSchedule,
    select_smallest_radius_within_best,
)
from wavelet_fm_structure.generate_full_adaptive_gallery import (  # noqa: E402
    nearest_reference_indices,
)
from wavelet_fm_structure.masks import (  # noqa: E402
    RadiusMaskFamily,
    build_central_targets,
    max_nonaliasing_radius,
)
from wavelet_fm_structure.protocol import DEFAULT_PROTOCOL  # noqa: E402
from wavelet_fm_structure.rgb_wavelets import (  # noqa: E402
    analyze_rgb,
    build_rgb_radius_group_families,
    build_rgb_subband_radius_groups,
    flatten_rgb,
    synthesize_rgb,
    unflatten_rgb,
)
from wavelet_score.indexing import WaveletLayout  # noqa: E402
from wavelet_score.mask_groups import WaveletMaskGroup  # noqa: E402
from wavelet_score.masked_empirical_fm import (  # noqa: E402
    MaskedEmpiricalWaveletVelocity,
)
from wavelet_score.transforms import WaveletTransform2D  # noqa: E402


FAMILY_TITLES: dict[RadiusMaskFamily, str] = {
    "same_band": "Same band",
    "orientation_augmented": "Orientation",
    "scale_augmented": "Scale",
    "full": "Full",
}
FAMILY_COLORS: dict[RadiusMaskFamily, str] = {
    "same_band": "#0072B2",
    "orientation_augmented": "#D55E00",
    "scale_augmented": "#009E73",
    "full": "#CC79A7",
}
MSE_TITLES: dict[str, str] = {
    **FAMILY_TITLES,
    "wiener_velocity": "Affine Wiener velocity",
}
MSE_COLORS: dict[str, str] = {
    **FAMILY_COLORS,
    "wiener_velocity": "#4D4D4D",
}
NUM_CHANNELS = 3
BASELINE_TITLES = {
    "full": r"Full wavelet ($r_b^*(t)$)",
    "fm_unet": "Pretrained FM U-Net",
    "wiener_velocity": "Affine Wiener velocity",
}


def parse_times(text: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("times must be comma-separated floats") from error
    if not times or any(not 0.0 < time < 1.0 for time in times):
        raise argparse.ArgumentTypeError("times must lie strictly inside (0,1)")
    if tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError("times must be unique and increasing")
    return times


def parse_trajectory_times(text: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "trajectory times must be comma-separated floats"
        ) from error
    if not times or any(not 0.0 <= time <= 1.0 for time in times):
        raise argparse.ArgumentTypeError("trajectory times must lie in [0,1]")
    if tuple(sorted(set(times))) != times:
        raise argparse.ArgumentTypeError(
            "trajectory times must be unique and increasing"
        )
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/cifar10_fm_calibrated_radius"
        ),
    )
    parser.add_argument("--times", type=parse_times, default=DEFAULT_PROTOCOL.times)
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument("--num-calibration-queries", type=int, default=32)
    parser.add_argument("--num-mse-queries", type=int, default=128)
    parser.add_argument("--num-generation-samples", type=int, default=8)
    parser.add_argument("--num-trajectory-samples", type=int, default=2)
    parser.add_argument(
        "--trajectory-times",
        type=parse_trajectory_times,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--data-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--reference-chunk-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=16)
    parser.add_argument("--neighbor-chunk-size", type=int, default=8192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    positive = (
        args.num_calibration_queries,
        args.num_mse_queries,
        args.num_generation_samples,
        args.num_trajectory_samples,
        args.num_steps,
        args.data_batch_size,
        args.model_batch_size,
        args.reference_chunk_size,
        args.query_batch_size,
        args.group_chunk_size,
        args.neighbor_chunk_size,
    )
    if any(value < 1 for value in positive):
        parser.error("Query, sample, step, batch, and chunk sizes must be positive.")
    if args.max_reference_images is not None and args.max_reference_images < 2:
        parser.error("max-reference-images must be at least two.")
    if args.num_trajectory_samples > args.num_generation_samples:
        parser.error("num-trajectory-samples cannot exceed num-generation-samples.")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative.")
    return args


def snapshot_indices(times: tuple[float, ...], num_steps: int) -> tuple[int, ...]:
    """Map requested times to unique nearest Euler state indices."""
    return tuple(sorted({int(round(time * num_steps)) for time in times}))


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_cifar10_split(
    split: str,
    batch_size: int,
    max_images: int | None,
    download: bool,
    num_workers: int,
) -> DatasetBundle:
    overrides = [
        "metrics.wandb.enabled=false",
        f"dataset.split={split}",
        f"dataset.batch_size={batch_size}",
        f"dataset.download={'true' if download else 'false'}",
        f"dataset.num_workers={num_workers}",
    ]
    if max_images is not None:
        overrides.append(f"dataset.subset_size={max_images}")
    cfg = load_config("configs/pca_locality/cifar10.yaml", overrides=overrides)
    bundle = build_dataset(cfg.dataset)
    if bundle.resolution != 32 or bundle.in_channels != NUM_CHANNELS:
        raise RuntimeError("This experiment requires RGB CIFAR-10 at 32x32.")
    return bundle


def collect_reference_wavelets(
    dataset: DatasetBundle,
    transform: WaveletTransform2D,
) -> tuple[np.ndarray, WaveletLayout]:
    """Transform the complete reference bank channelwise without a DWT matrix."""
    chunks: list[np.ndarray] = []
    shared_layout: WaveletLayout | None = None
    collected = 0
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        image_array = np.ascontiguousarray(images.cpu().numpy(), dtype=np.float32)
        if image_array.min() < -1.000001 or image_array.max() > 1.000001:
            raise RuntimeError("CIFAR-10 references are not normalized to [-1,1].")
        Y_rgb, layout = analyze_rgb(image_array, transform)
        if shared_layout is None:
            shared_layout = layout
        elif layout != shared_layout:
            raise RuntimeError("Reference batches produced inconsistent DWT layouts.")
        chunks.append(flatten_rgb(Y_rgb))
        collected += len(images)
        print(f"reference DWT: {collected:05d}/{len(dataset.dataset):05d}", flush=True)
    if not chunks or shared_layout is None:
        raise RuntimeError("The CIFAR-10 reference loader was empty.")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0)), shared_layout


def collect_indexed_images(dataset: DatasetBundle, indices: np.ndarray) -> np.ndarray:
    images: list[torch.Tensor] = []
    for index in indices:
        item = dataset.dataset[int(index)]
        image = item[0] if isinstance(item, (tuple, list)) else item
        images.append(image)
    result = torch.stack(images).numpy().astype(np.float32, copy=False)
    if result.shape[1:] != (NUM_CHANNELS, 32, 32):
        raise RuntimeError(f"Unexpected CIFAR-10 query shape: {result.shape}.")
    if result.min() < -1.000001 or result.max() > 1.000001:
        raise RuntimeError("CIFAR-10 queries are not normalized to [-1,1].")
    return np.ascontiguousarray(result)


def model_wavelet_state_and_velocity(
    model: torch.nn.Module,
    clean_images: np.ndarray,
    noise: np.ndarray,
    time: float,
    transform: WaveletTransform2D,
    device: torch.device,
    model_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct x_t in pixel space and transform the U-Net velocity channelwise."""
    states: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for start in range(0, len(clean_images), model_batch_size):
        stop = min(start + model_batch_size, len(clean_images))
        clean = torch.as_tensor(
            clean_images[start:stop], device=device, dtype=torch.float32
        )
        epsilon = torch.as_tensor(noise[start:stop], device=device, dtype=torch.float32)
        x_t = (1.0 - time) * epsilon + time * clean
        velocity = predict_fm_unet_velocity(model, x_t, time)
        if velocity.shape != x_t.shape or not torch.isfinite(velocity).all():
            raise RuntimeError("The pretrained FM U-Net returned an invalid velocity.")
        states.append(x_t.cpu().numpy())
        velocities.append(velocity.cpu().numpy())

    state_pixels = np.ascontiguousarray(np.concatenate(states, axis=0))
    velocity_pixels = np.ascontiguousarray(np.concatenate(velocities, axis=0))
    state_Y, state_layout = analyze_rgb(state_pixels, transform)
    velocity_Y, velocity_layout = analyze_rgb(velocity_pixels, transform)
    if state_layout != velocity_layout:
        raise RuntimeError("State and velocity DWT layouts differ.")
    return flatten_rgb(state_Y), flatten_rgb(velocity_Y)


def build_calibration_group_cache(
    layout: WaveletLayout,
) -> dict[tuple[str, int], tuple[WaveletMaskGroup, ...]]:
    cache: dict[tuple[str, int], tuple[WaveletMaskGroup, ...]] = {}
    for target in build_central_targets(layout):
        for radius in range(max_nonaliasing_radius(target) + 1):
            cache[(target.subband, radius)] = build_rgb_subband_radius_groups(
                layout,
                target.subband,
                radius,
                family="full",
                channels=NUM_CHANNELS,
            )
    return cache


def calibrate_radii(
    estimator: MaskedEmpiricalWaveletVelocity,
    model: torch.nn.Module,
    clean_images: np.ndarray,
    noise: np.ndarray,
    times: tuple[float, ...],
    transform: WaveletTransform2D,
    layout: WaveletLayout,
    device: torch.device,
    model_batch_size: int,
) -> tuple[FullRadiusSchedule, list[dict[str, object]]]:
    targets = build_central_targets(layout)
    subbands = tuple(target.subband for target in targets)
    groups = build_calibration_group_cache(layout)
    radius_rows: list[tuple[int, ...]] = []
    metrics: list[dict[str, object]] = []

    for time in times:
        started = perf_counter()
        print(f"\n=== radius calibration t={time:.2f} ===", flush=True)
        query_Yt, model_velocity_Y = model_wavelet_state_and_velocity(
            model,
            clean_images,
            noise,
            time,
            transform,
            device,
            model_batch_size,
        )
        selected_for_time: list[int] = []
        for target in targets:
            risks: dict[int, float] = {}
            target_rows: list[dict[str, object]] = []
            maximum = max_nonaliasing_radius(target)
            for radius in range(maximum + 1):
                output_indices, prediction = estimator.predict_velocity_subset(
                    query_Yt, time, groups[(target.subband, radius)]
                )
                expected_values = NUM_CHANNELS * math.prod(target.metadata.subband_shape)
                if len(output_indices) != expected_values:
                    raise RuntimeError(
                        f"{target.subband} should contain {expected_values} RGB outputs."
                    )
                risk = float(
                    np.mean(
                        (prediction - model_velocity_Y[:, output_indices]) ** 2,
                        dtype=np.float64,
                    )
                )
                if not np.isfinite(risk):
                    raise FloatingPointError("A radius calibration risk is non-finite.")
                risks[radius] = risk
                target_rows.append(
                    {
                        "t": time,
                        "subband": target.subband,
                        "radius": radius,
                        "velocity_mse_vs_fm_unet": risk,
                        "selected_radius": None,
                        "num_queries": len(clean_images),
                        "num_spatial_positions": math.prod(
                            target.metadata.subband_shape
                        ),
                        "num_channels": NUM_CHANNELS,
                    }
                )
                print(
                    f"  {target.subband} r={radius}/{maximum}: MSE={risk:.6e}",
                    flush=True,
                )
            selected = select_smallest_radius_within_best(risks, 0.01)
            for row in target_rows:
                row["selected_radius"] = selected
            metrics.extend(target_rows)
            selected_for_time.append(selected)
            print(f"  {target.subband}: selected r*={selected}", flush=True)
        radius_rows.append(tuple(selected_for_time))
        print(
            f"calibration t={time:.2f} elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )
    return FullRadiusSchedule(times, subbands, tuple(radius_rows)), metrics


def evaluate_families_against_model(
    estimator: MaskedEmpiricalWaveletVelocity,
    model: torch.nn.Module,
    clean_images: np.ndarray,
    noise: np.ndarray,
    schedule: FullRadiusSchedule,
    transform: WaveletTransform2D,
    layout: WaveletLayout,
    device: torch.device,
    model_batch_size: int,
) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    group_cache: dict[tuple[int, ...], dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]]] = {}
    for time in schedule.calibration_times:
        started = perf_counter()
        print(f"\n=== family MSE t={time:.2f} ===", flush=True)
        query_Yt, model_velocity_Y = model_wavelet_state_and_velocity(
            model,
            clean_images,
            noise,
            time,
            transform,
            device,
            model_batch_size,
        )
        radii = schedule.radii_at(time)
        signature = tuple(radii[subband] for subband in schedule.subbands)
        if signature not in group_cache:
            group_cache[signature] = build_rgb_radius_group_families(
                layout, radii, channels=NUM_CHANNELS
            )
        for family in RADIUS_FAMILIES:
            family_groups = group_cache[signature][family]
            estimator.set_groups(family_groups)
            prediction = estimator.predict_velocity(query_Yt, time)
            mse = float(
                np.mean((prediction - model_velocity_Y) ** 2, dtype=np.float64)
            )
            metrics.append(
                {
                    "t": time,
                    "family": family,
                    "velocity_mse_vs_fm_unet": mse,
                    "radii_by_subband": radii,
                    "num_queries": len(clean_images),
                }
            )
            print(f"  {FAMILY_TITLES[family]}: MSE={mse:.6e}", flush=True)
        print(
            f"family MSE t={time:.2f} elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )
    return metrics


@torch.no_grad()
def evaluate_wiener_against_model(
    model: torch.nn.Module,
    clean_images: np.ndarray,
    noise: np.ndarray,
    times: tuple[float, ...],
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    device: torch.device,
    model_batch_size: int,
) -> list[dict[str, object]]:
    """Measure affine Wiener velocity against the FM U-Net on matched states."""
    metrics: list[dict[str, object]] = []
    for time in times:
        started = perf_counter()
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, time)
        squared_error = 0.0
        num_values = 0
        for start in range(0, len(clean_images), model_batch_size):
            stop = min(start + model_batch_size, len(clean_images))
            clean = torch.as_tensor(
                clean_images[start:stop], device=device, dtype=torch.float32
            )
            epsilon = torch.as_tensor(
                noise[start:stop], device=device, dtype=torch.float32
            )
            x_t = (1.0 - time) * epsilon + time * clean
            model_velocity = predict_fm_unet_velocity(model, x_t, time)
            wiener_velocity = predict_wiener_velocity(
                x_t, time, wiener_operator, mean
            )
            difference = wiener_velocity - model_velocity
            squared_error += difference.square().sum(dtype=torch.float64).item()
            num_values += difference.numel()
        mse = squared_error / num_values
        metrics.append(
            {
                "t": time,
                "family": "wiener_velocity",
                "velocity_mse_vs_fm_unet": mse,
                "num_queries": len(clean_images),
                "measurement_space": (
                    "pixels; equal to channelwise orthonormal-wavelet MSE by Parseval"
                ),
            }
        )
        print(
            f"  Affine Wiener t={time:.2f}: MSE={mse:.6e} "
            f"elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )
    return metrics


def generate_samples(
    estimator: MaskedEmpiricalWaveletVelocity,
    schedule: FullRadiusSchedule,
    transform: WaveletTransform2D,
    layout: WaveletLayout,
    initial_pixels: np.ndarray,
    num_steps: int,
    trajectory_state_indices: tuple[int, ...],
    num_trajectory_samples: int,
) -> tuple[
    dict[RadiusMaskFamily, np.ndarray],
    dict[RadiusMaskFamily, np.ndarray],
    dict[int, np.ndarray],
]:
    """Generate all wavelet families and retain the Full trajectory."""
    expected_shape = (NUM_CHANNELS, *layout.image_shape)
    if initial_pixels.ndim != 4 or initial_pixels.shape[1:] != expected_shape:
        raise ValueError(
            f"initial_pixels must have shape [N, {expected_shape}], "
            f"got {initial_pixels.shape}."
        )
    if not 1 <= num_trajectory_samples <= len(initial_pixels):
        raise ValueError("Invalid number of trajectory samples.")
    if any(index < 0 or index > num_steps for index in trajectory_state_indices):
        raise ValueError("Trajectory state indices must lie in [0, num_steps].")

    initial_Y, initial_layout = analyze_rgb(initial_pixels, transform)
    if initial_layout != layout:
        raise RuntimeError("Initial noise produced an inconsistent wavelet layout.")
    initial_state = flatten_rgb(initial_Y)
    states = {family: initial_state.copy() for family in RADIUS_FAMILIES}
    full_snapshots: dict[int, np.ndarray] = {}
    if 0 in trajectory_state_indices:
        full_snapshots[0] = states["full"][:num_trajectory_samples].copy()
    group_cache: dict[tuple[int, ...], dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]]] = {}
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        time = step / num_steps
        started = perf_counter()
        radii = schedule.radii_at(time)
        signature = tuple(radii[subband] for subband in schedule.subbands)
        if signature not in group_cache:
            group_cache[signature] = build_rgb_radius_group_families(
                layout, radii, channels=NUM_CHANNELS
            )
        for family in RADIUS_FAMILIES:
            estimator.set_groups(group_cache[signature][family])
            states[family] += step_size * estimator.predict_velocity(
                states[family], time
            )
            if not np.isfinite(states[family]).all():
                raise FloatingPointError(
                    f"Non-finite {family} Euler state at t={time:.4f}."
                )
        state_index = step + 1
        if state_index in trajectory_state_indices:
            full_snapshots[state_index] = states["full"][:num_trajectory_samples].copy()
        if estimator.device.type == "cuda":
            torch.cuda.synchronize(estimator.device)
        print(
            f"generation step {step + 1:03d}/{num_steps} t={time:.4f} "
            f"elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )

    generated = {
        family: synthesize_rgb(
            unflatten_rgb(state, layout, channels=NUM_CHANNELS), layout, transform
        )
        for family, state in states.items()
    }
    full_trajectory = {
        index: synthesize_rgb(
            unflatten_rgb(state, layout, channels=NUM_CHANNELS), layout, transform
        )
        for index, state in full_snapshots.items()
    }
    return generated, states, full_trajectory


@torch.no_grad()
def generate_model_baselines(
    initial_pixels: np.ndarray,
    model: torch.nn.Module,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    num_steps: int,
    trajectory_state_indices: tuple[int, ...],
    num_trajectory_samples: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Generate U-Net and affine-Wiener baselines under the same Euler protocol."""
    initial = torch.as_tensor(initial_pixels, device=device, dtype=torch.float32)
    states = {
        "fm_unet": initial.clone(),
        "wiener_velocity": initial.clone(),
    }
    snapshots: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in states
    }
    if 0 in trajectory_state_indices:
        for name, state in states.items():
            snapshots[name][0] = (
                state[:num_trajectory_samples].detach().cpu().numpy().copy()
            )

    step_size = 1.0 / num_steps
    for step in range(num_steps):
        time = step / num_steps
        started = perf_counter()
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, time)
        states["fm_unet"].add_(
            step_size * predict_fm_unet_velocity(model, states["fm_unet"], time)
        )
        states["wiener_velocity"].add_(
            step_size
            * predict_wiener_velocity(
                states["wiener_velocity"], time, wiener_operator, mean
            )
        )
        for name, state in states.items():
            if not torch.isfinite(state).all():
                raise FloatingPointError(
                    f"Non-finite {name} Euler state at t={time:.4f}."
                )

        state_index = step + 1
        if state_index in trajectory_state_indices:
            for name, state in states.items():
                snapshots[name][state_index] = (
                    state[:num_trajectory_samples].detach().cpu().numpy().copy()
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"baseline step {state_index:03d}/{num_steps} t={time:.4f} "
            f"elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )

    generated = {
        name: state.detach().cpu().numpy().copy() for name, state in states.items()
    }
    return generated, snapshots


def display_rgb(normalized_images: np.ndarray) -> np.ndarray:
    return np.clip((normalized_images + 1.0) / 2.0, 0.0, 1.0)


def load_neighbor_images(
    dataset: DatasetBundle,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index in indices:
        item = dataset.dataset[int(index)]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError("CIFAR-10 samples must contain an image and label.")
        images.append(item[0].numpy())
        labels.append(int(item[1]))
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def plot_radius_heatmap(
    schedule: FullRadiusSchedule,
    out_path: Path,
) -> None:
    values = np.asarray(schedule.radii, dtype=np.int64).T
    maximum = int(values.max())
    boundaries = np.arange(-0.5, maximum + 1.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=256)
    figure, axis = plt.subplots(figsize=(10.2, 6.6), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", norm=norm)
    axis.set(
        xticks=np.arange(len(schedule.calibration_times)),
        xticklabels=[f"{time:.2f}" for time in schedule.calibration_times],
        yticks=np.arange(len(schedule.subbands)),
        yticklabels=schedule.subbands,
        xlabel="Flow time t (noise to data)",
        ylabel="RGB wavelet subband (coarse to fine)",
        title="Smallest Full radius within 1% of the best FM U-Net MSE",
    )
    middle = maximum / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = int(values[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value < middle else "black",
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, ticks=range(maximum + 1))
    colorbar.set_label(r"Shared RGB radius $r_b^*(t)$")
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def plot_model_mse(
    metrics: list[dict[str, object]],
    out_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    family_order: list[str] = list(RADIUS_FAMILIES)
    if any(row["family"] == "wiener_velocity" for row in metrics):
        family_order.append("wiener_velocity")
    for family in family_order:
        rows = sorted(
            (row for row in metrics if row["family"] == family),
            key=lambda row: float(row["t"]),
        )
        axis.plot(
            [float(row["t"]) for row in rows],
            [float(row["velocity_mse_vs_fm_unet"]) for row in rows],
            color=MSE_COLORS[family],
            marker="s" if family == "wiener_velocity" else "o",
            linestyle="--" if family == "wiener_velocity" else "-",
            linewidth=2.1,
            label=MSE_TITLES[family],
        )
    axis.set(
        xlabel="Flow time t (noise to data)",
        ylabel=r"MSE $\mathbb{E}_{x_t,b,k,c}[(\hat v-v_\theta)^2]$",
        title="RGB wavelet closed forms versus pretrained CIFAR-10 FM U-Net",
    )
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.24)
    axis.legend(title=r"Shared calibrated $r_b^*(t)$", frameon=False)
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def plot_generations_and_neighbors(
    generated: dict[RadiusMaskFamily, np.ndarray],
    neighbor_images: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_labels: np.ndarray,
    neighbor_mse: np.ndarray,
    out_path: Path,
) -> None:
    sample_count = len(generated["full"])
    columns = (*RADIUS_FAMILIES, "nearest")
    figure, axes = plt.subplots(
        sample_count,
        len(columns),
        figsize=(2.3 * len(columns), 2.15 * sample_count),
        squeeze=False,
    )
    displayed = {family: display_rgb(images) for family, images in generated.items()}
    displayed_neighbors = display_rgb(neighbor_images)
    for row in range(sample_count):
        for column, family in enumerate(RADIUS_FAMILIES):
            axis = axes[row, column]
            axis.imshow(np.transpose(displayed[family][row], (1, 2, 0)))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(FAMILY_TITLES[family])
        neighbor_axis = axes[row, -1]
        neighbor_axis.imshow(np.transpose(displayed_neighbors[row], (1, 2, 0)))
        neighbor_axis.set_xticks([])
        neighbor_axis.set_yticks([])
        if row == 0:
            neighbor_axis.set_title("Full nearest train")
        neighbor_axis.set_xlabel(
            f"idx={neighbor_indices[row]}, label={neighbor_labels[row]}\n"
            f"pixel MSE={neighbor_mse[row]:.3e}",
            fontsize=7,
        )
        axes[row, 0].set_ylabel(f"sample {row}", rotation=0, ha="right", va="center")
    figure.suptitle(
        "CIFAR-10 empirical FM generation with shared model-calibrated radii"
    )
    figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.985))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_baseline_generation_comparison(
    full_generated: np.ndarray,
    baseline_generated: dict[str, np.ndarray],
    num_steps: int,
    out_path: Path,
) -> None:
    """Compare final samples from Full, the FM U-Net, and affine Wiener."""
    images_by_model = {
        "full": full_generated,
        "fm_unet": baseline_generated["fm_unet"],
        "wiener_velocity": baseline_generated["wiener_velocity"],
    }
    sample_count = len(full_generated)
    if any(len(images) != sample_count for images in images_by_model.values()):
        raise ValueError("Baseline generation batches have inconsistent sizes.")

    model_order = tuple(images_by_model)
    figure, axes = plt.subplots(
        sample_count,
        len(model_order),
        figsize=(2.5 * len(model_order), 2.2 * sample_count),
        squeeze=False,
    )
    displayed = {
        name: display_rgb(images) for name, images in images_by_model.items()
    }
    for row in range(sample_count):
        for column, name in enumerate(model_order):
            axis = axes[row, column]
            axis.imshow(np.transpose(displayed[name][row], (1, 2, 0)))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(BASELINE_TITLES[name], fontsize=10)
        axes[row, 0].set_ylabel(
            f"sample {row}", rotation=0, ha="right", va="center", fontsize=8
        )
    figure.suptitle(
        f"Same initial noise and same {num_steps}-step explicit Euler solver",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.05, 0.0, 1.0, 0.985))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_baseline_trajectories(
    full_trajectory: dict[int, np.ndarray],
    baseline_trajectories: dict[str, dict[int, np.ndarray]],
    state_indices: tuple[int, ...],
    num_steps: int,
    num_samples: int,
    out_path: Path,
) -> None:
    """Plot matched-noise trajectories for the three velocity estimators."""
    trajectories = {
        "full": full_trajectory,
        "fm_unet": baseline_trajectories["fm_unet"],
        "wiener_velocity": baseline_trajectories["wiener_velocity"],
    }
    expected_indices = set(state_indices)
    for name, snapshots in trajectories.items():
        if set(snapshots) != expected_indices:
            raise ValueError(f"Incomplete trajectory snapshots for {name}.")
        if any(len(images) != num_samples for images in snapshots.values()):
            raise ValueError(f"Invalid trajectory batch size for {name}.")

    model_order = tuple(trajectories)
    row_count = num_samples * len(model_order)
    figure, axes = plt.subplots(
        row_count,
        len(state_indices),
        figsize=(2.0 * len(state_indices), 1.8 * row_count),
        squeeze=False,
    )
    for sample_index in range(num_samples):
        for model_offset, name in enumerate(model_order):
            row = sample_index * len(model_order) + model_offset
            for column, state_index in enumerate(state_indices):
                axis = axes[row, column]
                image = display_rgb(trajectories[name][state_index])[sample_index]
                axis.imshow(np.transpose(image, (1, 2, 0)))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"t={state_index / num_steps:.2f}", fontsize=10)
            axes[row, 0].set_ylabel(
                f"sample {sample_index}\n{BASELINE_TITLES[name]}",
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
            )
    figure.suptitle(
        f"Shared-noise generation trajectories ({num_steps}-step explicit Euler)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.11, 0.0, 1.0, 0.985))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"FM checkpoint not found: {checkpoint}")

    print("dataset=CIFAR-10 train bank / test calibration", flush=True)
    print("normalization=x -> 2x-1", flush=True)
    print("time_direction=t=0 noise, t=1 data", flush=True)
    print("path=x_t=(1-t) epsilon + t x_1", flush=True)
    print("model_output=velocity x_1-epsilon", flush=True)
    print("checkpoint_state=ema_model model_call=model(t_batch, x_t)", flush=True)
    print(f"device={device} checkpoint={checkpoint}", flush=True)

    train = load_cifar10_split(
        "train",
        args.data_batch_size,
        args.max_reference_images,
        args.download,
        args.num_workers,
    )
    test = load_cifar10_split(
        "test", args.data_batch_size, None, args.download, args.num_workers
    )
    if args.num_calibration_queries + args.num_mse_queries > len(test.dataset):
        raise ValueError("Calibration and MSE query sets exceed the CIFAR-10 test split.")

    transform = WaveletTransform2D(
        DEFAULT_PROTOCOL.wavelet_name,
        DEFAULT_PROTOCOL.decomposition_level,
        DEFAULT_PROTOCOL.mode,
    )
    reference_Y0, layout = collect_reference_wavelets(train, transform)
    reference_count = len(reference_Y0)
    zero_radii = {target.subband: 0 for target in build_central_targets(layout)}
    initial_groups = build_rgb_radius_group_families(
        layout, zero_radii, channels=NUM_CHANNELS
    )["full"]
    estimator = MaskedEmpiricalWaveletVelocity(
        family="independent",
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        group_chunk_size=args.group_chunk_size,
        device=device,
        dtype=torch.float32,
        endpoint_epsilon=1e-6,
    ).fit_with_groups(reference_Y0, layout, initial_groups)
    del reference_Y0

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(test.dataset))
    calibration_indices = permutation[: args.num_calibration_queries]
    mse_indices = permutation[
        args.num_calibration_queries : args.num_calibration_queries
        + args.num_mse_queries
    ]
    calibration_clean = collect_indexed_images(test, calibration_indices)
    mse_clean = collect_indexed_images(test, mse_indices)
    calibration_noise = rng.standard_normal(calibration_clean.shape).astype(np.float32)
    mse_noise = rng.standard_normal(mse_clean.shape).astype(np.float32)

    model = load_fm_unet(checkpoint, device, "cifar10")
    schedule, calibration_metrics = calibrate_radii(
        estimator,
        model,
        calibration_clean,
        calibration_noise,
        args.times,
        transform,
        layout,
        device,
        args.model_batch_size,
    )
    plot_radius_heatmap(schedule, args.out_dir / "selected_radius_heatmap.png")

    mse_metrics = evaluate_families_against_model(
        estimator,
        model,
        mse_clean,
        mse_noise,
        schedule,
        transform,
        layout,
        device,
        args.model_batch_size,
    )
    del calibration_clean, calibration_noise
    print("\n=== affine Wiener MSE baseline ===", flush=True)
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(train, device)
    mse_metrics.extend(
        evaluate_wiener_against_model(
            model,
            mse_clean,
            mse_noise,
            args.times,
            U,
            eigenvalues,
            Vh,
            mean,
            device,
            args.model_batch_size,
        )
    )
    plot_model_mse(mse_metrics, args.out_dir / "mse_vs_fm_unet.png")
    del mse_clean, mse_noise
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trajectory_state_indices = snapshot_indices(
        args.trajectory_times, args.num_steps
    )
    generation_rng = np.random.default_rng(args.seed)
    initial_pixels = generation_rng.standard_normal(
        (args.num_generation_samples, NUM_CHANNELS, *layout.image_shape)
    ).astype(np.float32)

    generated, final_wavelet_states, full_trajectory = generate_samples(
        estimator,
        schedule,
        transform,
        layout,
        initial_pixels,
        args.num_steps,
        trajectory_state_indices,
        args.num_trajectory_samples,
    )
    baseline_generated, baseline_trajectories = generate_model_baselines(
        initial_pixels,
        model,
        U,
        eigenvalues,
        Vh,
        mean,
        args.num_steps,
        trajectory_state_indices,
        args.num_trajectory_samples,
        device,
    )
    plot_baseline_generation_comparison(
        generated["full"],
        baseline_generated,
        args.num_steps,
        args.out_dir / "full_unet_wiener_comparison.png",
    )
    plot_baseline_trajectories(
        full_trajectory,
        baseline_trajectories,
        trajectory_state_indices,
        args.num_steps,
        args.num_trajectory_samples,
        args.out_dir / "full_unet_wiener_trajectories.png",
    )
    del model, U, eigenvalues, Vh, mean, baseline_generated, baseline_trajectories
    if device.type == "cuda":
        torch.cuda.empty_cache()

    neighbor_indices, squared_distances = nearest_reference_indices(
        final_wavelet_states["full"],
        estimator.reference_tensor,
        args.neighbor_chunk_size,
    )
    neighbor_images, neighbor_labels = load_neighbor_images(train, neighbor_indices)
    pixel_mse_01 = squared_distances / (4.0 * NUM_CHANNELS * layout.num_coefficients)
    plot_generations_and_neighbors(
        generated,
        neighbor_images,
        neighbor_indices,
        neighbor_labels,
        pixel_mse_01,
        args.out_dir / "generations_and_full_neighbors.png",
    )

    results = {
        "protocol": {
            "dataset": "CIFAR-10",
            "reference_split": "train",
            "calibration_split": "test",
            "mse_split": "test, disjoint from calibration queries",
            "num_reference_images": reference_count,
            "num_calibration_queries": args.num_calibration_queries,
            "num_mse_queries": args.num_mse_queries,
            "normalization": "per-channel x -> (x-0.5)/0.5, range [-1,1]",
            "reference_augmentation": "none; canonical CIFAR-10 train images",
            "time_direction": "t=0 Gaussian noise, t=1 clean data",
            "path": "x_t=(1-t)epsilon+t*x_1",
            "fm_model_output": "velocity x_1-epsilon",
            "fm_model_call": "model(t_batch, x_t)",
            "checkpoint_state": "ema_model at training step 400000",
            "checkpoint": str(checkpoint),
            "wavelet": layout.wavelet_name,
            "level": layout.decomposition_level,
            "mode": layout.mode,
            "rgb_layout": "independent channel DWT; channel-major flattening",
            "mask_rule": "a visible spatial site exposes all three RGB channels",
            "radius_rule": (
                "shared across RGB; smallest r with R_b(r,t) <= "
                "1.01 * min_r' R_b(r',t), calibrated with Full"
            ),
            "families": list(RADIUS_FAMILIES),
            "mse_baseline": "affine Wiener velocity including empirical-mean bias",
            "sampler": (
                "same explicit left-Euler time grid for all methods; Full is "
                "integrated in RGB wavelet coordinates and U-Net/Wiener in pixels"
            ),
            "num_steps": args.num_steps,
            "num_generation_samples": args.num_generation_samples,
            "num_trajectory_samples": args.num_trajectory_samples,
            "requested_trajectory_times": list(args.trajectory_times),
            "trajectory_state_indices": list(trajectory_state_indices),
            "realized_trajectory_times": [
                index / args.num_steps for index in trajectory_state_indices
            ],
            "generation_baselines": [
                "Full wavelet with calibrated radii",
                "pretrained FM U-Net",
                "affine Wiener velocity",
            ],
            "affine_wiener_definition": (
                "x1_hat=W_t*x_t+(I-t*W_t)*mu; "
                "velocity=(x1_hat-x_t)/(1-t)"
            ),
            "same_initial_noise_across_families_and_baselines": True,
            "neighbor_metric": "exact unclipped pixel L2 over the reference train bank",
            "device": str(device),
            "dtype": "float32",
            "seed": args.seed,
            "calibration_query_indices": calibration_indices.tolist(),
            "mse_query_indices": mse_indices.tolist(),
        },
        "selected_radii": {
            f"{time:.8g}": dict(zip(schedule.subbands, row, strict=True))
            for time, row in zip(
                schedule.calibration_times, schedule.radii, strict=True
            )
        },
        "radius_calibration": calibration_metrics,
        "mse_vs_fm_unet": mse_metrics,
        "full_neighbors": [
            {
                "sample": index,
                "train_index": int(neighbor_indices[index]),
                "label": int(neighbor_labels[index]),
                "pixel_mse_0_1_unclipped": float(pixel_mse_01[index]),
            }
            for index in range(args.num_generation_samples)
        ],
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved requested outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
