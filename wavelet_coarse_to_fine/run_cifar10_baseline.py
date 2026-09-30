"""Evaluate the hand-designed coarse-to-fine wavelet FM baseline on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

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
from wavelet_coarse_to_fine.masks import (  # noqa: E402
    CoarseToFineSchedule,
    TimeRegime,
    build_coarse_to_fine_rgb_groups,
    build_rgb_region_indices,
    scale_region,
)
from wavelet_fm_structure.adaptive_radius import FullRadiusSchedule  # noqa: E402
from wavelet_fm_structure.generate_full_adaptive_gallery import (  # noqa: E402
    nearest_reference_indices,
)
from wavelet_fm_structure.masks import build_central_targets  # noqa: E402
from wavelet_fm_structure.protocol import DEFAULT_PROTOCOL  # noqa: E402
from wavelet_fm_structure.rgb_wavelets import (  # noqa: E402
    analyze_rgb,
    build_rgb_radius_group_families,
    flatten_rgb,
    synthesize_rgb,
    unflatten_rgb,
)
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import (  # noqa: E402
    collect_indexed_images,
    collect_reference_wavelets,
    display_rgb,
    load_cifar10_split,
    load_neighbor_images,
    model_wavelet_state_and_velocity,
    parse_times,
    parse_trajectory_times,
    resolve_device,
    snapshot_indices,
)
from wavelet_score.indexing import WaveletLayout, build_metadata  # noqa: E402
from wavelet_score.mask_groups import WaveletMaskGroup  # noqa: E402
from wavelet_score.masked_empirical_fm import (  # noqa: E402
    MaskedEmpiricalWaveletVelocity,
)
from wavelet_score.transforms import WaveletTransform2D  # noqa: E402


NUM_CHANNELS = 3
ESTIMATORS = ("coarse_to_fine", "mask_a", "wiener_velocity")
METHODS = ("fm_unet", "coarse_to_fine", "mask_a", "wiener_velocity")
REGIONS = ("all", "coarse", "medium", "fine")
METHOD_TITLES = {
    "fm_unet": "Pretrained FM U-Net",
    "coarse_to_fine": "Coarse-to-fine mask",
    "mask_a": "Mask A: adaptive same-band",
    "wiener_velocity": "Affine Wiener velocity",
}
METHOD_COLORS = {
    "fm_unet": "#111111",
    "coarse_to_fine": "#0072B2",
    "mask_a": "#D55E00",
    "wiener_velocity": "#009E73",
}
REGION_TITLES = {
    "all": "All wavelet coefficients",
    "coarse": r"Coarse $C=(A_3,D_3)$",
    "medium": r"Medium $M=D_2$",
    "fine": r"Fine $F=D_1$",
}
DEFAULT_TIMES = (0.05, 0.15, 0.32, 0.34, 0.50, 0.65, 0.68, 0.85, 0.95)
DEFAULT_TRAJECTORY_TIMES = (0.0, 0.25, 0.32, 0.34, 0.55, 0.65, 0.68, 0.85, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mask-a-metrics",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/cifar10_fm_calibrated_radius/metrics.json"
        ),
        help="Existing radius calibration used by the same-band Mask A baseline.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_coarse_to_fine/results/cifar10_baseline"),
    )
    parser.add_argument("--times", type=parse_times, default=DEFAULT_TIMES)
    parser.add_argument("--num-train-queries", type=int, default=128)
    parser.add_argument("--num-test-queries", type=int, default=128)
    parser.add_argument("--num-generation-samples", type=int, default=8)
    parser.add_argument("--num-trajectory-samples", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument(
        "--trajectory-times",
        type=parse_trajectory_times,
        default=DEFAULT_TRAJECTORY_TIMES,
    )
    parser.add_argument("--max-reference-images", type=int, default=None)
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
    parser.add_argument("--skip-mse", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    positive = (
        args.num_train_queries,
        args.num_test_queries,
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
    if args.num_trajectory_samples > args.num_generation_samples:
        parser.error("num-trajectory-samples cannot exceed num-generation-samples.")
    if args.max_reference_images is not None and args.max_reference_images < 2:
        parser.error("max-reference-images must be at least two.")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative.")
    if args.skip_mse and args.skip_generation:
        parser.error("At least one experiment phase must be enabled.")
    return args


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_mask_a_schedule(
    metrics_path: Path,
    layout: WaveletLayout,
) -> tuple[FullRadiusSchedule, dict[str, object]]:
    """Load the earlier Full-calibrated radii used by same-band Mask A."""
    payload = json.loads(metrics_path.read_text())
    selected = payload.get("selected_radii")
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"No selected_radii found in {metrics_path}.")
    subbands = tuple(target.subband for target in build_central_targets(layout))
    rows: list[tuple[float, tuple[int, ...]]] = []
    for time_text, radii in selected.items():
        if not isinstance(radii, dict) or set(radii) != set(subbands):
            raise ValueError("Every Mask A radius row must cover all subbands.")
        rows.append(
            (
                float(time_text),
                tuple(int(radii[subband]) for subband in subbands),
            )
        )
    rows.sort(key=lambda item: item[0])
    schedule = FullRadiusSchedule(
        tuple(time for time, _ in rows),
        subbands,
        tuple(row for _, row in rows),
    )
    return schedule, payload


def build_coarse_group_cache(
    layout: WaveletLayout,
) -> dict[TimeRegime, tuple[WaveletMaskGroup, ...]]:
    return {
        regime: build_coarse_to_fine_rgb_groups(
            layout, regime, channels=NUM_CHANNELS
        )
        for regime in ("high_noise", "intermediate", "low_noise")
    }


def mask_a_groups_at(
    layout: WaveletLayout,
    schedule: FullRadiusSchedule,
    time: float,
    cache: dict[tuple[int, ...], tuple[WaveletMaskGroup, ...]],
) -> tuple[WaveletMaskGroup, ...]:
    radii = schedule.radii_at(time)
    signature = tuple(radii[subband] for subband in schedule.subbands)
    if signature not in cache:
        cache[signature] = build_rgb_radius_group_families(
            layout, radii, channels=NUM_CHANNELS
        )["same_band"]
    return cache[signature]


def query_indices(
    population: int,
    count: int,
    rng: np.random.Generator,
    excluded: set[int] | None = None,
) -> np.ndarray:
    excluded = excluded or set()
    candidates = np.asarray(
        [index for index in range(population) if index not in excluded],
        dtype=np.int64,
    )
    if count > len(candidates):
        raise ValueError("Requested query count exceeds the available split.")
    return rng.choice(candidates, size=count, replace=False)


def previous_test_query_indices(payload: dict[str, object]) -> set[int]:
    protocol = payload.get("protocol", {})
    if not isinstance(protocol, dict):
        return set()
    values: set[int] = set()
    for key in ("calibration_query_indices", "mse_query_indices"):
        indices = protocol.get(key, [])
        if isinstance(indices, list):
            values.update(int(index) for index in indices)
    return values


@torch.no_grad()
def wiener_wavelet_velocity(
    clean_images: np.ndarray,
    noise: np.ndarray,
    time: float,
    transform: WaveletTransform2D,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    operator = build_fm_wiener_operator(U, eigenvalues, Vh, time)
    chunks: list[np.ndarray] = []
    for start in range(0, len(clean_images), batch_size):
        stop = min(start + batch_size, len(clean_images))
        clean = torch.as_tensor(
            clean_images[start:stop], device=device, dtype=torch.float32
        )
        epsilon = torch.as_tensor(noise[start:stop], device=device, dtype=torch.float32)
        x_t = (1.0 - time) * epsilon + time * clean
        velocity = predict_wiener_velocity(x_t, time, operator, mean)
        chunks.append(velocity.cpu().numpy())
    pixels = np.ascontiguousarray(np.concatenate(chunks, axis=0))
    coefficients, _ = analyze_rgb(pixels, transform)
    return flatten_rgb(coefficients)


def metric_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    region_indices: dict[str, np.ndarray],
    *,
    split: str,
    method: str,
    time: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        indices = region_indices[region]
        estimate = prediction[:, indices].astype(np.float64, copy=False)
        truth = target[:, indices].astype(np.float64, copy=False)
        difference = estimate - truth
        numerator = np.sum(estimate * truth, axis=1)
        denominator = np.linalg.norm(estimate, axis=1) * np.linalg.norm(truth, axis=1)
        valid = denominator > np.finfo(np.float64).tiny
        cosine = float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else 0.0
        rows.append(
            {
                "split": split,
                "method": method,
                "region": region,
                "t": time,
                "velocity_mse_vs_fm_unet": float(np.mean(difference**2)),
                "cosine_similarity_vs_fm_unet": cosine,
                "num_queries": len(prediction),
                "num_coefficients_per_query": len(indices),
            }
        )
    return rows


def evaluate_split(
    split: str,
    estimator: MaskedEmpiricalWaveletVelocity,
    model: torch.nn.Module,
    clean_images: np.ndarray,
    noise: np.ndarray,
    times: tuple[float, ...],
    transform: WaveletTransform2D,
    layout: WaveletLayout,
    coarse_schedule: CoarseToFineSchedule,
    coarse_groups: dict[TimeRegime, tuple[WaveletMaskGroup, ...]],
    mask_a_schedule: FullRadiusSchedule,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    device: torch.device,
    model_batch_size: int,
) -> list[dict[str, object]]:
    scalar_regions = build_rgb_region_indices(layout, channels=NUM_CHANNELS)
    region_indices: dict[str, np.ndarray] = {
        "all": np.arange(NUM_CHANNELS * layout.num_coefficients, dtype=np.int64),
        **scalar_regions,
    }
    mask_a_cache: dict[tuple[int, ...], tuple[WaveletMaskGroup, ...]] = {}
    metrics: list[dict[str, object]] = []

    for time in times:
        started = perf_counter()
        print(f"\n=== {split} velocity metrics t={time:.2f} ===", flush=True)
        query_Yt, model_velocity_Y = model_wavelet_state_and_velocity(
            model,
            clean_images,
            noise,
            time,
            transform,
            device,
            model_batch_size,
        )

        estimator.set_groups(coarse_groups[coarse_schedule.regime_at(time)])
        coarse_prediction = estimator.predict_velocity(query_Yt, time)
        estimator.set_groups(
            mask_a_groups_at(layout, mask_a_schedule, time, mask_a_cache)
        )
        mask_a_prediction = estimator.predict_velocity(query_Yt, time)
        wiener_prediction = wiener_wavelet_velocity(
            clean_images,
            noise,
            time,
            transform,
            U,
            eigenvalues,
            Vh,
            mean,
            device,
            model_batch_size,
        )

        for method, prediction in (
            ("coarse_to_fine", coarse_prediction),
            ("mask_a", mask_a_prediction),
            ("wiener_velocity", wiener_prediction),
        ):
            method_rows = metric_rows(
                prediction,
                model_velocity_Y,
                region_indices,
                split=split,
                method=method,
                time=time,
            )
            metrics.extend(method_rows)
            global_row = next(row for row in method_rows if row["region"] == "all")
            print(
                f"  {METHOD_TITLES[method]}: "
                f"MSE={global_row['velocity_mse_vs_fm_unet']:.6e} "
                f"cos={global_row['cosine_similarity_vs_fm_unet']:.4f}",
                flush=True,
            )
        print(f"elapsed={perf_counter() - started:.2f}s", flush=True)
    return metrics


def rows_for(
    metrics: list[dict[str, object]],
    *,
    method: str,
    region: str,
    split: str,
) -> list[dict[str, object]]:
    return sorted(
        (
            row
            for row in metrics
            if row["method"] == method
            and row["region"] == region
            and row["split"] == split
        ),
        key=lambda row: float(row["t"]),
    )


def plot_global_mse(metrics: list[dict[str, object]], out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    for method in ESTIMATORS:
        for split, linestyle in (("test", "-"), ("train", "--")):
            rows = rows_for(metrics, method=method, region="all", split=split)
            axis.plot(
                [float(row["t"]) for row in rows],
                [float(row["velocity_mse_vs_fm_unet"]) for row in rows],
                color=METHOD_COLORS[method],
                marker="o" if split == "test" else None,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{METHOD_TITLES[method]} ({split})",
            )
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1.2)
    axis.set(
        xlabel="Flow time t (0 = noise, 1 = data)",
        ylabel=r"Per-coordinate velocity MSE  $E[(\hat v-v_\theta)^2]$",
        title="Approximation of the pretrained CIFAR-10 FM velocity",
    )
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncols=2)
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def plot_scale_mse(metrics: list[dict[str, object]], out_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True, constrained_layout=True)
    for axis, region in zip(axes, ("coarse", "medium", "fine"), strict=True):
        for method in ESTIMATORS:
            rows = rows_for(metrics, method=method, region=region, split="test")
            axis.plot(
                [float(row["t"]) for row in rows],
                [float(row["velocity_mse_vs_fm_unet"]) for row in rows],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.0,
                label=METHOD_TITLES[method],
            )
        for boundary in (1.0 / 3.0, 2.0 / 3.0):
            axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1.1)
        axis.set_title(REGION_TITLES[region])
        axis.set_xlabel("Flow time t")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel(r"Test velocity MSE  $E[(\hat V_b-V_{\theta,b})^2]$")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Where each estimator differs from the FM U-Net in wavelet scale")
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def plot_cosine(metrics: list[dict[str, object]], out_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    for axis, region in zip(axes.flat, REGIONS, strict=True):
        for method in ESTIMATORS:
            rows = rows_for(metrics, method=method, region=region, split="test")
            axis.plot(
                [float(row["t"]) for row in rows],
                [float(row["cosine_similarity_vs_fm_unet"]) for row in rows],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=1.9,
                label=METHOD_TITLES[method],
            )
        for boundary in (1.0 / 3.0, 2.0 / 3.0):
            axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1.0)
        axis.set_title(REGION_TITLES[region])
        axis.set_xlabel("Flow time t")
        axis.set_ylabel(r"Mean cosine$(\hat V_b,V_{\theta,b})$")
        axis.grid(True, alpha=0.22)
    axes.flat[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Velocity direction agreement on held-out CIFAR-10 queries")
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def plot_generalization_gap(metrics: list[dict[str, object]], out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for method in ESTIMATORS:
        train = rows_for(metrics, method=method, region="all", split="train")
        test = rows_for(metrics, method=method, region="all", split="test")
        train_by_time = {
            float(row["t"]): float(row["velocity_mse_vs_fm_unet"]) for row in train
        }
        times = [float(row["t"]) for row in test]
        gaps = [
            float(row["velocity_mse_vs_fm_unet"]) - train_by_time[float(row["t"])]
            for row in test
        ]
        axis.plot(
            times,
            gaps,
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2.0,
            label=METHOD_TITLES[method],
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    for boundary in (1.0 / 3.0, 2.0 / 3.0):
        axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1.1)
    axis.set(
        xlabel="Flow time t",
        ylabel=r"Generalization gap  $MSE_{test}(t)-MSE_{train}(t)$",
        title="Teacher-velocity generalization gap",
    )
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def wavelet_energy_rows(
    states: dict[str, np.ndarray],
    region_indices: dict[str, np.ndarray],
    time: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, Y_t in states.items():
        for region in ("coarse", "medium", "fine"):
            values = Y_t[:, region_indices[region]].astype(np.float64, copy=False)
            rows.append(
                {
                    "method": method,
                    "region": region,
                    "t": time,
                    "mean_squared_coefficient": float(np.mean(values**2)),
                }
            )
    return rows


@torch.no_grad()
def generate_all_methods(
    estimator: MaskedEmpiricalWaveletVelocity,
    model: torch.nn.Module,
    coarse_schedule: CoarseToFineSchedule,
    coarse_groups: dict[TimeRegime, tuple[WaveletMaskGroup, ...]],
    mask_a_schedule: FullRadiusSchedule,
    transform: WaveletTransform2D,
    layout: WaveletLayout,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    initial_pixels: np.ndarray,
    num_steps: int,
    trajectory_indices: tuple[int, ...],
    num_trajectory_samples: int,
    device: torch.device,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    list[dict[str, object]],
]:
    initial_coefficients, initial_layout = analyze_rgb(initial_pixels, transform)
    if initial_layout != layout:
        raise RuntimeError("Initial noise produced an inconsistent wavelet layout.")
    initial_Y = flatten_rgb(initial_coefficients)
    wavelet_states = {
        "coarse_to_fine": initial_Y.copy(),
        "mask_a": initial_Y.copy(),
    }
    pixel_states = {
        "fm_unet": torch.as_tensor(initial_pixels, device=device, dtype=torch.float32),
        "wiener_velocity": torch.as_tensor(
            initial_pixels, device=device, dtype=torch.float32
        ).clone(),
    }
    snapshots: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHODS}
    mask_a_cache: dict[tuple[int, ...], tuple[WaveletMaskGroup, ...]] = {}
    scalar_regions = build_rgb_region_indices(layout, channels=NUM_CHANNELS)
    region_indices: dict[str, np.ndarray] = {
        "all": np.arange(NUM_CHANNELS * layout.num_coefficients, dtype=np.int64),
        **scalar_regions,
    }

    def pixel_versions() -> dict[str, np.ndarray]:
        return {
            "fm_unet": pixel_states["fm_unet"].cpu().numpy().copy(),
            "coarse_to_fine": synthesize_rgb(
                unflatten_rgb(wavelet_states["coarse_to_fine"], layout, NUM_CHANNELS),
                layout,
                transform,
            ),
            "mask_a": synthesize_rgb(
                unflatten_rgb(wavelet_states["mask_a"], layout, NUM_CHANNELS),
                layout,
                transform,
            ),
            "wiener_velocity": pixel_states["wiener_velocity"].cpu().numpy().copy(),
        }

    def wavelet_versions() -> dict[str, np.ndarray]:
        values = dict(wavelet_states)
        for method in ("fm_unet", "wiener_velocity"):
            pixels = pixel_states[method].cpu().numpy()
            coefficients, current_layout = analyze_rgb(pixels, transform)
            if current_layout != layout:
                raise RuntimeError("Baseline state produced an inconsistent DWT layout.")
            values[method] = flatten_rgb(coefficients)
        return values

    if 0 in trajectory_indices:
        for method, images in pixel_versions().items():
            snapshots[method][0] = images[:num_trajectory_samples].copy()
    energy_metrics = wavelet_energy_rows(wavelet_versions(), region_indices, 0.0)
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        time = step / num_steps
        started = perf_counter()
        estimator.set_groups(coarse_groups[coarse_schedule.regime_at(time)])
        wavelet_states["coarse_to_fine"] += step_size * estimator.predict_velocity(
            wavelet_states["coarse_to_fine"], time
        )
        estimator.set_groups(
            mask_a_groups_at(layout, mask_a_schedule, time, mask_a_cache)
        )
        wavelet_states["mask_a"] += step_size * estimator.predict_velocity(
            wavelet_states["mask_a"], time
        )

        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, time)
        pixel_states["fm_unet"].add_(
            step_size * predict_fm_unet_velocity(model, pixel_states["fm_unet"], time)
        )
        pixel_states["wiener_velocity"].add_(
            step_size
            * predict_wiener_velocity(
                pixel_states["wiener_velocity"], time, wiener_operator, mean
            )
        )
        for method, state in wavelet_states.items():
            if not np.isfinite(state).all():
                raise FloatingPointError(f"Non-finite {method} state at t={time:.4f}.")
        for method, state in pixel_states.items():
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"Non-finite {method} state at t={time:.4f}.")

        state_index = step + 1
        state_time = state_index / num_steps
        energy_metrics.extend(
            wavelet_energy_rows(wavelet_versions(), region_indices, state_time)
        )
        if state_index in trajectory_indices:
            for method, images in pixel_versions().items():
                snapshots[method][state_index] = images[:num_trajectory_samples].copy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"generation {state_index:03d}/{num_steps} t={time:.4f} "
            f"regime={coarse_schedule.regime_at(time)} "
            f"elapsed={perf_counter() - started:.2f}s",
            flush=True,
        )

    generated = pixel_versions()
    final_wavelets = wavelet_versions()
    return generated, final_wavelets, snapshots, energy_metrics


def plot_final_generations(
    initial_pixels: np.ndarray,
    generated: dict[str, np.ndarray],
    neighbors: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_labels: np.ndarray,
    neighbor_mse: np.ndarray,
    num_steps: int,
    out_path: Path,
) -> None:
    columns = ("noise", *METHODS, "nearest")
    sample_count = len(initial_pixels)
    figure, axes = plt.subplots(
        sample_count,
        len(columns),
        figsize=(2.25 * len(columns), 2.05 * sample_count),
        squeeze=False,
    )
    displayed = {
        "noise": display_rgb(initial_pixels),
        **{method: display_rgb(images) for method, images in generated.items()},
        "nearest": display_rgb(neighbors),
    }
    titles = {
        "noise": "Initial noise",
        **METHOD_TITLES,
        "nearest": "Coarse-to-fine nearest train",
    }
    for row in range(sample_count):
        for column, method in enumerate(columns):
            axis = axes[row, column]
            axis.imshow(np.transpose(displayed[method][row], (1, 2, 0)))
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(titles[method], fontsize=9)
        axes[row, 0].set_ylabel(
            f"sample {row}", rotation=0, ha="right", va="center", fontsize=8
        )
        axes[row, -1].set_xlabel(
            f"idx={neighbor_indices[row]}, label={neighbor_labels[row]}\n"
            f"pixel MSE={neighbor_mse[row]:.3e}",
            fontsize=7,
        )
    figure.suptitle(
        f"CIFAR-10: same initial noise and {num_steps}-step explicit Euler",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.04, 0.0, 1.0, 0.985))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(
    snapshots: dict[str, dict[int, np.ndarray]],
    state_indices: tuple[int, ...],
    num_steps: int,
    num_samples: int,
    out_path: Path,
) -> None:
    row_count = len(METHODS) * num_samples
    figure, axes = plt.subplots(
        row_count,
        len(state_indices),
        figsize=(1.85 * len(state_indices), 1.72 * row_count),
        squeeze=False,
    )
    for sample in range(num_samples):
        for method_offset, method in enumerate(METHODS):
            row = sample * len(METHODS) + method_offset
            if set(snapshots[method]) != set(state_indices):
                raise ValueError(f"Incomplete trajectory for {method}.")
            for column, state_index in enumerate(state_indices):
                axis = axes[row, column]
                image = display_rgb(snapshots[method][state_index])[sample]
                axis.imshow(np.transpose(image, (1, 2, 0)))
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"t={state_index / num_steps:.3f}", fontsize=9)
            axes[row, 0].set_ylabel(
                f"sample {sample}\n{METHOD_TITLES[method]}",
                rotation=0,
                ha="right",
                va="center",
                fontsize=7,
            )
    figure.suptitle("Matched generation trajectories around the mask transitions")
    figure.tight_layout(rect=(0.10, 0.0, 1.0, 0.985))
    figure.savefig(out_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_wavelet_energy(
    metrics: list[dict[str, object]],
    out_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True, constrained_layout=True)
    final_energy = {
        (str(row["method"]), str(row["region"])): float(row["mean_squared_coefficient"])
        for row in metrics
        if np.isclose(float(row["t"]), 1.0)
    }
    for column, region in enumerate(("coarse", "medium", "fine")):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in metrics
                    if row["method"] == method and row["region"] == region
                ),
                key=lambda row: float(row["t"]),
            )
            times = [float(row["t"]) for row in rows]
            energies = [float(row["mean_squared_coefficient"]) for row in rows]
            axes[0, column].plot(
                times,
                energies,
                color=METHOD_COLORS[method],
                linewidth=1.9,
                label=METHOD_TITLES[method],
            )
            denominator = max(
                final_energy[(method, region)], np.finfo(np.float64).tiny
            )
            axes[1, column].plot(
                times,
                [value / denominator for value in energies],
                color=METHOD_COLORS[method],
                linewidth=1.9,
            )
        for row in range(2):
            for boundary in (1.0 / 3.0, 2.0 / 3.0):
                axes[row, column].axvline(
                    boundary, color="#777777", linestyle=":", linewidth=1.0
                )
            axes[row, column].grid(True, alpha=0.22)
        axes[0, column].set_title(REGION_TITLES[region])
        axes[1, column].set_xlabel("Flow time t")
    axes[0, 0].set_ylabel(r"Mean energy $E[|Y_b(t)|^2]$")
    axes[1, 0].set_ylabel(r"Relative energy $E_b(t)/E_b(1)$")
    axes[0, -1].legend(frameon=False, fontsize=8)
    figure.suptitle("Coarse-to-fine energy dynamics along matched Euler trajectories")
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def save_rows_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask_size_summary(
    layout: WaveletLayout,
    groups_by_regime: dict[TimeRegime, tuple[WaveletMaskGroup, ...]],
) -> dict[str, dict[str, dict[str, object]]]:
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for regime, groups in groups_by_regime.items():
        by_region: dict[str, list[int]] = {"coarse": [], "medium": [], "fine": []}
        descriptors = (
            layout.subbands[item.subband_index] for item in build_metadata(layout)
        )
        for descriptor, group in zip(descriptors, groups, strict=True):
            by_region[scale_region(descriptor, layout)].append(
                len(group.input_indices) // NUM_CHANNELS
            )
        summaries[regime] = {
            region: {
                "minimum_scalar_sites": min(values),
                "maximum_scalar_sites": max(values),
                "unique_scalar_site_counts": sorted(set(values)),
            }
            for region, values in by_region.items()
        }
    return summaries


def main() -> None:
    args = parse_args()
    args.out_dir = resolve_project_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_project_path(args.checkpoint)
    mask_a_metrics_path = resolve_project_path(args.mask_a_metrics)
    if not checkpoint.exists():
        raise FileNotFoundError(f"FM checkpoint not found: {checkpoint}")
    if not mask_a_metrics_path.exists():
        raise FileNotFoundError(
            "Mask A calibration is required but was not found: "
            f"{mask_a_metrics_path}"
        )
    device = resolve_device(args.device)

    print("dataset=CIFAR-10", flush=True)
    print("normalization=x -> 2x-1", flush=True)
    print("time_direction=t=0 noise, t=1 data", flush=True)
    print("path=x_t=(1-t) epsilon + t x_1", flush=True)
    print("model_output=velocity x_1-epsilon", flush=True)
    print("wavelet=db2 level=3 mode=periodization channelwise RGB", flush=True)
    print("coarse=C=(A3,D3), medium=M=D2, fine=F=D1", flush=True)
    print("time_regimes=[0,1/3), [1/3,2/3), [2/3,1]", flush=True)
    print(f"device={device}", flush=True)

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
    transform = WaveletTransform2D(
        DEFAULT_PROTOCOL.wavelet_name,
        DEFAULT_PROTOCOL.decomposition_level,
        DEFAULT_PROTOCOL.mode,
    )
    reference_Y0, layout = collect_reference_wavelets(train, transform)
    reference_count = len(reference_Y0)
    coarse_schedule = CoarseToFineSchedule()
    coarse_groups = build_coarse_group_cache(layout)
    mask_a_schedule, mask_a_payload = load_mask_a_schedule(
        mask_a_metrics_path, layout
    )
    estimator = MaskedEmpiricalWaveletVelocity(
        family="independent",
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        group_chunk_size=args.group_chunk_size,
        device=device,
        dtype=torch.float32,
        endpoint_epsilon=1e-6,
    ).fit_with_groups(reference_Y0, layout, coarse_groups["high_noise"])
    del reference_Y0

    model = load_fm_unet(checkpoint, device, "cifar10")
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(train, device)
    rng = np.random.default_rng(args.seed)
    train_indices = query_indices(
        len(train.dataset), args.num_train_queries, rng
    )
    excluded_test = previous_test_query_indices(mask_a_payload)
    test_indices = query_indices(
        len(test.dataset), args.num_test_queries, rng, excluded_test
    )
    train_clean = collect_indexed_images(train, train_indices)
    test_clean = collect_indexed_images(test, test_indices)
    shared_count = max(len(train_clean), len(test_clean))
    shared_noise = rng.standard_normal(
        (shared_count, NUM_CHANNELS, *layout.image_shape)
    ).astype(np.float32)
    train_noise = shared_noise[: len(train_clean)].copy()
    test_noise = shared_noise[: len(test_clean)].copy()

    mse_metrics: list[dict[str, object]] = []
    if not args.skip_mse:
        mse_metrics.extend(
            evaluate_split(
                "train",
                estimator,
                model,
                train_clean,
                train_noise,
                args.times,
                transform,
                layout,
                coarse_schedule,
                coarse_groups,
                mask_a_schedule,
                U,
                eigenvalues,
                Vh,
                mean,
                device,
                args.model_batch_size,
            )
        )
        mse_metrics.extend(
            evaluate_split(
                "test",
                estimator,
                model,
                test_clean,
                test_noise,
                args.times,
                transform,
                layout,
                coarse_schedule,
                coarse_groups,
                mask_a_schedule,
                U,
                eigenvalues,
                Vh,
                mean,
                device,
                args.model_batch_size,
            )
        )
        plot_global_mse(mse_metrics, args.out_dir / "velocity_mse_vs_fm_unet.png")
        plot_scale_mse(
            mse_metrics, args.out_dir / "velocity_mse_by_wavelet_scale.png"
        )
        plot_cosine(
            mse_metrics, args.out_dir / "velocity_cosine_by_wavelet_scale.png"
        )
        plot_generalization_gap(
            mse_metrics, args.out_dir / "velocity_generalization_gap.png"
        )
        save_rows_csv(mse_metrics, args.out_dir / "velocity_metrics.csv")

    generation_results: dict[str, object] = {}
    if not args.skip_generation:
        generation_rng = np.random.default_rng(args.seed)
        initial_pixels = generation_rng.standard_normal(
            (args.num_generation_samples, NUM_CHANNELS, *layout.image_shape)
        ).astype(np.float32)
        trajectory_indices = snapshot_indices(
            args.trajectory_times, args.num_steps
        )
        generated, final_wavelets, snapshots, energy_metrics = generate_all_methods(
            estimator,
            model,
            coarse_schedule,
            coarse_groups,
            mask_a_schedule,
            transform,
            layout,
            U,
            eigenvalues,
            Vh,
            mean,
            initial_pixels,
            args.num_steps,
            trajectory_indices,
            args.num_trajectory_samples,
            device,
        )
        neighbor_indices, squared_distances = nearest_reference_indices(
            final_wavelets["coarse_to_fine"],
            estimator.reference_tensor,
            args.neighbor_chunk_size,
        )
        neighbor_images, neighbor_labels = load_neighbor_images(
            train, neighbor_indices
        )
        neighbor_mse = squared_distances / (
            4.0 * NUM_CHANNELS * layout.num_coefficients
        )
        plot_final_generations(
            initial_pixels,
            generated,
            neighbor_images,
            neighbor_indices,
            neighbor_labels,
            neighbor_mse,
            args.num_steps,
            args.out_dir / "final_generations.png",
        )
        plot_trajectories(
            snapshots,
            trajectory_indices,
            args.num_steps,
            args.num_trajectory_samples,
            args.out_dir / "generation_trajectories.png",
        )
        plot_wavelet_energy(
            energy_metrics, args.out_dir / "wavelet_energy_dynamics.png"
        )
        save_rows_csv(energy_metrics, args.out_dir / "wavelet_energy_dynamics.csv")

        sample_arrays = {"initial_noise": initial_pixels}
        sample_arrays.update({f"final_{key}": value for key, value in generated.items()})
        sample_arrays["coarse_to_fine_nearest_train"] = neighbor_images
        np.savez_compressed(args.out_dir / "generated_samples.npz", **sample_arrays)
        trajectory_arrays = {
            f"{method}_step_{state_index:04d}": images
            for method, method_snapshots in snapshots.items()
            for state_index, images in method_snapshots.items()
        }
        np.savez_compressed(
            args.out_dir / "generation_trajectories.npz", **trajectory_arrays
        )
        generation_results = {
            "trajectory_state_indices": list(trajectory_indices),
            "realized_trajectory_times": [
                index / args.num_steps for index in trajectory_indices
            ],
            "coarse_to_fine_neighbors": [
                {
                    "sample": sample,
                    "train_index": int(neighbor_indices[sample]),
                    "label": int(neighbor_labels[sample]),
                    "pixel_mse_0_1_unclipped": float(neighbor_mse[sample]),
                }
                for sample in range(args.num_generation_samples)
            ],
            "wavelet_energy_dynamics": energy_metrics,
        }

    protocol = {
        "dataset": "CIFAR-10",
        "reference_split": "train",
        "train_query_split": "train; members of the empirical reference bank",
        "test_query_split": "test; disjoint from Mask A calibration/evaluation queries",
        "num_reference_images": reference_count,
        "num_train_queries": args.num_train_queries,
        "num_test_queries": args.num_test_queries,
        "normalization": "per-channel x -> (x-0.5)/0.5, range [-1,1]",
        "time_direction": "t=0 Gaussian noise, t=1 clean data",
        "path": "x_t=(1-t)epsilon+t*x_1",
        "fm_model_output": "velocity x_1-epsilon",
        "checkpoint": str(checkpoint),
        "wavelet": layout.wavelet_name,
        "level": layout.decomposition_level,
        "mode": layout.mode,
        "rgb_layout": "independent channel DWT; selected sites expose all RGB channels",
        "coarse_to_fine_regions": {
            "C": ["cA3", "cH3", "cV3", "cD3"],
            "M": ["cH2", "cV2", "cD2"],
            "F": ["cH1", "cV1", "cD1"],
        },
        "coarse_context": "3x3 periodic D3 patch over H/V/D plus one aligned A3 site",
        "medium_context": "3x3 periodic D2 patch over H/V/D",
        "fine_context": "3x3 periodic D1 patch over H/V/D",
        "target_is_always_visible": True,
        "time_regimes": {
            "high_noise": "0 <= t < 1/3",
            "intermediate": "1/3 <= t < 2/3",
            "low_noise": "2/3 <= t <= 1",
        },
        "mask_a": (
            "same-band periodic spatial neighborhood using the Full-calibrated "
            "r_b*(t) from the earlier CIFAR-10 experiment"
        ),
        "mask_a_metrics": str(mask_a_metrics_path),
        "mask_sizes": mask_size_summary(layout, coarse_groups),
        "mse_times": list(args.times),
        "num_steps": args.num_steps,
        "requested_trajectory_times": list(args.trajectory_times),
        "same_initial_noise_across_generation_methods": True,
        "matched_train_test_noise_when_query_counts_match": True,
        "neighbor_metric": "exact unclipped pixel L2 over CIFAR-10 train",
        "device": str(device),
        "dtype": "float32",
        "seed": args.seed,
        "train_query_indices": train_indices.tolist(),
        "test_query_indices": test_indices.tolist(),
    }
    payload = {
        "protocol": protocol,
        "velocity_metrics": mse_metrics,
        "generation": generation_results,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
