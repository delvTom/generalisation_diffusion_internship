"""Generate with fixed/adaptive wavelet masks and compare to empirical FM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_fm_structure.adaptive_radius import (
    RADIUS_FAMILIES,
    FullRadiusSchedule,
    build_radius_group_families,
    fixed_radii,
)
from wavelet_fm_structure.masks import RadiusMaskFamily
from wavelet_fm_structure.radius_study import sample_flow_matching_queries
from wavelet_score.data import load_mnist_32
from wavelet_score.fm_baselines import empirical_optimal_velocity
from wavelet_score.mask_groups import WaveletMaskGroup
from wavelet_score.masked_empirical_fm import MaskedEmpiricalWaveletVelocity
from wavelet_score.transforms import WaveletTransform2D

RadiusStrategy = Literal["fixed_r1", "adaptive_rstar"]
RADIUS_STRATEGIES: tuple[RadiusStrategy, ...] = ("fixed_r1", "adaptive_rstar")
FAMILY_TITLES = {
    "same_band": "Same band",
    "orientation_augmented": "Orientation",
    "scale_augmented": "Scale",
    "full": "Full",
}
FAMILY_COLORS = {
    "same_band": "#0072B2",
    "orientation_augmented": "#D55E00",
    "scale_augmented": "#009E73",
    "full": "#CC79A7",
}
STRATEGY_TITLES = {
    "fixed_r1": "fixed r=1",
    "adaptive_rstar": "shared adaptive r*(t)",
}


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from error
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


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
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-generation-samples", type=int, default=8)
    parser.add_argument("--num-mse-queries", type=int, default=128)
    parser.add_argument(
        "--trajectory-times",
        type=parse_float_list,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--max-reference-images", type=int, default=None)
    parser.add_argument("--reference-chunk-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--group-chunk-size", type=int, default=32)
    parser.add_argument("--candidate-chunk-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="float32 is substantially faster on consumer NVIDIA GPUs.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-mse", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    positive = (
        args.num_steps,
        args.num_generation_samples,
        args.num_mse_queries,
        args.reference_chunk_size,
        args.query_batch_size,
        args.group_chunk_size,
        args.candidate_chunk_size,
    )
    if any(value < 1 for value in positive):
        parser.error("Step, sample, query, and chunk sizes must be positive.")
    if args.max_reference_images is not None and args.max_reference_images < 2:
        parser.error("max-reference-images must be at least two")
    if any(not 0.0 <= time <= 1.0 for time in args.trajectory_times):
        parser.error("trajectory-times must lie in [0,1]")
    if args.skip_generation and args.skip_mse:
        parser.error("Cannot skip both generation and MSE evaluation.")
    return args


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_calibration_config(metrics_path: Path) -> dict[str, object]:
    config_path = metrics_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"The radius calibration config is missing next to {metrics_path}."
        )
    config = json.loads(config_path.read_text())
    required = ("normalization", "wavelet", "level", "mode")
    if any(key not in config for key in required):
        raise ValueError("The radius calibration config is incomplete.")
    return config


def snapshot_indices(
    requested_times: tuple[float, ...], num_steps: int
) -> tuple[int, ...]:
    """Map requested trajectory times to unique nearest Euler state indices."""
    return tuple(sorted({int(round(time * num_steps)) for time in requested_times}))


def variant_key(family: RadiusMaskFamily, strategy: RadiusStrategy) -> str:
    return f"{family}__{strategy}"


def display_images(
    normalized_images: np.ndarray,
    global_mean: float,
    global_std: float,
) -> np.ndarray:
    return np.clip(normalized_images * global_std + global_mean, 0.0, 1.0)


def plot_final_generations(
    generated: dict[tuple[RadiusMaskFamily, RadiusStrategy], np.ndarray],
    strategy: RadiusStrategy,
    out_path: Path,
) -> None:
    sample_count = len(generated[(RADIUS_FAMILIES[0], strategy)])
    figure, axes = plt.subplots(
        sample_count,
        len(RADIUS_FAMILIES),
        figsize=(2.25 * len(RADIUS_FAMILIES), 2.05 * sample_count),
        squeeze=False,
    )
    for row_index in range(sample_count):
        for column_index, family in enumerate(RADIUS_FAMILIES):
            axis = axes[row_index, column_index]
            axis.imshow(generated[(family, strategy)][row_index], cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(FAMILY_TITLES[family])
            if column_index == 0:
                axis.set_ylabel(f"sample {row_index}")
    figure.suptitle(
        f"MNIST-32 masked empirical FM generation: {STRATEGY_TITLES[strategy]}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(
    trajectories: dict[
        tuple[RadiusMaskFamily, RadiusStrategy], dict[int, np.ndarray]
    ],
    indices: tuple[int, ...],
    num_steps: int,
    out_path: Path,
) -> None:
    variants = tuple(
        (family, strategy)
        for strategy in RADIUS_STRATEGIES
        for family in RADIUS_FAMILIES
    )
    figure, axes = plt.subplots(
        len(variants),
        len(indices),
        figsize=(2.05 * len(indices), 1.85 * len(variants)),
        squeeze=False,
    )
    for row_index, (family, strategy) in enumerate(variants):
        for column_index, state_index in enumerate(indices):
            axis = axes[row_index, column_index]
            axis.imshow(
                trajectories[(family, strategy)][state_index][0],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"t={state_index / num_steps:.2f}")
            if column_index == 0:
                axis.set_ylabel(
                    f"{FAMILY_TITLES[family]}\n{STRATEGY_TITLES[strategy]}",
                    rotation=0,
                    ha="right",
                    va="center",
                )
    figure.suptitle(
        "Euler trajectories for sample 0 (same initial Gaussian noise)"
    )
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.98))
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mse_to_empirical(
    rows: list[dict[str, object]],
    out_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True, sharey=True)
    plot_floor = 1e-30
    for axis, strategy in zip(axes, RADIUS_STRATEGIES, strict=True):
        for family in RADIUS_FAMILIES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["family"] == family
                    and row["radius_strategy"] == strategy
                ),
                key=lambda row: float(row["t"]),
            )
            axis.plot(
                [float(row["t"]) for row in selected],
                [max(float(row["velocity_mse_wavelet"]), plot_floor) for row in selected],
                marker="o",
                linewidth=2,
                color=FAMILY_COLORS[family],
                label=FAMILY_TITLES[family],
            )
        axis.set_title(STRATEGY_TITLES[strategy])
        axis.set_xlabel("Flow time t (noise to data)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel(
        r"Per-coordinate MSE to empirical optimum "
        r"$\mathbb{E}\|\widehat V-V^*_{emp}\|_2^2/d$"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False)
    figure.suptitle("Masked wavelet FM velocity versus unconstrained empirical optimum")
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def empirical_velocity_in_batches(
    queries: np.ndarray,
    candidates: torch.Tensor,
    time: float,
    query_batch_size: int,
    candidate_chunk_size: int,
) -> np.ndarray:
    result = np.empty_like(queries)
    for start in range(0, len(queries), query_batch_size):
        stop = min(start + query_batch_size, len(queries))
        query = torch.as_tensor(
            queries[start:stop], device=candidates.device, dtype=candidates.dtype
        )
        prediction = empirical_optimal_velocity(
            query,
            candidates,
            time,
            candidate_chunk_size=candidate_chunk_size,
        )
        result[start:stop] = prediction.cpu().numpy()
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    numpy_dtype = np.float32 if args.dtype == "float32" else np.float64
    parseval_tolerance = 2e-5 if args.dtype == "float32" else 1e-10
    calibration = load_calibration_config(args.radius_metrics)
    normalization = calibration["normalization"]
    if not isinstance(normalization, dict):
        raise TypeError("Calibration normalization metadata must be a dictionary.")
    global_mean = float(normalization["global_mean"])
    global_std = float(normalization["global_std"])
    if global_std <= 0.0 or not np.isfinite([global_mean, global_std]).all():
        raise ValueError("Calibration normalization statistics are invalid.")

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
    fixed_radius_map = fixed_radii(layout, 1)

    group_cache: dict[
        tuple[int, ...],
        dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]],
    ] = {}

    def radii_for(strategy: RadiusStrategy, time: float) -> dict[str, int]:
        return fixed_radius_map if strategy == "fixed_r1" else schedule.radii_at(time)

    def groups_for(
        family: RadiusMaskFamily,
        strategy: RadiusStrategy,
        time: float,
    ) -> tuple[WaveletMaskGroup, ...]:
        radii = radii_for(strategy, time)
        signature = tuple(radii[subband] for subband in schedule.subbands)
        if signature not in group_cache:
            group_cache[signature] = build_radius_group_families(layout, radii)
        return group_cache[signature][family]

    initial_groups = groups_for("same_band", "fixed_r1", 0.0)
    estimator = MaskedEmpiricalWaveletVelocity(
        family="independent",
        reference_chunk_size=args.reference_chunk_size,
        query_batch_size=args.query_batch_size,
        group_chunk_size=args.group_chunk_size,
        device=device,
        dtype=torch_dtype,
        endpoint_epsilon=1e-8,
    ).fit_with_groups(reference_Y0, layout, initial_groups)

    config = {
        "dataset": "MNIST training empirical distribution, resized to 32x32",
        "num_reference_images": len(reference_Y0),
        "normalization": {"global_mean": global_mean, "global_std": global_std},
        "wavelet": layout.wavelet_name,
        "level": layout.decomposition_level,
        "mode": layout.mode,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "path": "Y_t=(1-t)xi+tY_0",
        "target": "V=Y_0-xi",
        "families": list(RADIUS_FAMILIES),
        "radius_strategies": {
            "fixed_r1": "r=1 in every subband",
            "adaptive_rstar": (
                "same full-family selected radius applied to every family; "
                "nearest calibrated time"
            ),
        },
        "radius_metrics": str(args.radius_metrics),
        "calibration_times": list(schedule.calibration_times),
        "adaptive_radii": {
            f"{time:.8g}": dict(zip(schedule.subbands, row, strict=True))
            for time, row in zip(
                schedule.calibration_times, schedule.radii, strict=True
            )
        },
        "num_steps": args.num_steps,
        "euler_rule": "explicit left Euler, t_n=n/num_steps",
        "num_generation_samples": args.num_generation_samples,
        "trajectory_times_requested": list(args.trajectory_times),
        "num_mse_queries": args.num_mse_queries,
        "mse_clean_queries_sampled_from_reference_bank": True,
        "mse_target": "unconstrained empirical conditional FM velocity",
        "parseval_tolerance": parseval_tolerance,
        "reference_chunk_size": args.reference_chunk_size,
        "query_batch_size": args.query_batch_size,
        "group_chunk_size": args.group_chunk_size,
        "candidate_chunk_size": args.candidate_chunk_size,
        "clipping_during_sampling": False,
        "display_clipping": "after inverse normalization, [0,1]",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"device={device} dtype={args.dtype} references={len(reference_Y0)} "
        f"steps={args.num_steps}",
        flush=True,
    )

    timing_rows: list[dict[str, object]] = []
    if not args.skip_generation:
        rng = np.random.default_rng(args.seed)
        initial_Y = rng.standard_normal(
            (args.num_generation_samples, layout.num_coefficients)
        ).astype(numpy_dtype)
        variants = tuple(
            (family, strategy)
            for strategy in RADIUS_STRATEGIES
            for family in RADIUS_FAMILIES
        )
        states = {variant: initial_Y.copy() for variant in variants}
        requested_indices = snapshot_indices(args.trajectory_times, args.num_steps)
        trajectory_Y = {
            variant: ({0: initial_Y.copy()} if 0 in requested_indices else {})
            for variant in variants
        }
        step_size = 1.0 / args.num_steps
        for step in range(args.num_steps):
            time = step / args.num_steps
            step_start = perf_counter()
            for family, strategy in variants:
                estimator.set_groups(groups_for(family, strategy, time))
                velocity = estimator.predict_velocity(states[(family, strategy)], time)
                states[(family, strategy)] += step_size * velocity
                if not np.isfinite(states[(family, strategy)]).all():
                    raise FloatingPointError(
                        f"Non-finite Euler state for {family}/{strategy} at t={time}."
                    )
            state_index = step + 1
            if state_index in requested_indices:
                for variant in variants:
                    trajectory_Y[variant][state_index] = states[variant].copy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - step_start
            timing_rows.append(
                {"phase": "generation", "step": step, "t": time, "seconds": elapsed}
            )
            print(
                f"generation step {state_index:03d}/{args.num_steps} "
                f"t={time:.4f} elapsed={elapsed:.2f}s",
                flush=True,
            )

        generated_normalized = {
            variant: transform.synthesize(state, layout)
            for variant, state in states.items()
        }
        generated_display = {
            variant: display_images(images, global_mean, global_std)
            for variant, images in generated_normalized.items()
        }
        trajectory_display = {
            variant: {
                index: display_images(
                    transform.synthesize(state, layout), global_mean, global_std
                )
                for index, state in snapshots.items()
            }
            for variant, snapshots in trajectory_Y.items()
        }
        for strategy in RADIUS_STRATEGIES:
            plot_final_generations(
                generated_display,
                strategy,
                args.out_dir / f"generation_{strategy}.png",
            )
        plot_trajectories(
            trajectory_display,
            requested_indices,
            args.num_steps,
            args.out_dir / "generation_trajectories.png",
        )
        np.savez_compressed(
            args.out_dir / "generated_samples.npz",
            initial_wavelet_noise=initial_Y,
            **{
                variant_key(*variant): images
                for variant, images in generated_normalized.items()
            },
        )
        np.savez_compressed(
            args.out_dir / "generation_trajectories.npz",
            trajectory_state_indices=np.asarray(requested_indices, dtype=np.int64),
            **{
                variant_key(*variant): np.stack(
                    [trajectory_Y[variant][index] for index in requested_indices]
                )
                for variant in variants
            },
        )

    mse_rows: list[dict[str, object]] = []
    if not args.skip_mse:
        queries = sample_flow_matching_queries(
            reference_Y0,
            args.num_mse_queries,
            args.seed + 1,
        )
        for time in schedule.calibration_times:
            time_start = perf_counter()
            query_Yt = (1.0 - time) * queries.noise + time * queries.clean_Y0
            empirical_velocity = empirical_velocity_in_batches(
                query_Yt,
                estimator.reference_tensor,
                time,
                args.query_batch_size,
                args.candidate_chunk_size,
            )
            for strategy in RADIUS_STRATEGIES:
                radii = radii_for(strategy, time)
                calibration_time = (
                    time
                    if strategy == "fixed_r1"
                    else schedule.nearest_calibration_time(time)
                )
                for family in RADIUS_FAMILIES:
                    groups = groups_for(family, strategy, time)
                    estimator.set_groups(groups)
                    prediction = estimator.predict_velocity(query_Yt, time)
                    difference = prediction - empirical_velocity
                    wavelet_mse = float(np.mean(difference**2))
                    pixel_difference = transform.synthesize(difference, layout)
                    pixel_mse = float(np.mean(pixel_difference**2))
                    parseval_error = abs(wavelet_mse - pixel_mse)
                    if parseval_error > parseval_tolerance:
                        raise FloatingPointError(
                            f"Wavelet/pixel MSE mismatch: {parseval_error:.3e}."
                        )
                    sizes = np.asarray(
                        [len(group.input_indices) for group in groups],
                        dtype=np.float64,
                    )
                    mse_rows.append(
                        {
                            "t": time,
                            "family": family,
                            "radius_strategy": strategy,
                            "radius_calibration_time": calibration_time,
                            "radii_by_subband": json.dumps(radii, sort_keys=True),
                            "mean_mask_size": float(np.mean(sizes)),
                            "max_mask_size": int(np.max(sizes)),
                            "velocity_mse_wavelet": wavelet_mse,
                            "velocity_mse_pixel": pixel_mse,
                            "clean_mse_equivalent": (1.0 - time) ** 2 * wavelet_mse,
                            "parseval_abs_error": parseval_error,
                            "num_queries": args.num_mse_queries,
                            "num_candidates": len(reference_Y0),
                        }
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - time_start
            timing_rows.append(
                {"phase": "mse", "step": "", "t": time, "seconds": elapsed}
            )
            write_rows(mse_rows, args.out_dir / "mse_vs_empirical_optimum.csv")
            (args.out_dir / "mse_vs_empirical_optimum.json").write_text(
                json.dumps(mse_rows, indent=2) + "\n"
            )
            print(f"MSE t={time:.2f} elapsed={elapsed:.2f}s", flush=True)
        plot_mse_to_empirical(
            mse_rows, args.out_dir / "mse_vs_empirical_optimum.png"
        )

    write_rows(timing_rows, args.out_dir / "timings.csv")
    (args.out_dir / "timings.json").write_text(json.dumps(timing_rows, indent=2) + "\n")
    print(f"saved adaptive-radius experiment to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
