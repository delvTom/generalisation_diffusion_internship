"""Generate MNIST samples from the fitted local wavelet posterior means.

The fitted models estimate E[X_1 | X_t] under the standard linear path

    X_t = (1 - t) epsilon + t X_1.

They therefore define a Flow Matching velocity through

    v_hat(X_t, t) = (E[X_1 | X_t] - X_t) / (1 - t).

This script is deliberately a small diagnostic. It reuses the saved GSM
parameters and moments from the denoising experiment; it does not refit them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wavelet-gsm-generation")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_gsm_local.gsm import GSMParameters
from wavelet_gsm_local.image_denoising import gaussian_vector_posterior_tensor
from wavelet_gsm_local.run_image_denoising import (
    denoise_detail_band_batched,
    load_models,
    load_source_config,
    resolve_device,
)
from wavelet_gsm_local.statistics import EnergyStatistics
from wavelet_gsm_local.wavelets import (
    BAND_NAMES,
    HaarLevel2Batch,
    analyze_haar_level2,
    synthesize_haar_level2,
)


METHODS = (
    "wiener_matched_local",
    "wiener_empirical_local",
    "gsm_local",
)
METHOD_LABELS = {
    "wiener_matched_local": "Wiener matched local",
    "wiener_empirical_local": "Wiener empirical local",
    "gsm_local": "GSM local",
}


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from error
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("Trajectory times must lie in [0,1].")
    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("wavelet_gsm_local/results/mnist64_haar2"),
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--num-trajectory-samples", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument(
        "--trajectory-times",
        type=parse_float_list,
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--compute-dtype", choices=("float32", "float64"), default="float32"
    )
    args = parser.parse_args()
    if min(args.num_samples, args.num_trajectory_samples, args.num_steps, args.batch_size) < 1:
        parser.error("Sample counts, steps, and batch size must be positive.")
    if args.num_trajectory_samples > args.num_samples:
        parser.error("num-trajectory-samples cannot exceed num-samples.")
    if args.out_dir is None:
        args.out_dir = args.source_dir / "fm_generation"
    return args


def load_moments(source_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, EnergyStatistics]]:
    path = source_dir / "full_image_denoising" / "input_moments.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_image_denoising.py for this source first."
        )
    with np.load(path) as arrays:
        approximation_mean = arrays["approximation_mean"].copy()
        approximation_covariance = arrays["approximation_covariance"].copy()
        statistics = {}
        for band in BAND_NAMES:
            zeros = np.zeros(9, dtype=np.float64)
            statistics[band] = EnergyStatistics(
                mean=arrays[f"{band}_empirical_mean"].copy(),
                covariance=arrays[f"{band}_empirical_covariance"].copy(),
                signed_correlation=zeros.copy(),
                square_correlation=zeros.copy(),
                fourth_order_excess=zeros.copy(),
            )
    return approximation_mean, approximation_covariance, statistics


def nearest_capture_steps(
    num_steps: int, trajectory_times: tuple[float, ...]
) -> dict[int, float]:
    captures: dict[int, float] = {}
    for time in trajectory_times:
        step = int(round(time * num_steps))
        step = min(max(step, 0), num_steps)
        captures[step] = step / num_steps
    captures[0] = 0.0
    captures[num_steps] = 1.0
    return captures


def reconstruct_clean_means(
    wavelets: dict[str, HaarLevel2Batch],
    time: float,
    methods: tuple[str, ...],
    models: dict[str, GSMParameters],
    statistics: dict[str, EnergyStatistics],
    approximation_mean: torch.Tensor,
    approximation_covariance: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], float]:
    num_samples = len(next(iter(wavelets.values())).approximation)
    approximation_shape = next(iter(wavelets.values())).approximation.shape[1:]
    all_approximations = np.concatenate(
        [wavelets[method].approximation.reshape(num_samples, -1) for method in methods],
        axis=0,
    )
    approximation_result = gaussian_vector_posterior_tensor(
        torch.as_tensor(all_approximations, device=device, dtype=torch.float64),
        time,
        approximation_mean,
        approximation_covariance,
    )
    approximation_predictions = approximation_result.mean.cpu().numpy()

    clean_means: dict[str, np.ndarray] = {}
    for method_index, method in enumerate(methods):
        start = method_index * num_samples
        stop = start + num_samples
        approximation = approximation_predictions[start:stop].reshape(
            num_samples, *approximation_shape
        )
        predicted_bands: dict[str, np.ndarray] = {}
        for band in BAND_NAMES:
            predictions = denoise_detail_band_batched(
                wavelets[method].bands[band],
                time,
                statistics[band],
                models[band],
                device=device,
                dtype=dtype,
                batch_size=batch_size,
            )
            predicted_bands[band] = predictions[method]
        clean_means[method] = synthesize_haar_level2(
            HaarLevel2Batch(approximation=approximation, bands=predicted_bands)
        ).astype(np.float32, copy=False)
    return clean_means, approximation_result.jitter


def plot_final_samples(
    initial_noise: np.ndarray,
    states: dict[str, np.ndarray],
    path: Path,
) -> None:
    columns = ("noise", *METHODS)
    titles = ("Initial noise", *(METHOD_LABELS[method] for method in METHODS))
    figure, axes = plt.subplots(
        len(initial_noise),
        len(columns),
        figsize=(2.0 * len(columns), 1.8 * len(initial_noise)),
        constrained_layout=True,
        squeeze=False,
    )
    for row in range(len(initial_noise)):
        images = (initial_noise[row], *(states[method][row] for method in METHODS))
        for column, (axis, image) in enumerate(zip(axes[row], images, strict=True)):
            axis.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
            if row == 0:
                axis.set_title(titles[column], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Flow Matching generation from local wavelet posterior means\n"
        "Same initial noise across methods; display clipped to [0,1]"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(
    trajectories: dict[str, dict[float, np.ndarray]],
    num_samples: int,
    path: Path,
) -> None:
    times = sorted(next(iter(trajectories.values())))
    rows = len(METHODS) * num_samples
    figure, axes = plt.subplots(
        rows,
        len(times),
        figsize=(1.75 * len(times), 1.7 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    row = 0
    for sample in range(num_samples):
        for method in METHODS:
            for column, time in enumerate(times):
                axis = axes[row, column]
                axis.imshow(
                    np.clip(trajectories[method][time][sample], 0.0, 1.0),
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                if row == 0:
                    axis.set_title(f"t={time:g}")
                if column == 0:
                    axis.set_ylabel(f"sample {sample}\n{METHOD_LABELS[method]}", fontsize=8)
                axis.set_xticks([])
                axis.set_yticks([])
            row += 1
    figure.suptitle("Euler trajectories under the posterior-mean velocity field")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    started = perf_counter()
    source_config = load_source_config(args.source_dir)
    image_size = int(source_config.get("image_size", 28))
    device = resolve_device(args.device)
    dtype = torch.float32 if args.compute_dtype == "float32" else torch.float64
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}; MNIST-{image_size}; samples={args.num_samples}; "
        f"Euler steps={args.num_steps}",
        flush=True,
    )

    models = load_models(args.source_dir)
    approximation_mean_np, approximation_covariance_np, statistics = load_moments(
        args.source_dir
    )
    approximation_mean = torch.as_tensor(
        approximation_mean_np, device=device, dtype=torch.float64
    )
    approximation_covariance = torch.as_tensor(
        approximation_covariance_np, device=device, dtype=torch.float64
    )

    generator = np.random.default_rng(args.seed)
    initial_noise = generator.standard_normal(
        (args.num_samples, image_size, image_size), dtype=np.float32
    )
    states = {method: initial_noise.copy() for method in METHODS}
    captures = nearest_capture_steps(args.num_steps, args.trajectory_times)
    trajectories = {
        method: {0.0: state.copy()} for method, state in states.items()
    }
    diagnostics: list[dict[str, object]] = []
    step_size = 1.0 / args.num_steps

    for step in range(args.num_steps):
        time = step / args.num_steps
        wavelets = {
            method: analyze_haar_level2(np.ascontiguousarray(states[method]))
            for method in METHODS
        }
        clean_means, jitter = reconstruct_clean_means(
            wavelets,
            time,
            METHODS,
            models,
            statistics,
            approximation_mean,
            approximation_covariance,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
        )
        for method in METHODS:
            velocity = (clean_means[method] - states[method]) / (1.0 - time)
            states[method] = np.ascontiguousarray(
                states[method] + step_size * velocity, dtype=np.float32
            )
            diagnostics.append(
                {
                    "step": step,
                    "time": time,
                    "method": method,
                    "velocity_rms": float(np.sqrt(np.mean(velocity.astype(np.float64) ** 2))),
                    "velocity_abs_max": float(np.max(np.abs(velocity))),
                    "state_rms_after_step": float(
                        np.sqrt(np.mean(states[method].astype(np.float64) ** 2))
                    ),
                    "state_abs_max_after_step": float(np.max(np.abs(states[method]))),
                    "a2_jitter": jitter,
                }
            )
            if not np.isfinite(states[method]).all():
                raise FloatingPointError(
                    f"Non-finite state for {method} after Euler step {step}."
                )
        capture_step = step + 1
        if capture_step in captures:
            capture_time = captures[capture_step]
            for method in METHODS:
                trajectories[method][capture_time] = states[method].copy()
        if step == 0 or capture_step in captures or capture_step == args.num_steps:
            print(f"step {capture_step:04d}/{args.num_steps} (t={capture_step / args.num_steps:.4f})", flush=True)

    plot_final_samples(initial_noise, states, args.out_dir / "final_samples.png")
    plot_trajectories(
        trajectories,
        args.num_trajectory_samples,
        args.out_dir / "trajectories.png",
    )
    np.savez_compressed(
        args.out_dir / "samples.npz",
        initial_noise=initial_noise,
        **{f"final_{method}": state for method, state in states.items()},
        **{
            f"trajectory_{method}_t{time:.6f}": images
            for method, by_time in trajectories.items()
            for time, images in by_time.items()
        },
    )
    write_csv(args.out_dir / "numerical_diagnostics.csv", diagnostics)
    config = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "image_size": image_size,
        "methods": list(METHODS),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "device": str(device),
        "compute_dtype": str(dtype),
        "path": "X_t=(1-t)epsilon+tX_1",
        "velocity": "(E[X_1|X_t]-X_t)/(1-t)",
        "solver": "explicit Euler with left-endpoint velocity evaluations",
        "approximation_band": "shared global affine Wiener posterior",
        "detail_bands": "saved same-band 3x3 local posterior; self posterior at borders",
        "normalization": "MNIST pixels in [0,1]; raw states are not clipped",
        "display": "clipped to [0,1]",
        "trajectory_times_actual": sorted(next(iter(trajectories.values()))),
        "elapsed_seconds": perf_counter() - started,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
