"""Generate MNIST-32 samples with the three degree-three wavelet regressors."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wavelet-fm-polynomial")

import matplotlib.pyplot as plt
import numpy as np
import pywt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wavelet_fm_polynomial.features import (  # noqa: E402
    FeatureFamily,
    build_feature_structure,
    feature_count,
)
from wavelet_fm_polynomial.regression import fit_nested_degrees  # noqa: E402
from wavelet_score.data import load_mnist_32  # noqa: E402
from wavelet_score.indexing import WaveletLayout  # noqa: E402
from wavelet_score.transforms import WaveletTransform2D  # noqa: E402

FAMILIES: tuple[FeatureFamily, ...] = ("independent", "band", "local")
DEFAULT_TRAJECTORY_TIMES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)


def parse_family_tuple(text: str) -> tuple[FeatureFamily, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values or any(value not in FAMILIES for value in values):
        raise argparse.ArgumentTypeError(
            "Families must be selected from independent, band, and local."
        )
    return tuple(dict.fromkeys(values))  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_fm_polynomial/results/mnist32_generation_D3"),
    )
    parser.add_argument("--num-fit", type=int, default=60_000)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-euler-steps", type=int, default=20)
    parser.add_argument("--families", type=parse_family_tuple, default=FAMILIES)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--mode", default="periodization")
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--moment-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=12_345)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if min(
        args.num_fit,
        args.num_samples,
        args.num_euler_steps,
        args.level,
        args.batch_size,
    ) < 1:
        parser.error("Counts, wavelet level, and batch size must be positive.")
    if args.local_radius < 1 or args.ridge < 0.0:
        parser.error("local-radius must be positive and ridge must be non-negative.")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def prepare_fit_coefficients(
    images: np.ndarray,
    transform: WaveletTransform2D,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, WaveletLayout]:
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(images.shape, dtype=np.float32)
    clean_coefficients, layout = transform.analyze(images)
    noise_coefficients, noise_layout = transform.analyze(noise)
    if noise_layout != layout:
        raise RuntimeError("Clean and noise wavelet layouts differ.")
    return clean_coefficients, noise_coefficients, layout


def synthesize_tensor(
    coefficients: torch.Tensor,
    transform: WaveletTransform2D,
    layout: WaveletLayout,
) -> np.ndarray:
    values = coefficients.detach().cpu().to(torch.float32).numpy()
    return transform.synthesize(values, layout)


def plot_generation_grid(
    images: dict[FeatureFamily, np.ndarray],
    families: tuple[FeatureFamily, ...],
    path: Path,
) -> None:
    num_samples = len(next(iter(images.values())))
    figure, axes = plt.subplots(
        num_samples,
        len(families),
        figsize=(2.25 * len(families), 2.05 * num_samples),
        squeeze=False,
        constrained_layout=True,
    )
    for column, family in enumerate(families):
        axes[0, column].set_title(f"{family.capitalize()}, D=3")
        for row in range(num_samples):
            axes[row, column].imshow(
                np.clip(images[family][row], 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("MNIST-32 generation from identical initial noises (display clipped to [0,1])")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_trajectory(
    trajectories: dict[FeatureFamily, np.ndarray],
    times: np.ndarray,
    families: tuple[FeatureFamily, ...],
    path: Path,
    sample_index: int = 0,
) -> None:
    selected_indices = sorted(
        {int(np.argmin(np.abs(times - requested))) for requested in DEFAULT_TRAJECTORY_TIMES}
    )
    figure, axes = plt.subplots(
        len(families),
        len(selected_indices),
        figsize=(2.0 * len(selected_indices), 2.15 * len(families)),
        squeeze=False,
        constrained_layout=True,
    )
    for column, index in enumerate(selected_indices):
        axes[0, column].set_title(f"t={times[index]:.2f}")
    for row, family in enumerate(families):
        axes[row, 0].set_ylabel(f"{family.capitalize()}\nD=3")
        for column, index in enumerate(selected_indices):
            axes[row, column].imshow(
                np.clip(trajectories[family][index, sample_index], 0.0, 1.0),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(
        "Euler trajectory from the same initial noise\n"
        "Each D=3 estimator is fitted at the Euler evaluation time"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    moment_dtype = torch.float32 if args.moment_dtype == "float32" else torch.float64
    solve_dtype = torch.float64
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if not pywt.Wavelet(args.wavelet).orthogonal:
        raise ValueError("This experiment requires an orthogonal wavelet basis.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "families": list(args.families),
        "device_resolved": str(device),
        "degree": 3,
        "path": "X_t = t X + (1-t) epsilon",
        "target": "V = X - epsilon",
        "temporal_model": "direct D=3 fit at every nonzero Euler evaluation time",
        "euler_evaluation_times": [
            step / args.num_euler_steps for step in range(args.num_euler_steps)
        ],
        "display_clipping_only": True,
    }
    with (args.out_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(f"device: {device}")
    print(f"families: {', '.join(args.families)}")
    print(f"Euler steps: {args.num_euler_steps}")
    print(f"nonzero temporal fits per family: {args.num_euler_steps - 1}")
    train = load_mnist_32(
        args.data_root,
        "train",
        batch_size=args.batch_size,
        max_samples=args.num_fit,
        download=args.download,
    )
    transform = WaveletTransform2D(args.wavelet, args.level, args.mode)
    clean, fit_noise, layout = prepare_fit_coefficients(
        train.images, transform, args.fit_seed
    )
    del train
    gc.collect()
    structure = build_feature_structure(layout, local_radius=args.local_radius)
    targets = np.ascontiguousarray(clean - fit_noise)
    for family in args.families:
        count = feature_count(structure, family, 3)
        print(f"{family} D=3 features: {count} (intercept included)")
        if len(clean) <= count - 1:
            raise ValueError(
                f"{family} D=3 needs more than {count - 1} fit samples; got {len(clean)}."
            )

    diagnostics: dict[str, object] = {"fits": {}, "sampling": {}}
    start_time = perf_counter()
    generation_rng = np.random.default_rng(args.generation_seed)
    initial_noise = generation_rng.standard_normal(
        (args.num_samples, *layout.image_shape), dtype=np.float32
    )
    initial_Y, initial_layout = transform.analyze(initial_noise)
    if initial_layout != layout:
        raise RuntimeError("Generation noise has an incompatible wavelet layout.")
    initial_state = torch.as_tensor(initial_Y, device=device, dtype=solve_dtype)
    mean = torch.as_tensor(clean.mean(axis=0), device=device, dtype=solve_dtype)
    states = {family: initial_state.clone() for family in args.families}
    coefficient_trajectories = {
        family: [initial_state.detach().cpu().clone()] for family in args.families
    }
    euler_times = np.linspace(0.0, 1.0, args.num_euler_steps + 1, dtype=np.float64)
    step_size = 1.0 / args.num_euler_steps

    print("\n=== Euler generation ===")
    for step, time in enumerate(euler_times[:-1]):
        print(
            f"\n=== Euler step {step + 1:02d}/{args.num_euler_steps}, "
            f"t={time:.4f} ==="
        )
        if step == 0:
            # At t=0, X_t is pure noise and E[V | X_t] = mu - X_t.
            for family in args.families:
                velocity = mean - states[family]
                states[family] = states[family] + step_size * velocity
        else:
            Y_t_fit = np.ascontiguousarray(time * clean + (1.0 - time) * fit_noise)
            for family in args.families:
                family_start = perf_counter()
                print(f"  fitting and evaluating {family}, D=3...")
                model = fit_nested_degrees(
                    Y_t_fit,
                    targets,
                    structure=structure,
                    family=family,
                    degrees=(3,),
                    device=device,
                    batch_size=args.batch_size,
                    moment_dtype=moment_dtype,
                    solve_dtype=solve_dtype,
                    ridge=args.ridge,
                )[3]
                velocity = model.predict_tensor(states[family])
                states[family] = states[family] + step_size * velocity
                fit_section = diagnostics["fits"]
                if not isinstance(fit_section, dict):
                    raise RuntimeError("Malformed diagnostics container.")
                fit_section[f"step={step:02d}/t={time:g}/{family}/D=3"] = {
                    **model.diagnostics,
                    "elapsed_seconds": perf_counter() - family_start,
                }
                print(
                    "    residual="
                    f"{model.diagnostics['relative_normal_equation_residual']:.3e}, "
                    f"elapsed={perf_counter() - family_start:.1f}s"
                )
                del model, velocity
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del Y_t_fit

        for family in args.families:
            if not torch.isfinite(states[family]).all():
                raise RuntimeError(
                    f"Non-finite generation state for {family} after Euler step {step + 1}."
                )
            coefficient_trajectories[family].append(states[family].detach().cpu().clone())
        diagnostics["elapsed_seconds"] = perf_counter() - start_time
        with (args.out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, indent=2)

    final_images = {
        family: synthesize_tensor(states[family], transform, layout)
        for family in args.families
    }
    trajectory_images = {
        family: transform.synthesize(
            torch.stack(coefficient_trajectories[family])
            .reshape(-1, layout.num_coefficients)
            .to(torch.float32)
            .numpy(),
            layout,
        ).reshape(args.num_euler_steps + 1, args.num_samples, *layout.image_shape)
        for family in args.families
    }

    plot_generation_grid(
        final_images, args.families, args.out_dir / "generation_grid_D3.png"
    )
    plot_trajectory(
        trajectory_images,
        euler_times,
        args.families,
        args.out_dir / "trajectory_D3.png",
    )
    arrays: dict[str, np.ndarray] = {
        "initial_noise": initial_noise,
        "euler_times": euler_times,
    }
    for family in args.families:
        arrays[f"final_{family}"] = final_images[family]
        arrays[f"trajectory_{family}"] = trajectory_images[family]
    np.savez_compressed(args.out_dir / "generation_D3.npz", **arrays)

    sampling_section = diagnostics["sampling"]
    if not isinstance(sampling_section, dict):
        raise RuntimeError("Malformed sampling diagnostics container.")
    sampling_section.update(
        {
            "elapsed_total_seconds": perf_counter() - start_time,
            "num_euler_steps": args.num_euler_steps,
            "trajectory_times": euler_times.tolist(),
            "final_raw_ranges": {
                family: [float(images.min()), float(images.max())]
                for family, images in final_images.items()
            },
        }
    )
    with (args.out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(f"\nresults: {args.out_dir}")


if __name__ == "__main__":
    main()
