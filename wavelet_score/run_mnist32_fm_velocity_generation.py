"""Train structured wavelet FM velocities at Euler nodes and generate MNIST-32."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_score.data import load_mnist_32
from wavelet_score.fm_baselines import (
    WaveletWienerVelocity,
    empirical_optimal_velocity,
)
from wavelet_score.run_mnist32_denoising import (
    MODEL_NAMES,
    make_model,
    parse_number_list,
    resolve_device,
)
from wavelet_score.transforms import WaveletTransform2D
from wavelet_score.torch_regressors import torch_dtype

DISPLAY_NAMES = {
    "independent": "Independent",
    "band_coupled": "Band-coupled",
    "local_coupled": "Local",
    "wiener": "Wiener FM (best affine)",
}
COLORS = {
    "independent": "#2878b5",
    "band_coupled": "#f28e2b",
    "local_coupled": "#3a923a",
    "wiener": "#b33c86",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_score/results/mnist32_fm_velocity_generation"),
    )
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--degrees", type=str, default="1,2,3")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-eval-samples", type=int, default=1024)
    parser.add_argument("--num-empirical-samples", type=int, default=16)
    parser.add_argument("--num-empirical-candidates", type=int, default=None)
    parser.add_argument("--candidate-chunk-size", type=int, default=512)
    parser.add_argument("--num-generation-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.degrees = parse_number_list(args.degrees, int)
    positive = (
        args.num_steps,
        args.batch_size,
        args.num_eval_samples,
        args.num_empirical_samples,
        args.candidate_chunk_size,
        args.num_generation_samples,
    )
    if any(value < 1 for value in positive):
        parser.error("Step, batch, evaluation, candidate, and generation counts must be positive.")
    if any(degree < 1 for degree in args.degrees):
        parser.error("Every polynomial degree must be positive.")
    if args.ridge < 0.0 or args.local_radius < 1:
        parser.error("ridge must be non-negative and local-radius positive.")
    return args


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction.astype(np.float64, copy=False) - target.astype(
        np.float64, copy=False
    )
    return float(np.mean(np.square(difference)))


def fit_velocity_model(
    family: str,
    degree: int,
    Y_t: np.ndarray,
    velocity_target: np.ndarray,
    layout,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
):
    model = make_model(
        family,
        degree,
        args.ridge,
        args.local_radius,
        "torch",
        device,
        dtype,
    )
    if family == "independent":
        return model.fit(Y_t, velocity_target, batch_size=args.batch_size)
    return model.fit(Y_t, velocity_target, layout, batch_size=args.batch_size)


def save_metrics(rows: list[dict[str, object]], out_dir: Path) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_mse_panels(
    rows: list[dict[str, object]],
    degrees: tuple[int, ...],
    metric: str,
    ylabel: str,
    out_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1, len(degrees), figsize=(5.1 * len(degrees), 4.4), squeeze=False, sharey=True
    )
    for axis, degree in zip(axes[0], degrees, strict=True):
        for family in (*MODEL_NAMES, "wiener"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["model"] == family
                    and (family == "wiener" or int(row["degree"]) == degree)
                ),
                key=lambda row: float(row["t"]),
            )
            if not selected:
                continue
            axis.plot(
                [float(row["t"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o" if family == "wiener" else None,
                markersize=3,
                linewidth=2,
                color=COLORS[family],
                label=DISPLAY_NAMES[family],
            )
        axis.set_title(f"Polynomial degree D={degree}")
        axis.set_xlabel("Flow time t")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("MNIST-32 Flow Matching velocity regression", fontsize=14)
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_time_averaged_mse(
    rows: list[dict[str, object]], degrees: tuple[int, ...], out_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, metric, title in zip(
        axes,
        ("target_mse", "empirical_optimum_mse"),
        ("Supervised FM target", "Unconstrained empirical optimum"),
        strict=True,
    ):
        for family in MODEL_NAMES:
            values = []
            for degree in degrees:
                selected = [
                    float(row[metric])
                    for row in rows
                    if row["model"] == family and int(row["degree"]) == degree
                ]
                values.append(float(np.mean(selected)))
            axis.plot(
                degrees,
                values,
                marker="o",
                linewidth=2,
                color=COLORS[family],
                label=DISPLAY_NAMES[family],
            )
        wiener_values = [
            float(row[metric]) for row in rows if row["model"] == "wiener"
        ]
        if wiener_values:
            axis.axhline(
                np.mean(wiener_values),
                color=COLORS["wiener"],
                linestyle="--",
                linewidth=2,
                label=DISPLAY_NAMES["wiener"],
            )
        axis.set_title(title)
        axis.set_xlabel("Polynomial degree D")
        axis.set_ylabel("Time-averaged per-coordinate MSE")
        axis.set_xticks(degrees)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_generations(
    generated: dict[tuple[str, int | None], np.ndarray],
    degrees: tuple[int, ...],
    out_dir: Path,
) -> None:
    wiener = generated[("wiener", None)]
    for degree in degrees:
        columns = [("wiener", None)] + [(family, degree) for family in MODEL_NAMES]
        figure, axes = plt.subplots(
            len(wiener), len(columns), figsize=(2.2 * len(columns), 2.1 * len(wiener)), squeeze=False
        )
        for row_index in range(len(wiener)):
            for column_index, key in enumerate(columns):
                axis = axes[row_index, column_index]
                axis.imshow(np.clip(generated[key][row_index], 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(DISPLAY_NAMES[key[0]], fontsize=11)
        figure.suptitle(
            f"Euler generation from identical noise - degree D={degree}", fontsize=14
        )
        figure.tight_layout()
        figure.savefig(out_dir / f"generation_degree_{degree}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    compute_dtype = torch_dtype(args.compute_dtype)
    storage_dtype = np.float64 if args.compute_dtype == "float64" else np.float32

    train = load_mnist_32(
        args.data_root,
        "train",
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        download=args.download,
    )
    test = load_mnist_32(
        args.data_root,
        "test",
        batch_size=args.batch_size,
        max_samples=args.num_eval_samples,
        download=args.download,
    )
    transform = WaveletTransform2D(args.wavelet, args.level, mode="periodization")
    Y_0_train, layout = transform.analyze(train.images.astype(storage_dtype, copy=False))
    Y_0_eval, eval_layout = transform.analyze(test.images.astype(storage_dtype, copy=False))
    if eval_layout != layout:
        raise RuntimeError("Train and evaluation wavelet layouts differ.")

    rng = np.random.default_rng(args.seed)
    xi_train = rng.standard_normal(Y_0_train.shape).astype(storage_dtype)
    xi_eval = rng.standard_normal(Y_0_eval.shape).astype(storage_dtype)
    velocity_train = Y_0_train - xi_train
    velocity_eval = Y_0_eval - xi_eval
    initial_images = rng.standard_normal(
        (args.num_generation_samples, *layout.image_shape)
    ).astype(storage_dtype)
    initial_Y, initial_layout = transform.analyze(initial_images)
    if initial_layout != layout:
        raise RuntimeError("Generation and training layouts differ.")

    candidates_count = args.num_empirical_candidates or len(Y_0_train)
    candidates_count = min(candidates_count, len(Y_0_train))
    empirical_count = min(args.num_empirical_samples, len(Y_0_eval))
    candidate_tensor = torch.as_tensor(
        Y_0_train[:candidates_count], device=device, dtype=compute_dtype
    )
    wiener = WaveletWienerVelocity.fit(Y_0_train, device, compute_dtype)

    states: dict[tuple[str, int | None], np.ndarray] = {
        (family, degree): initial_Y.copy()
        for degree in args.degrees
        for family in MODEL_NAMES
    }
    wiener_state = torch.as_tensor(initial_Y, device=device, dtype=compute_dtype)
    step_size = 1.0 / args.num_steps
    rows: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []

    config = {
        **{
            key: value
            for key, value in vars(args).items()
            if key not in ("data_root", "out_dir")
        },
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "degrees": list(args.degrees),
        "num_train_samples": len(Y_0_train),
        "num_evaluation_samples": len(Y_0_eval),
        "num_empirical_candidates_used": candidates_count,
        "euler_rule": "explicit left Euler: t_n=n/K, n=0,...,K-1",
        "path": "Y_t=(1-t)xi+tY_0",
        "target": "V_Y=Y_0-xi",
        "wiener_covariance_normalization": "1/(N-1), matching fm_comparisons",
        "clipping_during_sampling": False,
        "display_clipping": "[0,1]",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(
        f"device={device} train={len(Y_0_train)} eval={len(Y_0_eval)} "
        f"wavelet={args.wavelet} steps={args.num_steps}",
        flush=True,
    )
    for step in range(args.num_steps):
        t = step / args.num_steps
        start_time = perf_counter()
        Y_t_train = (1.0 - t) * xi_train + t * Y_0_train
        Y_t_eval = (1.0 - t) * xi_eval + t * Y_0_eval

        empirical_queries = torch.as_tensor(
            Y_t_eval[:empirical_count], device=device, dtype=compute_dtype
        )
        empirical_target = empirical_optimal_velocity(
            empirical_queries,
            candidate_tensor,
            t,
            candidate_chunk_size=args.candidate_chunk_size,
        ).cpu().numpy()

        eval_tensor = torch.as_tensor(Y_t_eval, device=device, dtype=compute_dtype)
        wiener_eval = wiener.predict(eval_tensor, t).cpu().numpy()
        rows.append(
            {
                "step": step,
                "t": t,
                "model": "wiener",
                "degree": None,
                "target_mse": mse(wiener_eval, velocity_eval),
                "empirical_optimum_mse": mse(
                    wiener_eval[:empirical_count], empirical_target
                ),
            }
        )
        wiener_state = wiener_state + step_size * wiener.predict(wiener_state, t)

        for degree in args.degrees:
            for family in MODEL_NAMES:
                model = fit_velocity_model(
                    family,
                    degree,
                    Y_t_train,
                    velocity_train,
                    layout,
                    args,
                    device,
                    compute_dtype,
                )
                eval_prediction = model.predict(Y_t_eval, batch_size=args.batch_size)
                rows.append(
                    {
                        "step": step,
                        "t": t,
                        "model": family,
                        "degree": degree,
                        "target_mse": mse(eval_prediction, velocity_eval),
                        "empirical_optimum_mse": mse(
                            eval_prediction[:empirical_count], empirical_target
                        ),
                    }
                )
                key = family, degree
                state_velocity = model.predict(states[key], batch_size=args.batch_size)
                states[key] = states[key] + step_size * state_velocity
                del model

        elapsed = perf_counter() - start_time
        timings.append({"step": step, "t": t, "seconds": elapsed})
        save_metrics(rows, args.out_dir)
        (args.out_dir / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        print(
            f"step {step + 1:03d}/{args.num_steps} t={t:.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    generated = {
        key: transform.synthesize(state.astype(storage_dtype, copy=False), layout)
        for key, state in states.items()
    }
    generated[("wiener", None)] = transform.synthesize(
        wiener_state.cpu().numpy().astype(storage_dtype, copy=False), layout
    )
    np.savez_compressed(
        args.out_dir / "generated_samples.npz",
        initial_noise=initial_images,
        wiener=generated[("wiener", None)],
        **{
            f"{family}_degree_{degree}": generated[(family, degree)]
            for degree in args.degrees
            for family in MODEL_NAMES
        },
    )
    plot_mse_panels(
        rows,
        args.degrees,
        "target_mse",
        r"Test MSE: $\mathbb{E}\|\hat v(Y_t,t)-V_Y\|^2/d$",
        args.out_dir / "mse_vs_time_target.png",
    )
    plot_mse_panels(
        rows,
        args.degrees,
        "empirical_optimum_mse",
        r"MSE to empirical optimum: $\mathbb{E}\|\hat v-v^*_{emp}\|^2/d$",
        args.out_dir / "mse_vs_time_empirical_optimum.png",
    )
    plot_time_averaged_mse(rows, args.degrees, args.out_dir / "mse_vs_degree.png")
    plot_generations(generated, args.degrees, args.out_dir)
    print(f"saved outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
