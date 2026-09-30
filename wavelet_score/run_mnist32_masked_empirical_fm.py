"""Benchmark exact masked empirical wavelet FM estimators on MNIST-32.

Inputs: official MNIST images, wavelet mask family and time grid. Outputs:
velocity-risk tables and plots in ``--out-dir``. Example::

    python -m wavelet_score.run_mnist32_masked_empirical_fm --help
"""

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
from wavelet_score.masked_empirical_fm import MaskedEmpiricalWaveletVelocity
from wavelet_score.transforms import WaveletTransform2D

FAMILIES = ("independent", "band_tied", "local", "local_band_tied")
COLORS = {
    "independent": "#0072B2",
    "band_tied": "#D55E00",
    "local": "#009E73",
    "local_band_tied": "#CC79A7",
}


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_score/results/mnist32_masked_empirical_fm_small"),
    )
    parser.add_argument("--wavelet", choices=("haar", "db2"), default="haar")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--times", type=str, default="0,0.25,0.5,0.75,0.9,0.95,0.99")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--num-references", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=64)
    parser.add_argument("--load-batch-size", type=int, default=1024)
    parser.add_argument("--reference-chunk-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--group-chunk-size", type=int, default=32)
    parser.add_argument("--endpoint-epsilon", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.times = parse_float_list(args.times)
    if not args.times or any(t < 0 or t > 1 - args.endpoint_epsilon for t in args.times):
        parser.error("Every time must lie in [0,1-endpoint-epsilon].")
    if min(args.num_references, args.num_test, args.radius + 1) < 1:
        parser.error("Sample counts must be positive and radius non-negative.")
    if min(args.reference_chunk_size, args.query_batch_size, args.group_chunk_size) < 1:
        parser.error("Chunk sizes must be positive.")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def squared_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first.astype(np.float64, copy=False) - second.astype(np.float64, copy=False)) ** 2


def add_metric(
    rows: list[dict[str, object]],
    family: str,
    t: float,
    target: str,
    domain: str,
    region: str,
    value: float,
) -> None:
    rows.append({
        "family": family,
        "t": t,
        "target": target,
        "domain": domain,
        "region": region,
        "mse": value,
    })


def plot_global_mse(rows: list[dict[str, object]], out_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, target, title in zip(
        axes,
        ("velocity", "clean"),
        (r"Velocity: $\mathbb{E}\|\hat V_Y-V_Y\|^2/d$", r"Clean: $\mathbb{E}\|\hat Y_0-Y_0\|^2/d$"),
        strict=True,
    ):
        for family in FAMILIES:
            selected = [
                row for row in rows
                if row["family"] == family
                and row["target"] == target
                and row["domain"] == "wavelet"
                and row["region"] == "all"
            ]
            axis.plot(
                [row["t"] for row in selected],
                [row["mse"] for row in selected],
                marker="o",
                linewidth=2,
                label=family.replace("_", " "),
                color=COLORS[family],
            )
        axis.set(title=title, xlabel="Flow time t", ylabel="Mean squared error")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def plot_subband_mse(rows: list[dict[str, object]], out_path: Path) -> None:
    regions = sorted({str(row["region"]) for row in rows if row["region"] != "all"})
    figure, axes = plt.subplots(
        len(FAMILIES), 1, figsize=(11, 3.1 * len(FAMILIES)), sharex=True, constrained_layout=True
    )
    for axis, family in zip(axes, FAMILIES, strict=True):
        for region in regions:
            selected = [
                row for row in rows
                if row["family"] == family
                and row["target"] == "velocity"
                and row["domain"] == "wavelet"
                and row["region"] == region
            ]
            axis.plot([row["t"] for row in selected], [row["mse"] for row in selected], marker=".", label=region)
        axis.set_title(family.replace("_", " "))
        axis.set_ylabel("Velocity MSE")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Flow time t")
    axes[0].legend(ncol=5, fontsize=7, frameon=False)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    numpy_dtype = np.float32 if args.dtype == "float32" else np.float64
    print(f"device: {device}; wavelet: {args.wavelet}; level: {args.level}", flush=True)

    train = load_mnist_32(
        args.data_root, "train", batch_size=args.load_batch_size,
        max_samples=args.num_references, download=args.download,
    )
    test = load_mnist_32(
        args.data_root, "test", batch_size=args.load_batch_size,
        max_samples=args.num_test, download=args.download,
    )
    transform = WaveletTransform2D(args.wavelet, args.level, mode="periodization")
    reference_Y0, layout = transform.analyze(train.images.astype(numpy_dtype, copy=False))
    test_Y0, test_layout = transform.analyze(test.images.astype(numpy_dtype, copy=False))
    if test_layout != layout:
        raise RuntimeError("Train and test wavelet layouts differ.")
    generator = np.random.default_rng(args.seed)
    noise = generator.standard_normal(test_Y0.shape).astype(numpy_dtype)

    config = {
        **{
            key: value for key, value in vars(args).items()
            if key not in ("data_root", "out_dir", "times")
        },
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "times": list(args.times),
        "families": list(FAMILIES),
        "resolved_device": str(device),
        "num_coefficients": layout.num_coefficients,
        "path": "Y_t=(1-t)xi+tY_0",
        "target": "V_Y=Y_0-xi",
        "clipping_for_reported_mse": False,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows: list[dict[str, object]] = []
    benchmarks: list[dict[str, object]] = []
    for family in FAMILIES:
        estimator = MaskedEmpiricalWaveletVelocity(
            family=family,
            radius=args.radius,
            reference_chunk_size=args.reference_chunk_size,
            query_batch_size=args.query_batch_size,
            group_chunk_size=args.group_chunk_size,
            device=device,
            dtype=torch_dtype,
            endpoint_epsilon=args.endpoint_epsilon,
        ).fit(reference_Y0, layout)
        for t in args.times:
            Y_t = (1.0 - t) * noise + t * test_Y0
            target_velocity = test_Y0 - noise
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = perf_counter()
            clean_prediction = estimator.predict_clean(Y_t, t)
            velocity_prediction = (clean_prediction - Y_t) / (1.0 - t)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
            )
            clean_pixels = transform.synthesize(clean_prediction, layout)
            velocity_pixels = transform.synthesize(velocity_prediction, layout)
            target_velocity_pixels = transform.synthesize(target_velocity, layout)
            clean_error = squared_error(clean_prediction, test_Y0)
            velocity_error = squared_error(velocity_prediction, target_velocity)
            add_metric(rows, family, t, "clean", "wavelet", "all", float(clean_error.mean()))
            add_metric(rows, family, t, "velocity", "wavelet", "all", float(velocity_error.mean()))
            add_metric(
                rows, family, t, "clean", "pixel", "all",
                float(squared_error(clean_pixels, test.images).mean()),
            )
            add_metric(
                rows, family, t, "velocity", "pixel", "all",
                float(squared_error(velocity_pixels, target_velocity_pixels).mean()),
            )
            for descriptor in layout.subbands:
                region = (
                    f"A_L{descriptor.pywt_level}" if descriptor.coefficient_type == "scaling"
                    else f"{descriptor.orientation}_L{descriptor.pywt_level}"
                )
                add_metric(
                    rows, family, t, "clean", "wavelet", region,
                    float(clean_error[:, descriptor.flat_slice].mean()),
                )
                add_metric(
                    rows, family, t, "velocity", "wavelet", region,
                    float(velocity_error[:, descriptor.flat_slice].mean()),
                )
            benchmarks.append({
                "family": family,
                "t": t,
                "seconds": elapsed,
                "queries_per_second": len(test_Y0) / elapsed,
                "peak_cuda_memory_mib": peak_memory,
            })
            print(
                f"{family:17s} t={t:.3f} velocity_mse={velocity_error.mean():.6f} "
                f"clean_mse={clean_error.mean():.6f} time={elapsed:.2f}s",
                flush=True,
            )

    (args.out_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out_dir / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2) + "\n")
    with (args.out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "benchmarks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmarks[0]))
        writer.writeheader()
        writer.writerows(benchmarks)
    plot_global_mse(rows, args.out_dir / "mse_vs_time.png")
    plot_subband_mse(rows, args.out_dir / "subband_velocity_mse_vs_time.png")
    print(f"saved outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
