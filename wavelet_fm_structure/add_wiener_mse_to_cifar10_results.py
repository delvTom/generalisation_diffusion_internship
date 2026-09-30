"""Add the affine-Wiener MSE baseline to a completed CIFAR-10 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fm_comparisons.compare_fm_unet_closed_form_mse import (
    load_fm_unet,
    load_or_compute_wiener_svd,
)
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import (
    NUM_CHANNELS,
    PROJECT_ROOT,
    collect_indexed_images,
    evaluate_wiener_against_model,
    load_cifar10_split,
    plot_model_mse,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/cifar10_fm_calibrated_radius"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--data-batch-size", type=int, default=512)
    parser.add_argument("--model-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if min(args.data_batch_size, args.model_batch_size) < 1:
        parser.error("Batch sizes must be positive.")
    if args.num_workers < 0:
        parser.error("num-workers must be non-negative.")
    return args


def reproduce_mse_noise(
    seed: int,
    dataset_size: int,
    calibration_indices: np.ndarray,
    mse_indices: np.ndarray,
) -> np.ndarray:
    """Replay the original RNG sequence to recover the exact MSE noises."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dataset_size)
    calibration_count = len(calibration_indices)
    mse_count = len(mse_indices)
    if not np.array_equal(permutation[:calibration_count], calibration_indices):
        raise RuntimeError("Stored calibration indices do not match the recorded seed.")
    expected_mse = permutation[
        calibration_count : calibration_count + mse_count
    ]
    if not np.array_equal(expected_mse, mse_indices):
        raise RuntimeError("Stored MSE indices do not match the recorded seed.")

    calibration_shape = (calibration_count, NUM_CHANNELS, 32, 32)
    rng.standard_normal(calibration_shape)
    return rng.standard_normal(
        (mse_count, NUM_CHANNELS, 32, 32)
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    metrics_path = args.results_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Wait for the main experiment to finish first."
        )
    results = json.loads(metrics_path.read_text())
    protocol = results["protocol"]
    if protocol.get("dataset") != "CIFAR-10":
        raise RuntimeError("The stored experiment is not the CIFAR-10 protocol.")

    device = resolve_device(args.device)
    test = load_cifar10_split(
        "test", args.data_batch_size, None, args.download, args.num_workers
    )
    train = load_cifar10_split(
        "train", args.data_batch_size, None, args.download, args.num_workers
    )
    calibration_indices = np.asarray(
        protocol["calibration_query_indices"], dtype=np.int64
    )
    mse_indices = np.asarray(protocol["mse_query_indices"], dtype=np.int64)
    clean_images = collect_indexed_images(test, mse_indices)
    noise = reproduce_mse_noise(
        int(protocol["seed"]),
        len(test.dataset),
        calibration_indices,
        mse_indices,
    )

    existing_metrics = results["mse_vs_fm_unet"]
    times = tuple(
        sorted(
            {
                float(row["t"])
                for row in existing_metrics
                if row["family"] != "wiener_velocity"
            }
        )
    )
    checkpoint = args.checkpoint or Path(protocol["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"FM checkpoint not found: {checkpoint}")

    model = load_fm_unet(checkpoint, device, "cifar10")
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(train, device)
    wiener_metrics = evaluate_wiener_against_model(
        model,
        clean_images,
        noise,
        times,
        U,
        eigenvalues,
        Vh,
        mean,
        device,
        args.model_batch_size,
    )

    results["mse_vs_fm_unet"] = [
        row for row in existing_metrics if row["family"] != "wiener_velocity"
    ] + wiener_metrics
    protocol["mse_baseline"] = (
        "affine Wiener velocity including empirical-mean bias"
    )
    protocol["wiener_mse_added_by_postprocessing"] = True
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n")
    temporary_path.replace(metrics_path)
    plot_model_mse(
        results["mse_vs_fm_unet"], args.results_dir / "mse_vs_fm_unet.png"
    )
    print(f"updated {metrics_path}")
    print(f"updated {args.results_dir / 'mse_vs_fm_unet.png'}")


if __name__ == "__main__":
    main()
