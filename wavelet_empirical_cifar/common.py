"""Shared CIFAR-10 protocol and compact reporting for three independent studies."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import (  # noqa: E402
    collect_indexed_images,
    collect_reference_wavelets,
    display_rgb,
    load_cifar10_split,
    model_wavelet_state_and_velocity,
    resolve_device,
)
from wavelet_fm_structure.rgb_wavelets import analyze_rgb, flatten_rgb, synthesize_rgb, unflatten_rgb  # noqa: E402
from wavelet_score.masked_empirical_fm import MaskedEmpiricalWaveletVelocity  # noqa: E402
from wavelet_score.transforms import WaveletTransform2D  # noqa: E402

TIMES = (0.10, 0.30, 0.50, 0.75, 0.90)


def parser(description: str, output: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--out-dir", type=Path, default=Path(output))
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--max-reference-images", type=int, default=None,
                        help="Explicit screening cap; omitted uses all CIFAR-10 training images")
    result.add_argument("--calibration-queries", type=int, default=8)
    result.add_argument("--evaluation-queries", type=int, default=16)
    result.add_argument("--generation-samples", type=int, default=4)
    result.add_argument("--steps", type=int, default=30)
    result.add_argument("--reference-chunk", type=int, default=512)
    result.add_argument("--query-batch", type=int, default=4)
    result.add_argument("--group-chunk", type=int, default=16)
    result.add_argument("--seed", type=int, default=1729)
    result.add_argument("--no-generation", action="store_true")
    result.add_argument("--no-unet", action="store_true")
    return result


def prepare(args: argparse.Namespace):
    if min(args.calibration_queries, args.evaluation_queries, args.generation_samples,
           args.steps, args.reference_chunk, args.query_batch, args.group_chunk) < 1:
        raise ValueError("Counts and chunk sizes must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    transform = WaveletTransform2D("db2", 3, "periodization")
    train = load_cifar10_split("train", 512, args.max_reference_images, False, 0)
    test = load_cifar10_split("test", 512, None, False, 0)
    if args.calibration_queries + args.evaluation_queries > len(test.dataset):
        raise ValueError("Calibration and evaluation sets must be disjoint")
    references, layout = collect_reference_wavelets(train, transform)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(test.dataset))
    cal_ids = indices[:args.calibration_queries]
    eval_ids = indices[args.calibration_queries:args.calibration_queries + args.evaluation_queries]
    clean = {"calibration": collect_indexed_images(test, cal_ids),
             "evaluation": collect_indexed_images(test, eval_ids)}
    noise = {key: rng.standard_normal(value.shape).astype(np.float32) for key, value in clean.items()}
    generation_noise = np.random.default_rng(args.seed + 1).standard_normal(
        (args.generation_samples, 3, 32, 32)).astype(np.float32)
    config = dict(vars(args))
    config.update(wavelet="db2", level=3, mode="periodization", image_normalization="[-1,1]",
                  path="x_t=(1-t)*noise+t*clean", reference_count=len(references),
                  calibration_indices=cal_ids.tolist(), evaluation_indices=eval_ids.tolist(),
                  times=TIMES)
    config["out_dir"] = str(config["out_dir"])
    save_json(args.out_dir / "config.json", config)
    return device, transform, references, layout, clean, noise, generation_noise


def make_estimator(args, device, references, layout, groups):
    return MaskedEmpiricalWaveletVelocity(
        family="independent", device=device, dtype=torch.float32,
        reference_chunk_size=args.reference_chunk,
        query_batch_size=args.query_batch,
        group_chunk_size=args.group_chunk,
        endpoint_epsilon=1e-6,
    ).fit_with_groups(references, layout, groups)


def make_model(device, use_unet):
    if not use_unet:
        return None
    from fm_comparisons.compare_fm_unet_closed_form_mse import DEFAULT_CHECKPOINT, load_fm_unet
    checkpoint = Path(DEFAULT_CHECKPOINT)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"U-Net checkpoint missing: {checkpoint}; pass --no-unet explicitly")
    return load_fm_unet(checkpoint, device, "cifar10")


def state_and_targets(clean, noise, time, transform):
    clean_y, layout = analyze_rgb(clean, transform)
    noise_y, noise_layout = analyze_rgb(noise, transform)
    if layout != noise_layout:
        raise RuntimeError("DWT layouts differ")
    clean_flat = flatten_rgb(clean_y)
    state = (1-time)*flatten_rgb(noise_y) + time*clean_flat
    return np.ascontiguousarray(state), clean_flat


def mse(prediction, target):
    return float(np.mean((prediction.astype(np.float64) - target.astype(np.float64))**2))


def evaluate_estimator(estimator, groups, clean, noise, time, transform, model, device):
    state, target = state_and_targets(clean, noise, time, transform)
    estimator.set_groups(groups)
    start = perf_counter()
    prediction = estimator.predict_clean(state, time)
    result = {"clean_mse": mse(prediction, target), "seconds": perf_counter()-start}
    if model is not None:
        _, velocity = model_wavelet_state_and_velocity(model, clean, noise, time,
                                                        transform, device, 32)
        model_clean = state + (1-time)*velocity
        result["unet_clean_mse"] = mse(model_clean, target)
        result["agreement_mse"] = mse(prediction, model_clean)
    return result


def generate_estimator(estimator, group_at_time, initial_pixels, transform, layout, steps):
    initial_y, check = analyze_rgb(initial_pixels, transform)
    if check != layout:
        raise RuntimeError("Generation DWT layout differs")
    state = flatten_rgb(initial_y).copy()
    for step in range(steps):
        time = step / steps
        estimator.set_groups(group_at_time(time))
        state += estimator.predict_velocity(state, time) / steps
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite Euler state at step {step}")
        print(f"Euler {step+1}/{steps}", flush=True)
    return synthesize_rgb(unflatten_rgb(state, layout), layout, transform)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def save_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_mse(rows, output, metric="clean_mse", ylabel="Clean-image MSE, test"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in dict.fromkeys(row["variant"] for row in rows):
        selected = [row for row in rows if row["variant"] == variant and row["split"] == "evaluation"]
        ax.plot([row["time"] for row in selected], [row[metric] for row in selected],
                marker="o", label=variant)
    ax.set(xlabel="Flow time t", ylabel=ylabel, title="CIFAR-10: reconstruction")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_generation_grid(generated, references, transform, layout, out_dir, device):
    from wavelet_fm_structure.generate_full_adaptive_gallery import nearest_reference_indices
    reference_t = torch.as_tensor(np.ascontiguousarray(references), device=device)
    rows = []
    for variant, images in generated.items():
        coefficients, check = analyze_rgb(images.astype(np.float32), transform)
        if check != layout:
            raise RuntimeError("Generated layout differs")
        indices, squared = nearest_reference_indices(flatten_rgb(coefficients), reference_t, 8192)
        neighbors = synthesize_rgb(unflatten_rgb(references[indices], layout), layout, transform)
        rows.append((variant, images, neighbors))
        np.save(out_dir / f"generated_{variant}.npy", images)
        save_rows(out_dir / f"neighbors_{variant}.csv", [
            {"sample": j, "reference_bank_row": int(index), "pixel_l2": float(np.sqrt(distance)),
             "pixel_mse": float(distance / (3*32*32))}
            for j, (index, distance) in enumerate(zip(indices, squared, strict=True))])
    fig, axes = plt.subplots(len(rows), 2*len(rows[0][1]), figsize=(2.2*2*len(rows[0][1]), 2.5*len(rows)))
    axes = np.asarray(axes).reshape(len(rows), -1)
    for row, (name, images, neighbors) in enumerate(rows):
        for col in range(len(images)):
            axes[row, 2*col].imshow(display_rgb(images[col].transpose(1, 2, 0)))
            axes[row, 2*col+1].imshow(display_rgb(neighbors[col].transpose(1, 2, 0)))
            axes[row, 2*col].set_title(f"{name} / {col}", fontsize=8)
            axes[row, 2*col+1].set_title("train NN", fontsize=8)
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "generations_and_neighbors.png", dpi=200)
    plt.close(fig)
