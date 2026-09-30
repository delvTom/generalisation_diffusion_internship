"""CelebA-HQ counterpart of the four-column CIFAR-10 Wiener-mask gallery.

The full 30,000-image empirical bank and every wavelet output site are used.
Masks and Euler states are cached so an interrupted run can resume.
"""

from __future__ import annotations

import argparse
import json
from math import isqrt
from pathlib import Path

import numpy as np
import torch

from fm_comparisons.compare_fm_unet_closed_form_mse import (
    DEFAULT_CELEBA_HQ_CHECKPOINT,
    fm_unet_velocity_forward,
    load_experiment_config,
    load_fm_unet,
)
from local_diffusion.data import build_dataset
from local_diffusion.utils.wiener import load_wiener_filter
from wavelet_empirical_cifar.common import save_json
from wavelet_empirical_cifar.masks import (
    artem_threshold_groups,
    groups_from_sites,
    sensitivity_groups,
)
from wavelet_empirical_cifar.report_wiener_threshold import plot_generations
from wavelet_fm_structure.rgb_wavelets import (
    analyze_rgb,
    flatten_rgb,
    synthesize_rgb,
    unflatten_rgb,
)
from wavelet_score.masked_empirical_fm import MaskedEmpiricalWaveletVelocity
from wavelet_score.transforms import WaveletTransform2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "wavelet_empirical_cifar/results/celeba_hq_tau_0p05"
DEFAULT_WIENER = ROOT / "data/models/wiener/celeba_hq_64"
MASK_TIMES = (0.10, 0.30, 0.50, 0.75, 0.90)
METHODS = ("artem_threshold", "sensitivity_9", "wiener_global", "fm_unet")


def wavelet_svd_columns(matrix, transform, device, batch_size=64):
    """Transform the image-shaped columns of an SVD factor, without a DWT matrix."""
    dimension = matrix.shape[0]
    resolution = isqrt(dimension // 3)
    if matrix.shape != (dimension, dimension) or dimension != 3 * resolution**2:
        raise ValueError("Expected a full square RGB SVD factor")
    result = torch.empty((dimension, dimension), dtype=torch.float32, device=device)
    for start in range(0, dimension, batch_size):
        stop = min(start + batch_size, dimension)
        images = matrix[:, start:stop].T.contiguous().numpy().reshape(
            -1, 3, resolution, resolution)
        coefficients, _ = analyze_rgb(images, transform)
        result[:, start:stop] = torch.as_tensor(
            flatten_rgb(coefficients).T.copy(), device=device)
        print(f"Wiener columns in wavelets: {stop}/{dimension}", flush=True)
    return result


def save_groups(path, groups, dimension):
    lengths = [len(group.input_indices) // 3 for group in groups]
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    indices = np.concatenate([group.input_indices[::3] % dimension for group in groups])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, offsets=offsets, indices=indices)
    temporary.replace(path)


def load_groups(path, layout, family):
    with np.load(path, allow_pickle=False) as saved:
        offsets, indices = saved["offsets"], saved["indices"]
    if len(offsets) != layout.num_coefficients + 1 or offsets[-1] != len(indices):
        raise ValueError(f"Invalid mask cache: {path}")
    sites = [indices[offsets[q]:offsets[q + 1]]
             for q in range(layout.num_coefficients)]
    return groups_from_sites(layout, sites, family)


def build_mask_cache(layout, transform, wiener_dir, out_dir, device):
    missing = [time for time in MASK_TIMES
               if any(not (out_dir / f"mask_{name}_t{time:.2f}.npz").exists()
                      for name in ("artem_threshold", "sensitivity_9"))]
    if not missing:
        print("Using cached wavelet masks", flush=True)
        return

    u, values, vh, _ = load_wiener_filter(wiener_dir, device=torch.device("cpu"))
    if u.shape != (3 * 64 * 64, 3 * 64 * 64):
        raise ValueError("Cached Wiener factor does not match CelebA-HQ 64x64")
    left = wavelet_svd_columns(u, transform, device)
    right = wavelet_svd_columns(vh.T, transform, device)
    values = values.to(device=device, dtype=torch.float32)
    for time in missing:
        gains = time * values / (time * time * values + (1 - time) ** 2)
        denoiser = ((left * gains.unsqueeze(0)) @ right.T).cpu().numpy()
        for name, make in (
            ("artem_threshold", lambda: artem_threshold_groups(layout, denoiser, .05)),
            ("sensitivity_9", lambda: sensitivity_groups(layout, denoiser, 9)),
        ):
            path = out_dir / f"mask_{name}_t{time:.2f}.npz"
            if not path.exists():
                groups = make()
                save_groups(path, groups, layout.num_coefficients)
                sizes = [len(group.input_indices) // 3 for group in groups]
                print(f"Saved {name} t={time:.2f}: median={np.median(sizes):g}, "
                      f"max={max(sizes)} RGB sites", flush=True)
        del denoiser
    del left, right, u, vh
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_mask_schedule(out_dir, layout, name):
    return {time: load_groups(out_dir / f"mask_{name}_t{time:.2f}.npz", layout, name)
            for time in MASK_TIMES}


def save_state(path, state, next_step):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, state=state, next_step=np.int64(next_step))
    temporary.replace(path)


def resume_state(path, initial):
    if not path.exists():
        return initial.copy(), 0
    with np.load(path, allow_pickle=False) as saved:
        state, next_step = saved["state"], int(saved["next_step"])
    if state.shape != initial.shape or state.dtype != initial.dtype:
        raise ValueError(f"Checkpoint is incompatible with this run: {path}")
    return state, next_step


def generate_wiener(initial, wiener_dir, steps, out_dir, device):
    output = out_dir / "generated_wiener_global.npy"
    if output.exists():
        return
    u, values, vh, mean = load_wiener_filter(wiener_dir, device=device)
    u, vh, values, mean = (item.to(dtype=torch.float32) for item in (u, vh, values, mean))
    checkpoint = out_dir / "state_wiener_global.npz"
    state, start = resume_state(checkpoint, initial)
    state = torch.as_tensor(state, device=device)
    for step in range(start, steps):
        time = step / steps
        gains = time * values / (time*time*values + (1-time)**2)
        flat = state.flatten(start_dim=1)
        clean = mean + (((flat - time*mean) @ vh.T) * gains) @ u.T
        state = (flat + (clean - flat) / ((1-time)*steps)).reshape_as(state)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite Wiener state at step {step}")
        save_state(checkpoint, state.cpu().numpy(), step + 1)
        print(f"Wiener Euler {step + 1}/{steps}", flush=True)
    np.save(output, state.cpu().numpy())
    del u, vh, values, mean
    if device.type == "cuda":
        torch.cuda.empty_cache()


def generate_unet(initial, checkpoint_path, steps, batch_size, out_dir, device):
    output = out_dir / "generated_fm_unet.npy"
    if output.exists():
        return
    model = load_fm_unet(checkpoint_path, device, "celeba_hq")
    checkpoint = out_dir / "state_fm_unet.npz"
    state, start = resume_state(checkpoint, initial)
    with torch.no_grad():
        for step in range(start, steps):
            time = step / steps
            for begin in range(0, len(state), batch_size):
                end = min(begin + batch_size, len(state))
                batch = torch.as_tensor(state[begin:end].copy(), device=device)
                velocity = fm_unet_velocity_forward(model, batch, time)
                state[begin:end] = (batch + velocity / steps).cpu().numpy()
            if not np.isfinite(state).all():
                raise FloatingPointError(f"Nonfinite FM U-Net state at step {step}")
            save_state(checkpoint, state, step + 1)
            print(f"FM U-Net Euler {step + 1}/{steps}", flush=True)
    np.save(output, state)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def collect_reference_wavelets(dataset, transform):
    chunks = []
    layout = None
    total = len(dataset.dataset)
    collected = 0
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        pixels = images.numpy().astype(np.float32, copy=False)
        if pixels.min() < -1.000001 or pixels.max() > 1.000001:
            raise RuntimeError("CelebA-HQ images must be normalized to [-1,1]")
        coefficients, observed = analyze_rgb(pixels, transform)
        if layout is None:
            layout = observed
        elif layout != observed:
            raise RuntimeError("Inconsistent wavelet layouts")
        chunks.append(flatten_rgb(coefficients))
        collected += len(images)
        print(f"CelebA-HQ reference DWT: {collected}/{total}", flush=True)
    if layout is None:
        raise RuntimeError("Reference bank is empty")
    return np.ascontiguousarray(np.concatenate(chunks)), layout


def verify_wiener_reference_mean(references, transform, wiener_dir):
    cached_pixel_mean = torch.load(
        wiener_dir / "mean.pt", map_location="cpu", weights_only=True)
    cached_wavelets, _ = analyze_rgb(
        cached_pixel_mean.numpy().reshape(1, 3, 64, 64), transform)
    cached = flatten_rgb(cached_wavelets)[0].astype(np.float64)
    observed = np.mean(references, axis=0, dtype=np.float64)
    relative_error = np.linalg.norm(cached - observed) / max(np.linalg.norm(observed), 1e-12)
    print(f"Wiener cache / reference-bank mean relative error: {relative_error:.3e}", flush=True)
    if relative_error > 1e-3:
        raise RuntimeError("Cached Wiener mean does not match the empirical reference bank")


def generate_empirical(name, initial, references, layout, transform, steps, out_dir,
                       device, reference_chunk, query_batch, group_chunk):
    output = out_dir / f"generated_{name}.npy"
    if output.exists():
        return
    schedule = load_mask_schedule(out_dir, layout, name)
    initial_coefficients, observed = analyze_rgb(initial, transform)
    if observed != layout:
        raise RuntimeError("Initial-noise wavelet layout differs from references")
    initial_state = flatten_rgb(initial_coefficients)
    checkpoint = out_dir / f"state_{name}.npz"
    state, start = resume_state(checkpoint, initial_state)
    estimator = MaskedEmpiricalWaveletVelocity(
        family="independent", device=device, dtype=torch.float32,
        reference_chunk_size=reference_chunk, query_batch_size=query_batch,
        group_chunk_size=group_chunk, endpoint_epsilon=1e-6,
    ).fit_with_groups(references, layout, schedule[MASK_TIMES[0]])
    for step in range(start, steps):
        time = step / steps
        anchor = min(MASK_TIMES, key=lambda value: abs(value - time))
        estimator.set_groups(schedule[anchor])
        state += estimator.predict_velocity(state, time) / steps
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Nonfinite {name} state at step {step}")
        save_state(checkpoint, state, step + 1)
        print(f"{name} Euler {step + 1}/{steps} (mask t={anchor:g})", flush=True)
    images = synthesize_rgb(unflatten_rgb(state, layout), layout, transform)
    np.save(output, images)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-reference-images", type=int, default=None,
                        help="Explicit screening cap; omit for all 30,000 CelebA-HQ images")
    parser.add_argument("--reference-chunk", type=int, default=512)
    parser.add_argument("--query-batch", type=int, default=4)
    parser.add_argument("--group-chunk", type=int, default=8)
    parser.add_argument("--unet-batch", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_samples, args.steps, args.reference_chunk, args.query_batch,
           args.group_chunk, args.unet_batch) < 1:
        parser.error("All counts and chunk sizes must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this full experiment on a GPU host")
    checkpoint_path = ROOT / DEFAULT_CELEBA_HQ_CHECKPOINT
    required = [checkpoint_path] + [DEFAULT_WIENER / f"{name}.pt"
                                    for name in ("U", "LA", "Vh", "mean")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pretrained assets: {missing}")
    config = load_experiment_config("celeba_hq", 256, args.max_reference_images,
                                    split="train", num_workers=0)
    dataset = build_dataset(config.dataset)
    if dataset.resolution != 64 or dataset.in_channels != 3:
        raise RuntimeError("Expected CelebA-HQ normalized RGB at 64x64")
    if args.max_reference_images is None and len(dataset.dataset) != 30000:
        raise RuntimeError("Expected the full 30,000-image CelebA-HQ bank")
    protocol = {
        "dataset": "celeba_hq", "resolution": 64, "reference_count": len(dataset.dataset),
        "num_samples": args.num_samples, "steps": args.steps, "seed": args.seed,
        "path": "x_t=(1-t)*noise+t*clean", "normalization": "[-1,1]",
        "wavelet": "db2", "level": 3, "mode": "periodization",
        "mask_times": MASK_TIMES, "mask_threshold": .05, "sensitivity_sites": 9,
        "unet_checkpoint": str(checkpoint_path), "wiener_cache": str(DEFAULT_WIENER),
    }
    print(json.dumps(protocol, indent=2), flush=True)
    if args.validate_only:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != json.loads(json.dumps(protocol)):
            raise ValueError("Existing output uses a different protocol; choose another --out-dir")
    else:
        save_json(config_path, protocol)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    transform = WaveletTransform2D("db2", 3, "periodization")
    _, layout = transform.analyze(np.zeros((1, 64, 64), dtype=np.float32))
    initial = np.random.default_rng(args.seed + 1).standard_normal(
        (args.num_samples, 3, 64, 64)).astype(np.float32)
    build_mask_cache(layout, transform, DEFAULT_WIENER, args.out_dir, device)
    generate_wiener(initial, DEFAULT_WIENER, args.steps, args.out_dir, device)
    generate_unet(initial, checkpoint_path, args.steps, args.unet_batch, args.out_dir, device)
    if any(not (args.out_dir / f"generated_{name}.npy").exists()
           for name in ("artem_threshold", "sensitivity_9")):
        references, reference_layout = collect_reference_wavelets(dataset, transform)
        if reference_layout != layout:
            raise RuntimeError("Dataset and mask wavelet layouts differ")
        if args.max_reference_images is None:
            verify_wiener_reference_mean(references, transform, DEFAULT_WIENER)
        for name in ("artem_threshold", "sensitivity_9"):
            generate_empirical(name, initial, references, layout, transform, args.steps,
                               args.out_dir, device, args.reference_chunk, args.query_batch,
                               args.group_chunk)
    images = {name: np.load(args.out_dir / f"generated_{name}.npy") for name in METHODS}
    plot_generations({
        "Empirical closed-form\n(Wiener tau=0.05 mask)": images["artem_threshold"],
        "Empirical closed-form\n(sensitivity-9 mask)": images["sensitivity_9"],
        "Global affine Wiener": images["wiener_global"],
        "FM U-Net": images["fm_unet"],
    }, args.out_dir / "generation_comparison_four.png")
    print(f"Saved {args.out_dir / 'generation_comparison_four.png'}", flush=True)


if __name__ == "__main__":
    main()
