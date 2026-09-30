"""Readable wavelet masks and paired five-method CIFAR-10 generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pywt
import torch

from wavelet_empirical_cifar.common import save_json
from wavelet_empirical_cifar.masks import artem_threshold_groups, sensitivity_groups
from wavelet_empirical_cifar.wiener import (
    covariance_eigensystem, denoiser_from_eigensystem, empirical_moments,
    restricted_wiener_clean,
)
from wavelet_fm_structure.rgb_wavelets import (
    analyze_rgb, flatten_rgb, synthesize_rgb, unflatten_rgb,
)
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import (
    collect_reference_wavelets, display_rgb, load_cifar10_split, resolve_device,
)
from wavelet_score.transforms import WaveletTransform2D


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_RESULTS = ROOT / "wavelet_empirical_cifar/results/wiener_masks_tau_0p05"
FIXED_RESULTS = ROOT / "wavelet_empirical_cifar/results/wiener_masks"
MAP_TIMES = (0.10, 0.50, 0.90)


def matching_protocol(threshold: dict, fixed: dict) -> None:
    fields = ("seed", "steps", "generation_samples", "wavelet", "level", "mode",
              "image_normalization", "path", "reference_count")
    mismatched = [field for field in fields if threshold[field] != fixed[field]]
    if mismatched:
        raise ValueError(f"Cannot pair generations: protocol differs in {mismatched}")
    if threshold["max_reference_images"] is not None:
        raise ValueError("This figure expects the full CIFAR-10 reference bank")


def packed_mask(transform, layout, sites: np.ndarray) -> np.ndarray:
    scalar = np.zeros((1, layout.num_coefficients), dtype=np.float32)
    scalar[0, sites] = 1.0
    array, _ = pywt.coeffs_to_array(transform.unflatten_coefficients(scalar, layout),
                                    axes=(-2, -1))
    return array[0]


def band_boxes(transform, layout):
    scalar = np.zeros((1, layout.num_coefficients), dtype=np.float32)
    _, slices = pywt.coeffs_to_array(transform.unflatten_coefficients(scalar, layout),
                                     axes=(-2, -1))
    boxes = [(f"A{layout.decomposition_level}", slices[0])]
    for level, detail in zip(range(layout.decomposition_level, 0, -1), slices[1:], strict=True):
        for orientation, key in (("H", "da"), ("V", "ad"), ("D", "dd")):
            boxes.append((f"{orientation}{level}", detail[key]))
    return boxes


def draw_boxes(axis, boxes, annotate):
    for name, region in boxes:
        rows, cols = region[-2:]
        y0, y1 = rows.start or 0, rows.stop
        x0, x1 = cols.start or 0, cols.stop
        axis.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0,
                                 fill=False, edgecolor="#667180", linewidth=.7))
        if annotate:
            axis.text((x0+x1-1)/2, (y0+y1-1)/2, name,
                      ha="center", va="center", fontsize=7,
                      color="#263238", fontweight="bold")


def plot_masks(transform, layout, mask_groups, output):
    targets = tuple(
        layout.subbands[index].flat_start
        + (layout.subbands[index].shape[0] // 2) * layout.subbands[index].shape[1]
        + layout.subbands[index].shape[1] // 2
        for index in (0, 4, 7)
    )
    names = ("A3", "H2", "H1")
    boxes = band_boxes(transform, layout)
    colors = mcolors.ListedColormap(("#f3f5f7", "#137ba6"))
    fig, axes = plt.subplots(3, 4, figsize=(11.4, 8.8), layout="constrained")
    for row, (target, name) in enumerate(zip(targets, names, strict=True)):
        target_pixel = np.argwhere(packed_mask(transform, layout,
                                               np.asarray([target])) > .5)[0]
        axes[row, 0].imshow(np.zeros((32, 32)), cmap=colors, vmin=0, vmax=1,
                            interpolation="nearest")
        draw_boxes(axes[row, 0], boxes, annotate=True)
        axes[row, 0].scatter(target_pixel[1], target_pixel[0], marker="x", c="#df412e", s=45)
        axes[row, 0].set_title(f"Target {name} (q={target})", fontsize=10)
        for col, time in enumerate(MAP_TIMES, start=1):
            group = mask_groups[time][target]
            sites = np.unique(group.input_indices % layout.num_coefficients)
            image = packed_mask(transform, layout, sites)
            axes[row, col].imshow(image, cmap=colors, vmin=0, vmax=1,
                                  interpolation="nearest")
            draw_boxes(axes[row, col], boxes, annotate=False)
            axes[row, col].scatter(target_pixel[1], target_pixel[0], marker="x",
                                   c="#df412e", s=45)
            axes[row, col].set_title(f"t={time:g} | {len(sites)} sites", fontsize=10)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Wiener tau=0.05 mask in wavelet space | blue: observed, red: target", fontsize=12)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def global_wiener_clean(state, mean_t, values, vectors, time):
    query = torch.as_tensor(state, device=values.device, dtype=torch.float64)
    centered = query - time * mean_t
    gains = time * values / (time*time*values + (1-time)**2)
    return (mean_t + ((centered @ vectors) * gains) @ vectors.T).cpu().numpy()


def generate_affine_pair(initial, transform, layout, mean, covariance, eigen,
                         sensitivity_groups_by_time, anchors, steps, device):
    initial_y, check = analyze_rgb(initial, transform)
    if check != layout:
        raise RuntimeError("Saved protocol and noise DWT layouts differ")
    start = flatten_rgb(initial_y).astype(np.float64)
    states = {"wiener_global": start.copy(), "wiener_restricted_sensitivity9": start.copy()}
    mean_t = torch.as_tensor(mean, dtype=torch.float64, device=device)
    values, vectors = eigen
    for step in range(steps):
        time = step / steps
        tic = perf_counter()
        global_clean = global_wiener_clean(states["wiener_global"], mean_t,
                                            values, vectors, time)
        selected = min(anchors, key=lambda anchor: abs(anchor-time))
        restricted_clean = restricted_wiener_clean(
            states["wiener_restricted_sensitivity9"], mean, covariance, time,
            sensitivity_groups_by_time[selected], device=device)
        states["wiener_global"] += (global_clean - states["wiener_global"]) / ((1-time)*steps)
        states["wiener_restricted_sensitivity9"] += (
            restricted_clean - states["wiener_restricted_sensitivity9"]) / ((1-time)*steps)
        if any(not np.isfinite(state).all() for state in states.values()):
            raise FloatingPointError(f"Non-finite affine state at step {step}")
        print(f"affine Euler {step+1}/{steps} t={time:.3f} {perf_counter()-tic:.1f}s", flush=True)
    return {name: synthesize_rgb(unflatten_rgb(state.astype(np.float32), layout),
                                 layout, transform) for name, state in states.items()}


def plot_generations(columns: dict[str, np.ndarray], output: Path):
    first = next(iter(columns.values()))
    count = len(first)
    if first.ndim != 4 or first.shape[1] != 3 or first.shape[2] != first.shape[3]:
        raise ValueError("Expected square RGB image batches")
    if any(images.shape != first.shape for images in columns.values()):
        raise ValueError("All compared images must have the same batch shape")
    fig, axes = plt.subplots(count, len(columns), figsize=(2.45*len(columns), 2.5*count),
                             squeeze=False, layout="constrained")
    for col, (title, images) in enumerate(columns.items()):
        axes[0, col].set_title(title, fontsize=10)
        for row in range(count):
            axes[row, col].imshow(display_rgb(images[row].transpose(1, 2, 0)),
                                  interpolation="nearest")
            axes[row, col].axis("off")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_saved_four(threshold_results: Path, fixed_results: Path) -> Path:
    output = threshold_results / "generation_comparison_four.png"
    plot_generations({
        "Empirical closed-form\n(Wiener tau=0.05 mask)":
            np.load(threshold_results / "generated_artem_threshold.npy"),
        "Empirical closed-form\n(sensitivity-9 mask)":
            np.load(threshold_results / "generated_sensitivity_9.npy"),
        "Global affine Wiener":
            np.load(threshold_results / "generated_wiener_global.npy"),
        "FM U-Net": np.load(fixed_results / "generated_fm_unet.npy"),
    }, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-results", type=Path, default=THRESHOLD_RESULTS)
    parser.add_argument("--fixed-results", type=Path, default=FIXED_RESULTS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--maps-only", action="store_true")
    parser.add_argument("--four-only", action="store_true",
                        help="Plot four saved generations without recomputing masks or samples")
    args = parser.parse_args()
    if args.maps_only and args.four_only:
        parser.error("--maps-only and --four-only cannot be combined")
    threshold = json.loads((args.threshold_results / "config.json").read_text())
    fixed = json.loads((args.fixed_results / "config.json").read_text())
    matching_protocol(threshold, fixed)
    selection = json.loads((args.fixed_results / "selection.json").read_text())
    if selection["selected_budget_rgb_sites"] != 9:
        raise ValueError("This comparison requires the saved sensitivity-9 baseline")
    if args.four_only:
        print(f"Saved {plot_saved_four(args.threshold_results, args.fixed_results)}", flush=True)
        return
    device = resolve_device(args.device)
    transform = WaveletTransform2D(threshold["wavelet"], threshold["level"], threshold["mode"])
    train = load_cifar10_split("train", 512, None, False, 0)
    references, layout = collect_reference_wavelets(train, transform)
    mean, covariance = empirical_moments(references, device=device)
    eigen = covariance_eigensystem(covariance, device)
    anchors = tuple(threshold["times"])
    wanted = set(anchors if not args.maps_only else MAP_TIMES)
    threshold_groups = {}
    sensitivity = {}
    for time in sorted(wanted):
        denoiser = denoiser_from_eigensystem(*eigen, time)
        if time in MAP_TIMES:
            threshold_groups[time] = artem_threshold_groups(layout, denoiser,
                                                              threshold["artem_threshold"])
        if not args.maps_only:
            sensitivity[time] = sensitivity_groups(layout, denoiser, 9)
    plot_masks(transform, layout, threshold_groups,
               args.threshold_results / "artem_wavelet_mask_mosaic.png")
    print("Saved wavelet mask mosaic", flush=True)
    if args.maps_only:
        return

    count = threshold["generation_samples"]
    initial = np.random.default_rng(threshold["seed"]+1).standard_normal(
        (count, 3, 32, 32)).astype(np.float32)
    generated = generate_affine_pair(initial, transform, layout, mean, covariance,
                                     eigen, sensitivity, anchors, threshold["steps"], device)
    for name, images in generated.items():
        np.save(args.threshold_results / f"generated_{name}.npy", images)
    unet = np.load(args.fixed_results / "generated_fm_unet.npy")
    threshold_images = np.load(args.threshold_results / "generated_artem_threshold.npy")
    sensitivity_images = np.load(args.threshold_results / "generated_sensitivity_9.npy")
    plot_generations({
        "Empirical closed-form\n(Wiener tau=0.05 mask)": threshold_images,
        "Empirical closed-form\n(sensitivity-9 mask)": sensitivity_images,
        "Global affine Wiener": generated["wiener_global"],
        "Restricted affine Wiener\n(sensitivity-9 mask)": generated["wiener_restricted_sensitivity9"],
        "FM U-Net": unet,
    }, args.threshold_results / "generation_comparison_five.png")
    save_json(args.threshold_results / "comparison_five_protocol.json", {
        "same_initial_noise_seed": threshold["seed"]+1,
        "euler_steps": threshold["steps"],
        "wiener_global": "full RGB wavelet covariance, affine conditional clean mean",
        "wiener_restricted": "same sensitivity-9 groups as empirical, affine regression recomputed at each Euler t",
        "sensitivity_masks": "nearest of five saved times",
        "unet_source": str(args.fixed_results / "generated_fm_unet.npy"),
        "wavelet": threshold["wavelet"],
    })
    print("Saved five-method comparison", flush=True)


if __name__ == "__main__":
    main()
