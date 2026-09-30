"""Add a paired CIFAR-10 FM U-Net row to saved Wiener-mask generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_empirical_cifar.common import make_model
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import display_rgb, resolve_device
from fm_comparisons.compare_fm_unet_closed_form_mse import predict_fm_unet_velocity


@torch.no_grad()
def generate_unet(model, initial: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    state = torch.as_tensor(initial, device=device, dtype=torch.float32).clone()
    for step in range(steps):
        state.add_(predict_fm_unet_velocity(model, state, step / steps) / steps)
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"Non-finite U-Net state at step {step}")
        print(f"U-Net Euler {step+1}/{steps}", flush=True)
    return state.cpu().numpy()


def compare(results: Path, device: torch.device):
    config = json.loads((results / "config.json").read_text())
    selection = json.loads((results / "selection.json").read_text())
    budget = selection["selected_budget_rgb_sites"]
    names = (f"geometric_{budget}", f"sensitivity_{budget}")
    saved = {name: np.load(results / f"generated_{name}.npy") for name in names}
    threshold_path = results / "generated_artem_threshold.npy"
    if threshold_path.exists():
        saved["artem_threshold"] = np.load(threshold_path)
    count = config["generation_samples"]
    expected_shape = (count, 3, 32, 32)
    if any(images.shape != expected_shape for images in saved.values()):
        raise ValueError("Saved generations do not match the recorded protocol")
    if config["path"] != "x_t=(1-t)*noise+t*clean":
        raise ValueError("Unexpected flow path in the saved protocol")
    initial = np.random.default_rng(config["seed"] + 1).standard_normal(
        expected_shape).astype(np.float32)
    model = make_model(device, True)
    unet = generate_unet(model, initial, config["steps"], device)
    np.save(results / "generated_fm_unet.npy", unet)
    columns = {"FM U-Net": unet, "Geometric mask": saved[names[0]],
               "Wiener sensitivity mask": saved[names[1]]}
    if "artem_threshold" in saved:
        columns["Artem threshold"] = saved["artem_threshold"]
    fig, axes = plt.subplots(count, len(columns),
                             figsize=(2.75*len(columns), 2.5*count), squeeze=False)
    for col, (name, images) in enumerate(columns.items()):
        axes[0, col].set_title(name, fontsize=11)
        for row in range(count):
            axes[row, col].imshow(display_rgb(images[row].transpose(1, 2, 0)),
                                  interpolation="nearest")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(results / "generation_comparison_unet.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
        default=Path("wavelet_empirical_cifar/results/wiener_masks"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    compare(args.results, resolve_device(args.device))


if __name__ == "__main__":
    main()
