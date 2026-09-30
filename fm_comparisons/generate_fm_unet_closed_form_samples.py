"""Generate CIFAR-10 samples with FM U-Net and closed-form velocities.

The same initial Gaussian noises are evolved with four velocity fields:
FM U-Net, x1_mask, velocity_mask, and the biased linear Wiener velocity.
The script saves one final-image grid for direct visual comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (
    MASK_KINDS,
    binarize_like_artem,
    build_fm_wiener_operator,
    build_mask_operator,
    dataset_output_slug,
    default_checkpoint_for_dataset,
    load_dataset,
    load_experiment_config,
    load_fm_unet,
    load_or_compute_wiener_svd,
    predict_closed_form_x1,
    predict_fm_unet_velocity,
    predict_wiener_velocity,
)

GENERATOR_KINDS = ("fm_unet", "x1_mask", "velocity_mask", "wiener_velocity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


@torch.no_grad()
def predict_closed_form_velocity(
    xt: torch.Tensor,
    t: float,
    candidate_dataset,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Predict velocity by first estimating x1, then applying (x1_hat - x_t)/(1-t)."""
    x1_hat = predict_closed_form_x1(
        xt=xt,
        t=t,
        candidate_dataset=candidate_dataset,
        mask=mask,
        temperature=temperature,
    )
    return (x1_hat - xt) / (1.0 - t)


@torch.no_grad()
def euler_generate(
    initial_noise: torch.Tensor,
    model: nn.Module,
    candidate_dataset,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    num_steps: int,
    temperature: float,
    mask_threshold: float,
) -> dict[str, torch.Tensor]:
    """Run uniform-step Euler generation for all velocity fields."""
    samples = {kind: initial_noise.clone() for kind in GENERATOR_KINDS}
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
        masks = {
            mask_kind: binarize_like_artem(
                build_mask_operator(wiener_operator, t, mask_kind),
                mask_threshold,
            )
            for mask_kind in MASK_KINDS
        }

        samples["fm_unet"] = samples["fm_unet"] + step_size * predict_fm_unet_velocity(
            model, samples["fm_unet"], t
        )
        samples["wiener_velocity"] = samples["wiener_velocity"] + step_size * predict_wiener_velocity(
            samples["wiener_velocity"], t, wiener_operator, mean
        )

        for mask_kind in MASK_KINDS:
            velocity = predict_closed_form_velocity(
                xt=samples[mask_kind],
                t=t,
                candidate_dataset=candidate_dataset,
                mask=masks[mask_kind],
                temperature=temperature,
            )
            samples[mask_kind] = samples[mask_kind] + step_size * velocity

    return samples


def save_final_grid(
    samples: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save a grid with rows=samples and columns=velocity fields."""
    num_samples = next(iter(samples.values())).shape[0]
    num_cols = len(GENERATOR_KINDS)
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(2.1 * num_cols, 2.1 * num_samples),
        squeeze=False,
    )

    for col, kind in enumerate(GENERATOR_KINDS):
        axes[0, col].set_title(kind, fontsize=16)
        images = postprocess(samples[kind].detach().cpu())
        images = images.permute(0, 2, 3, 1).numpy()
        for row in range(num_samples):
            ax = axes[row, col]
            ax.imshow(images[row], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path(
        "fm_comparisons/results"
    ) / f"fm_generation_comparison_{dataset_output_slug(args.dataset)}"
    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)

    cfg = load_experiment_config(args.dataset, args.batch_size, subset_size=None)
    candidate_cfg = load_experiment_config(args.dataset, args.batch_size, subset_size=args.num_xi)
    pca_params = cfg.model.params or {}
    temperature = float(pca_params.get("temperature", 1.0))
    mask_threshold = float(pca_params.get("mask_threshold", 0.02))

    dataset = load_dataset(cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    model = load_fm_unet(checkpoint_path, device, args.dataset)
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(candidate_dataset, device)

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"num_xi: {args.num_xi}")
    print(f"temperature: {temperature}")
    print(f"mask_threshold: {mask_threshold}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generator,
    )

    samples = euler_generate(
        initial_noise=initial_noise,
        model=model,
        candidate_dataset=candidate_dataset,
        U=U,
        eigenvalues=eigenvalues,
        Vh=Vh,
        mean=mean,
        num_steps=args.num_steps,
        temperature=temperature,
        mask_threshold=mask_threshold,
    )

    out_path = out_dir / "final_samples.png"
    save_final_grid(samples, dataset.postprocess, out_path)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
