"""Generate CIFAR-10 samples from a trained AsymFlow U-Net.

The checkpoint model predicts the low-rank AsymFlow target

    u_A(x_t, t) ~= x_1 - P eps,

with our Flow Matching convention

    x_t = t x_1 + (1 - t) eps.

For sampling, we reconstruct a full velocity with

    u = P u_A + (I - P) (u_A - x_t) / (1 - t),

then integrate from noise to data with uniform-step Euler.

Input: trained checkpoint and matching PCA basis. Output: sample grid and tensor.
Example: python -m asymflow.generate_asymflow_unet_cifar10 --checkpoint PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import trange

from asymflow.train_asymflow_unet_cifar10 import (
    build_cifar10_unet,
    load_cifar10_dataset,
    load_patchwise_pca_basis,
    load_pca_basis,
    project_low_rank,
)


DEFAULT_CHECKPOINT = Path(
    "asymflow/results/asymflow_unet_cifar10_paper_loss/"
    "rank_0256_steps_400001/checkpoints/latest.pt"
)
DEFAULT_OUT_DIR = Path(
    "asymflow/results/asymflow_unet_cifar10_paper_loss/"
    "rank_0256_steps_400001/generation"
)


def parse_args() -> argparse.Namespace:
    """Parse sampling options."""
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 samples from a trained AsymFlow U-Net."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument(
        "--projection-mode",
        choices=["global", "patchwise"],
        default=None,
        help="Defaults to the projection mode stored in the checkpoint config.",
    )
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument(
        "--patch-pca-dir",
        type=Path,
        default=Path("asymflow/results/patchwise_pca_cifar10"),
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=4e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-raw-model", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def infer_rank(args: argparse.Namespace, checkpoint: dict) -> int:
    """Return the PCA rank, preferring an explicit CLI value."""
    if args.rank is not None:
        return args.rank
    config = checkpoint.get("config", {})
    if "rank" not in config:
        raise ValueError("Checkpoint config does not contain rank; pass --rank explicitly.")
    return int(config["rank"])


def infer_projection_mode(args: argparse.Namespace, checkpoint: dict) -> str:
    """Return global or patchwise, preferring an explicit CLI value."""
    if args.projection_mode is not None:
        return args.projection_mode
    return str(checkpoint.get("config", {}).get("projection_mode", "global"))


def infer_patch_size(args: argparse.Namespace, checkpoint: dict) -> int:
    """Return the patch size for patch-wise checkpoints."""
    if args.patch_size is not None:
        return args.patch_size
    return int(checkpoint.get("config", {}).get("patch_size", 8))


def load_projection_basis(
    rank: int,
    projection_mode: str,
    patch_size: int,
    patch_pca_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the PCA basis matching the checkpoint projection mode."""
    if projection_mode == "global":
        return load_pca_basis(rank, device)
    if projection_mode == "patchwise":
        return load_patchwise_pca_basis(rank, patch_size, patch_pca_dir, device)
    raise ValueError(f"Unknown projection_mode: {projection_mode}")


def load_asymflow_model(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[nn.Module, dict]:
    """Load the trained AsymFlow U-Net and checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = "ema_model" if use_ema else "model"
    if state_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain key: {state_key}")

    model = build_cifar10_unet(device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, checkpoint


def reconstruct_full_velocity(
    asymflow_output: torch.Tensor,
    xt: torch.Tensor,
    t: float,
    basis: torch.Tensor,
    projection_mode: str,
    patch_size: int,
    sigma_min: float,
) -> torch.Tensor:
    """Convert the AsymFlow low-rank output into a full image-space velocity."""
    projected_output = project_low_rank(asymflow_output, basis, projection_mode, patch_size)
    sigma = max(1.0 - t, sigma_min)
    complement_argument = (asymflow_output - xt) / sigma
    projected_complement_argument = project_low_rank(
        complement_argument, basis, projection_mode, patch_size
    )
    complement_velocity = complement_argument - projected_complement_argument
    return projected_output + complement_velocity


@torch.no_grad()
def euler_generate(
    model: nn.Module,
    initial_noise: torch.Tensor,
    basis: torch.Tensor,
    projection_mode: str,
    patch_size: int,
    num_steps: int,
    sigma_min: float,
    show_progress: bool = True,
) -> torch.Tensor:
    """Integrate the reconstructed velocity from t=0 to t=1."""
    samples = initial_noise.clone()
    step_size = 1.0 / num_steps

    for step in trange(num_steps, desc="Euler sampling", disable=not show_progress):
        t = (step + 0.5) / num_steps
        t_batch = torch.full(
            (samples.shape[0],),
            t,
            device=samples.device,
            dtype=samples.dtype,
        )
        asymflow_output = model(t_batch, samples)
        velocity = reconstruct_full_velocity(
            asymflow_output=asymflow_output,
            xt=samples,
            t=t,
            basis=basis,
            projection_mode=projection_mode,
            patch_size=patch_size,
            sigma_min=sigma_min,
        )
        samples = samples + step_size * velocity

    return samples


def save_image_grid(samples: torch.Tensor, out_path: Path) -> None:
    """Save generated images as a square-ish grid."""
    images = ((samples.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    images = images.permute(0, 2, 3, 1).numpy()

    num_samples = images.shape[0]
    num_cols = min(8, num_samples)
    num_rows = (num_samples + num_cols - 1) // num_cols

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(2.0 * num_cols, 2.0 * num_rows),
        squeeze=False,
    )
    for index, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if index < num_samples:
            ax.imshow(images[index], interpolation="nearest")
        else:
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    rank: int,
    projection_mode: str,
    patch_size: int,
    checkpoint: dict,
) -> None:
    """Save generation metadata next to the generated samples."""
    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "rank": rank,
        "projection_mode": projection_mode,
        "patch_size": patch_size if projection_mode == "patchwise" else None,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sigma_min": args.sigma_min,
        "seed": args.seed,
        "weights": "model" if args.use_raw_model else "ema_model",
    }
    with (out_dir / "generation_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Load the trained model, sample images, and save a grid."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_asymflow_model(
        checkpoint_path=args.checkpoint,
        device=device,
        use_ema=not args.use_raw_model,
    )
    rank = infer_rank(args, checkpoint)
    projection_mode = infer_projection_mode(args, checkpoint)
    patch_size = infer_patch_size(args, checkpoint)
    basis, spectrum = load_projection_basis(
        rank=rank,
        projection_mode=projection_mode,
        patch_size=patch_size,
        patch_pca_dir=args.patch_pca_dir,
        device=device,
    )
    explained_variance = spectrum[:rank].sum() / spectrum.sum()

    dataset = load_cifar10_dataset(batch_size=args.num_samples)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generator,
    )

    print("dataset: cifar10")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_step: {checkpoint.get('step', 'unknown')}")
    print(f"weights: {'model' if args.use_raw_model else 'ema_model'}")
    print(f"projection_mode: {projection_mode}")
    print(f"rank: {rank}")
    if projection_mode == "patchwise":
        print(f"patch_size: {patch_size}")
        print(f"patch_pca_dir: {args.patch_pca_dir}")
    print(f"explained_variance_at_rank: {explained_variance.item():.4f}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"sigma_min: {args.sigma_min}")

    samples = euler_generate(
        model=model,
        initial_noise=initial_noise,
        basis=basis,
        projection_mode=projection_mode,
        patch_size=patch_size,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_image_grid(samples, args.out_dir / "samples.png")
    torch.save(samples.detach().cpu(), args.out_dir / "samples.pt")
    save_metadata(args.out_dir, args, rank, projection_mode, patch_size, checkpoint)
    print(f"saved samples to {args.out_dir}")


if __name__ == "__main__":
    main()
