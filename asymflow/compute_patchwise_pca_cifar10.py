"""Compute a shared patch-wise PCA basis for CIFAR-10 images.

This script builds the projection used by the patch-wise AsymFlow experiments.
It treats every image patch as one sample in R^(3 * patch_size * patch_size),
then computes one PCA basis shared by all patch positions.

Input: CIFAR-10 via the upstream dataset loader. Output: basis and eigenvalues
under asymflow/results/patchwise_pca_cifar10 by default.
Example: python -m asymflow.compute_patchwise_pca_cifar10 --patch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALITY_ROOT = PROJECT_ROOT
LOCALITY_SRC = LOCALITY_ROOT / "src"


DEFAULT_OUT_DIR = Path("asymflow/results/patchwise_pca_cifar10")


def parse_args() -> argparse.Namespace:
    """Parse patch PCA settings."""
    parser = argparse.ArgumentParser(
        description="Compute a shared patch-wise PCA basis for CIFAR-10."
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@contextmanager
def working_directory(path: Path):
    """Temporarily run code from a different working directory."""
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def ensure_locality_importable() -> None:
    """Make the locality repo source package importable in this repo."""
    if not LOCALITY_SRC.exists():
        raise FileNotFoundError(f"Could not find locality source directory: {LOCALITY_SRC}")
    sys.path.insert(0, str(LOCALITY_SRC))


def load_cifar10_dataset(batch_size: int):
    """Load CIFAR-10 from the locality repo with its [-1, 1] transform."""
    ensure_locality_importable()
    from local_diffusion.configuration import load_config
    from local_diffusion.data import build_dataset

    with working_directory(LOCALITY_ROOT):
        cfg = load_config(
            "configs/pca_locality/cifar10.yaml",
            overrides=[
                "metrics.wandb.enabled=false",
                f"dataset.batch_size={batch_size}",
            ],
        )
    return build_dataset(cfg.dataset)


def patchify_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images [B, C, H, W] into patches [B, M, C*p*p]."""
    batch_size, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size {(height, width)} is not divisible by patch_size={patch_size}."
        )

    patches_per_height = height // patch_size
    patches_per_width = width // patch_size
    patches = images.reshape(
        batch_size,
        channels,
        patches_per_height,
        patch_size,
        patches_per_width,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(
        batch_size,
        patches_per_height * patches_per_width,
        channels * patch_size * patch_size,
    )


def unpatchify_images(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert patches [B, M, C*p*p] back into images [B, C, H, W]."""
    batch_size, num_patches, patch_dim = patches.shape
    patches_per_height = height // patch_size
    patches_per_width = width // patch_size
    expected_num_patches = patches_per_height * patches_per_width
    expected_patch_dim = channels * patch_size * patch_size

    if num_patches != expected_num_patches:
        raise ValueError(f"Expected {expected_num_patches} patches, got {num_patches}.")
    if patch_dim != expected_patch_dim:
        raise ValueError(f"Expected patch_dim={expected_patch_dim}, got {patch_dim}.")

    patches = patches.reshape(
        batch_size,
        patches_per_height,
        patches_per_width,
        channels,
        patch_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.reshape(batch_size, channels, height, width)


@torch.no_grad()
def compute_patch_mean(
    dataloader,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Compute the mean patch vector over all CIFAR-10 patches."""
    patch_sum: torch.Tensor | None = None
    num_patches_seen = 0

    for batch in tqdm(dataloader, desc="patch mean"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        patches = patchify_images(images, patch_size)  # [B, M, patch_dim]
        patches = patches.reshape(-1, patches.shape[-1]).double()  # [B*M, patch_dim]

        if patch_sum is None:
            patch_sum = torch.zeros(patches.shape[-1], device=device, dtype=torch.float64)

        patch_sum += patches.sum(dim=0)
        num_patches_seen += patches.shape[0]

    if patch_sum is None:
        raise RuntimeError("No CIFAR-10 patches were processed.")
    return patch_sum / num_patches_seen, num_patches_seen


@torch.no_grad()
def compute_patch_covariance(
    dataloader,
    patch_size: int,
    patch_mean: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Accumulate sum_z (z - mean)(z - mean)^T over all CIFAR-10 patches."""
    patch_cov_sum: torch.Tensor | None = None
    num_patches_seen = 0

    for batch in tqdm(dataloader, desc="patch covariance"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        patches = patchify_images(images, patch_size)
        patches = patches.reshape(-1, patches.shape[-1]).double()
        patches = patches - patch_mean

        if patch_cov_sum is None:
            patch_dim = patches.shape[-1]
            patch_cov_sum = torch.zeros((patch_dim, patch_dim), device=device, dtype=torch.float64)

        patch_cov_sum += patches.T @ patches
        num_patches_seen += patches.shape[0]

    if patch_cov_sum is None:
        raise RuntimeError("No CIFAR-10 patches were processed.")
    return patch_cov_sum / num_patches_seen, num_patches_seen


def save_patch_pca(
    out_dir: Path,
    patch_size: int,
    patch_mean: torch.Tensor,
    covariance: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    num_patches: int,
) -> None:
    """Save patch PCA tensors and metadata."""
    patch_dir = out_dir / f"patch_size_{patch_size:03d}"
    patch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(patch_mean.detach().cpu().float(), patch_dir / "patch_mean.pt")
    torch.save(covariance.detach().cpu().float(), patch_dir / "covariance.pt")
    torch.save(eigenvalues.detach().cpu().float(), patch_dir / "eigenvalues.pt")
    torch.save(eigenvectors.detach().cpu().float(), patch_dir / "eigenvectors.pt")
    torch.save(eigenvectors.detach().cpu().float(), patch_dir / "basis_full.pt")

    metadata = {
        "dataset": "cifar10",
        "patch_size": patch_size,
        "patch_dim": int(patch_mean.numel()),
        "num_patches": int(num_patches),
        "centered": True,
        "pca_convention": "covariance = E[(z - mean)(z - mean)^T]",
        "eigen_order": "descending",
    }
    with (patch_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Compute the patch Gram matrix used for patch-wise PCA."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    patch_dir = args.out_dir / f"patch_size_{args.patch_size:03d}"
    expected_files = [
        patch_dir / "patch_mean.pt",
        patch_dir / "covariance.pt",
        patch_dir / "eigenvalues.pt",
        patch_dir / "basis_full.pt",
        patch_dir / "metadata.json",
    ]
    if not args.overwrite and all(path.exists() for path in expected_files):
        print(f"Patch-wise PCA already exists: {patch_dir}")
        print("Use --overwrite to recompute it.")
        return

    dummy = torch.randn(2, 3, 32, 32, device=device)
    patches = patchify_images(dummy, args.patch_size)
    reconstructed = unpatchify_images(
        patches=patches,
        patch_size=args.patch_size,
        channels=3,
        height=32,
        width=32,
    )
    max_error = (dummy - reconstructed).abs().max()
    print(f"patch_size: {args.patch_size}")
    print(f"input_shape: {tuple(dummy.shape)}")
    print(f"patches_shape: {tuple(patches.shape)}")
    print(f"reconstructed_shape: {tuple(reconstructed.shape)}")
    print(f"max_reconstruction_error: {max_error.item():.2e}")
    print(f"patch_dim: {patches.shape[-1]}")
    print(f"num_patches_per_image: {patches.shape[1]}")

    dataset = load_cifar10_dataset(args.batch_size)
    patch_mean, num_patches_seen = compute_patch_mean(
        dataloader=dataset.dataloader,
        patch_size=args.patch_size,
        device=device,
    )
    covariance, num_cov_patches_seen = compute_patch_covariance(
        dataloader=dataset.dataloader,
        patch_size=args.patch_size,
        patch_mean=patch_mean,
        device=device,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(dims=(0,)).contiguous()
    eigenvectors = eigenvectors.flip(dims=(1,)).contiguous()
    save_patch_pca(
        out_dir=args.out_dir,
        patch_size=args.patch_size,
        patch_mean=patch_mean,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        num_patches=num_patches_seen,
    )
    print(f"num_images: {len(dataset.dataset)}")
    print(f"num_patches_seen: {num_patches_seen}")
    print(f"num_cov_patches_seen: {num_cov_patches_seen}")
    print(f"patch_mean_shape: {tuple(patch_mean.shape)}")
    print(f"covariance_shape: {tuple(covariance.shape)}")
    print(f"eigenvectors_shape: {tuple(eigenvectors.shape)}")
    print(f"saved_to: {args.out_dir / f'patch_size_{args.patch_size:03d}'}")


if __name__ == "__main__":
    main()
