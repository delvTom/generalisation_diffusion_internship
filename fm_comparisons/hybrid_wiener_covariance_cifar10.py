"""Evaluate an orthogonal hybrid Wiener covariance on CIFAR-10.

The experiment compares three affine Wiener denoisers in the Flow Matching
convention x_t = t x_1 + (1 - t) eps:

    global_wiener:
        full global covariance from Artem's cached CIFAR-10 Wiener SVD.
    patch_wiener:
        block-local covariance built from non-overlapping image patches.
    hybrid_wiener_orthogonal:
        top global PCA covariance plus a patch covariance built on residuals
        and projected onto the orthogonal complement of the global subspace.

The script saves metric curves and sensitivity-field visualizations.
Inputs: CIFAR-10 and a matching FM U-Net checkpoint; output: metrics and plots.
Example: python -m fm_comparisons.hybrid_wiener_covariance_cifar10 --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from compare_fm_unet_closed_form_mse import (  # noqa: E402
    collect_images,
    default_checkpoint_for_dataset,
    fm_unet_velocity_forward,
    load_dataset,
    load_experiment_config,
    load_fm_unet,
    load_or_compute_wiener_svd,
    parse_t_values,
)


MODEL_KINDS = ("global_wiener", "patch_wiener", "hybrid_wiener_orthogonal")


def parse_args() -> argparse.Namespace:
    """Parse experiment settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--global-rank", type=int, default=64)
    parser.add_argument("--cov-subset-size", type=int, default=None)
    parser.add_argument("--num-xt", type=int, default=128)
    parser.add_argument("--xt-batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--t-values", default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95")
    parser.add_argument("--sigma-min", type=float, default=4e-2)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/models/hybrid_wiener"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fm_comparisons/results/hybrid_wiener_covariance_cifar10"),
    )
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--skip-unet", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def pixel_index(channel: int, y: int, x: int, height: int, width: int) -> int:
    """Map (channel, y, x) to a CHW-flattened index."""
    return channel * height * width + y * width + x


def default_target_pixels(height: int, width: int) -> list[tuple[int, int]]:
    """Return a compact set of target pixels for sensitivity plots."""
    center_y, center_x = height // 2, width // 2
    offset = max(1, min(height, width) // 5)
    return [
        (center_y, center_x),
        (max(0, center_y - offset), center_x),
        (min(height - 1, center_y + offset), center_x),
        (center_y, max(0, center_x - offset)),
        (center_y, min(width - 1, center_x + offset)),
    ]


def patch_indices(
    num_channels: int,
    height: int,
    width: int,
    patch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return CHW-flattened indices for each non-overlapping patch."""
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image shape {(height, width)} is not divisible by patch_size={patch_size}.")

    indices: list[torch.Tensor] = []
    for patch_y in range(0, height, patch_size):
        for patch_x in range(0, width, patch_size):
            patch_indices_list = []
            for channel in range(num_channels):
                for y in range(patch_y, patch_y + patch_size):
                    for x in range(patch_x, patch_x + patch_size):
                        patch_indices_list.append(pixel_index(channel, y, x, height, width))
            indices.append(torch.tensor(patch_indices_list, device=device, dtype=torch.long))
    return indices


def extract_patches_flat(
    flat_images: torch.Tensor,
    patch_index_list: list[torch.Tensor],
) -> torch.Tensor:
    """Extract patches from CHW-flattened images as [B, M, Dp]."""
    patches = [flat_images.index_select(dim=1, index=indices) for indices in patch_index_list]
    return torch.stack(patches, dim=1)


def project_global(flat: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project row-vector batches onto span(basis)."""
    if basis.shape[1] == 0:
        return torch.zeros_like(flat)
    return (flat @ basis) @ basis.T


@torch.no_grad()
def compute_patch_means(
    dataloader,
    mean: torch.Tensor,
    global_basis: torch.Tensor,
    patch_index_list: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute patch means for centered data and global-residual data."""
    patch_sum: torch.Tensor | None = None
    residual_patch_sum: torch.Tensor | None = None
    num_patches = 0

    for batch in tqdm(dataloader, desc="patch means"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        flat = images.to(device=device, dtype=torch.float32).flatten(start_dim=1)
        centered = flat - mean.unsqueeze(0)
        residual = centered - project_global(centered, global_basis)
        patches = extract_patches_flat(centered, patch_index_list).reshape(-1, len(patch_index_list[0]))
        residual_patches = extract_patches_flat(residual, patch_index_list).reshape(-1, len(patch_index_list[0]))

        if patch_sum is None:
            patch_sum = torch.zeros(patches.shape[-1], device=device, dtype=torch.float64)
            residual_patch_sum = torch.zeros_like(patch_sum)

        patch_sum += patches.double().sum(dim=0)
        residual_patch_sum += residual_patches.double().sum(dim=0)
        num_patches += patches.shape[0]

    if patch_sum is None or residual_patch_sum is None:
        raise RuntimeError("No patches were processed.")
    return patch_sum / num_patches, residual_patch_sum / num_patches, num_patches


@torch.no_grad()
def compute_patch_covariances(
    dataloader,
    mean: torch.Tensor,
    global_basis: torch.Tensor,
    patch_index_list: list[torch.Tensor],
    patch_mean: torch.Tensor,
    residual_patch_mean: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute local patch covariance and residual local patch covariance."""
    patch_cov: torch.Tensor | None = None
    residual_patch_cov: torch.Tensor | None = None
    num_patches = 0

    for batch in tqdm(dataloader, desc="patch covariances"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        flat = images.to(device=device, dtype=torch.float32).flatten(start_dim=1)
        centered = flat - mean.unsqueeze(0)
        residual = centered - project_global(centered, global_basis)

        patches = extract_patches_flat(centered, patch_index_list).reshape(-1, len(patch_index_list[0])).double()
        residual_patches = extract_patches_flat(residual, patch_index_list).reshape(-1, len(patch_index_list[0])).double()
        patches = patches - patch_mean
        residual_patches = residual_patches - residual_patch_mean

        if patch_cov is None:
            patch_dim = patches.shape[-1]
            patch_cov = torch.zeros((patch_dim, patch_dim), device=device, dtype=torch.float64)
            residual_patch_cov = torch.zeros_like(patch_cov)

        patch_cov += patches.T @ patches
        residual_patch_cov += residual_patches.T @ residual_patches
        num_patches += patches.shape[0]

    if patch_cov is None or residual_patch_cov is None:
        raise RuntimeError("No patches were processed.")
    denom = max(1, num_patches - 1)
    return patch_cov / denom, residual_patch_cov / denom, num_patches


def build_block_patch_covariance(
    patch_cov: torch.Tensor,
    patch_index_list: list[torch.Tensor],
    num_pixels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Embed one shared patch covariance as a block-local image covariance."""
    covariance = torch.zeros((num_pixels, num_pixels), device=device, dtype=dtype)
    patch_cov = patch_cov.to(device=device, dtype=dtype)
    for indices in patch_index_list:
        covariance[indices[:, None], indices[None, :]] = patch_cov
    return covariance


def eigendecompose_psd(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize and eigendecompose a covariance matrix in descending order."""
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    eigenvalues = eigenvalues.flip(dims=(0,)).contiguous()
    eigenvectors = eigenvectors.flip(dims=(1,)).contiguous()
    eigenvalues = eigenvalues.clamp(min=0.0).float()
    eigenvectors = eigenvectors.float()
    return eigenvalues, eigenvectors


def cache_path(args: argparse.Namespace, dataset_name: str, resolution: int) -> Path:
    """Return the cache path for hybrid covariance eigensystems."""
    return (
        args.cache_dir
        / f"{dataset_name}_{resolution}"
        / f"patch_{args.patch_size:03d}_global_rank_{args.global_rank:04d}.pt"
    )


@torch.no_grad()
def build_or_load_covariance_models(
    dataset,
    U_full: torch.Tensor,
    eigenvalues_full: torch.Tensor,
    mean: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Build or load global, patch, and orthogonal-hybrid covariance eigensystems."""
    path = cache_path(args, dataset.name, dataset.resolution)
    if path.exists() and not args.recompute:
        print(f"loading cached hybrid covariance from {path}")
        return torch.load(path, map_location=device, weights_only=True)

    num_pixels = dataset.in_channels * dataset.resolution * dataset.resolution
    if args.global_rank < 0 or args.global_rank > U_full.shape[1]:
        raise ValueError(f"global_rank must be in [0, {U_full.shape[1]}], got {args.global_rank}.")

    mean = mean.to(device=device, dtype=torch.float32)
    U_global = U_full[:, : args.global_rank].to(device=device, dtype=torch.float32).contiguous()
    lambda_global = eigenvalues_full[: args.global_rank].to(device=device, dtype=torch.float32).contiguous()
    patch_index_list = patch_indices(
        num_channels=dataset.in_channels,
        height=dataset.resolution,
        width=dataset.resolution,
        patch_size=args.patch_size,
        device=device,
    )

    patch_mean, residual_patch_mean, num_mean_patches = compute_patch_means(
        dataloader=dataset.dataloader,
        mean=mean,
        global_basis=U_global,
        patch_index_list=patch_index_list,
        device=device,
    )
    patch_cov, residual_patch_cov, num_cov_patches = compute_patch_covariances(
        dataloader=dataset.dataloader,
        mean=mean,
        global_basis=U_global,
        patch_index_list=patch_index_list,
        patch_mean=patch_mean,
        residual_patch_mean=residual_patch_mean,
        device=device,
    )

    print("building dense patch covariance")
    B_patch = build_block_patch_covariance(
        patch_cov=patch_cov,
        patch_index_list=patch_index_list,
        num_pixels=num_pixels,
        dtype=torch.float32,
        device=device,
    )
    B_res_patch = build_block_patch_covariance(
        patch_cov=residual_patch_cov,
        patch_index_list=patch_index_list,
        num_pixels=num_pixels,
        dtype=torch.float32,
        device=device,
    )

    print("orthogonalizing residual patch covariance")
    identity = torch.eye(num_pixels, device=device, dtype=torch.float32)
    if U_global.shape[1] == 0:
        Q_perp = identity
        Sigma_global_lr = torch.zeros_like(B_patch)
    else:
        projector = U_global @ U_global.T
        Q_perp = identity - projector
        Sigma_global_lr = (U_global * lambda_global.unsqueeze(0)) @ U_global.T

    B_perp = Q_perp @ B_res_patch @ Q_perp
    B_perp = 0.5 * (B_perp + B_perp.T)
    Sigma_hybrid = Sigma_global_lr + B_perp
    Sigma_hybrid = 0.5 * (Sigma_hybrid + Sigma_hybrid.T)

    print("eigendecomposing patch and hybrid covariances")
    patch_eigenvalues, patch_eigenvectors = eigendecompose_psd(B_patch)
    hybrid_eigenvalues, hybrid_eigenvectors = eigendecompose_psd(Sigma_hybrid)

    diagnostics = {
        "num_mean_patches": int(num_mean_patches),
        "num_cov_patches": int(num_cov_patches),
        "global_rank": int(args.global_rank),
        "patch_size": int(args.patch_size),
        "patch_dim": int(len(patch_index_list[0])),
        "num_patches_per_image": int(len(patch_index_list)),
        "U_orthogonality_mean_abs": float((U_global.T @ U_global - torch.eye(U_global.shape[1], device=device)).abs().mean().item())
        if U_global.shape[1] > 0
        else 0.0,
        "residual_global_overlap_mean_abs": 0.0,
        "U_T_B_perp_mean_abs": float((U_global.T @ B_perp).abs().mean().item()) if U_global.shape[1] > 0 else 0.0,
        "B_perp_U_mean_abs": float((B_perp @ U_global).abs().mean().item()) if U_global.shape[1] > 0 else 0.0,
        "patch_eig_min": float(patch_eigenvalues.min().item()),
        "patch_eig_max": float(patch_eigenvalues.max().item()),
        "hybrid_eig_min": float(hybrid_eigenvalues.min().item()),
        "hybrid_eig_max": float(hybrid_eigenvalues.max().item()),
    }

    models = {
        "mean": mean.detach().cpu(),
        "global_eigenvalues": eigenvalues_full.detach().cpu().float(),
        "global_eigenvectors": U_full.detach().cpu().float(),
        "patch_eigenvalues": patch_eigenvalues.detach().cpu(),
        "patch_eigenvectors": patch_eigenvectors.detach().cpu(),
        "hybrid_eigenvalues": hybrid_eigenvalues.detach().cpu(),
        "hybrid_eigenvectors": hybrid_eigenvectors.detach().cpu(),
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(models, path)
    print(f"saved hybrid covariance cache to {path}")
    print(json.dumps(diagnostics, indent=2))
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in models.items()}


def wiener_operator_from_eig(
    t: float,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build W_t from covariance eigensystem for x_t = t x_1 + (1-t) eps."""
    eigenvalues = eigenvalues.to(dtype=torch.float32)
    eigenvectors = eigenvectors.to(dtype=torch.float32)
    shrink = t * eigenvalues / (t * t * eigenvalues + (1.0 - t) ** 2 + eps)
    return (eigenvectors * shrink.unsqueeze(0)) @ eigenvectors.T


def affine_wiener_x1_hat(
    xt: torch.Tensor,
    t: float,
    wiener_operator: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Apply x1_hat = mu + W_t (x_t - t mu) to a batch of images."""
    original_shape = xt.shape
    xt_flat = xt.flatten(start_dim=1)
    mean = mean.to(device=xt.device, dtype=xt.dtype)
    centered_xt = xt_flat - t * mean.unsqueeze(0)
    x1_hat = mean.unsqueeze(0) + centered_xt @ wiener_operator.T
    return x1_hat.reshape(original_shape)


def cosine_similarity_mean(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean cosine similarity over a batch."""
    pred_flat = prediction.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    numerator = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.norm(dim=1) * target_flat.norm(dim=1)
    return (numerator / denominator.clamp(min=1e-12)).mean().item()


@torch.no_grad()
def evaluate_models(
    covariance_models: dict,
    x1_all: torch.Tensor,
    eps_all: torch.Tensor,
    model: nn.Module | None,
    t_values: list[float],
    xt_batch_size: int,
    sigma_min: float,
    device: torch.device,
) -> list[dict]:
    """Evaluate covariance models against true x1/velocity and optionally U-Net velocity."""
    mean = covariance_models["mean"].to(device)
    eigensystems = {
        "global_wiener": (
            covariance_models["global_eigenvalues"].to(device),
            covariance_models["global_eigenvectors"].to(device),
        ),
        "patch_wiener": (
            covariance_models["patch_eigenvalues"].to(device),
            covariance_models["patch_eigenvectors"].to(device),
        ),
        "hybrid_wiener_orthogonal": (
            covariance_models["hybrid_eigenvalues"].to(device),
            covariance_models["hybrid_eigenvectors"].to(device),
        ),
    }
    results: list[dict] = []
    true_velocity_all = x1_all - eps_all

    for t in t_values:
        print(f"evaluating t={t:.2f}")
        xt_all = t * x1_all + (1.0 - t) * eps_all
        unet_velocity_all = None
        if model is not None:
            unet_chunks = []
            for start in range(0, xt_all.shape[0], xt_batch_size):
                xt_batch = xt_all[start : start + xt_batch_size]
                unet_chunks.append(fm_unet_velocity_forward(model, xt_batch, t))
            unet_velocity_all = torch.cat(unet_chunks, dim=0)

        for kind, (eigenvalues, eigenvectors) in eigensystems.items():
            W_t = wiener_operator_from_eig(t, eigenvalues, eigenvectors)
            x1_hat_chunks = []
            velocity_chunks = []
            for start in range(0, xt_all.shape[0], xt_batch_size):
                xt_batch = xt_all[start : start + xt_batch_size]
                x1_hat = affine_wiener_x1_hat(xt_batch, t, W_t, mean)
                velocity = (x1_hat - xt_batch) / max(1.0 - t, sigma_min)
                x1_hat_chunks.append(x1_hat)
                velocity_chunks.append(velocity)
            x1_hat_all = torch.cat(x1_hat_chunks, dim=0)
            velocity_all = torch.cat(velocity_chunks, dim=0)

            row = {
                "model": kind,
                "t": float(t),
                "x1_mse": float(torch.mean((x1_hat_all - x1_all) ** 2).item()),
                "velocity_mse": float(torch.mean((velocity_all - true_velocity_all) ** 2).item()),
                "velocity_cosine": cosine_similarity_mean(velocity_all, true_velocity_all),
            }
            if unet_velocity_all is not None:
                row["unet_velocity_mse"] = float(torch.mean((velocity_all - unet_velocity_all) ** 2).item())
                row["unet_velocity_cosine"] = cosine_similarity_mean(velocity_all, unet_velocity_all)
            results.append(row)

    return results


def save_metrics(metrics: list[dict], out_dir: Path) -> None:
    """Save metrics JSON and compact plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    for metric_name, ylabel in [
        ("x1_mse", "x1 MSE"),
        ("velocity_mse", "velocity MSE"),
        ("velocity_cosine", "velocity cosine"),
        ("unet_velocity_mse", "U-Net velocity MSE"),
    ]:
        if metric_name not in metrics[0]:
            continue
        plt.figure(figsize=(7, 4.5))
        for kind in MODEL_KINDS:
            rows = [row for row in metrics if row["model"] == kind]
            plt.plot([row["t"] for row in rows], [row[metric_name] for row in rows], marker="o", label=kind)
        plt.xlabel("t")
        plt.ylabel(ylabel)
        if "mse" in metric_name:
            plt.yscale("log")
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_name}.png", dpi=180)
        plt.close()


def sensitivity_field(
    operator: torch.Tensor,
    y: int,
    x: int,
    channel: int,
    num_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Extract normalized same-channel sensitivity field from one Wiener row."""
    q = pixel_index(channel, y, x, height, width)
    row = operator[q].abs().reshape(num_channels, height, width)[channel]
    return row / row.max().clamp(min=1e-12)


def save_sensitivity_plot(
    covariance_models: dict,
    dataset,
    t_values: list[float],
    channel: int,
    out_dir: Path,
) -> None:
    """Plot sensitivity fields for global, patch, and hybrid Wiener operators."""
    eigensystems = {
        "global_wiener": (
            covariance_models["global_eigenvalues"],
            covariance_models["global_eigenvectors"],
        ),
        "patch_wiener": (
            covariance_models["patch_eigenvalues"],
            covariance_models["patch_eigenvectors"],
        ),
        "hybrid_wiener_orthogonal": (
            covariance_models["hybrid_eigenvalues"],
            covariance_models["hybrid_eigenvectors"],
        ),
    }
    target_pixels = default_target_pixels(dataset.resolution, dataset.resolution)
    num_rows = len(target_pixels) * len(MODEL_KINDS)
    num_cols = len(t_values)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(2.0 * num_cols, 2.0 * num_rows),
        squeeze=False,
    )

    for col, t in enumerate(t_values):
        axes[0, col].set_title(f"t={t:.2f}", fontsize=10)

    image = None
    for pixel_id, (y, x) in enumerate(target_pixels):
        for kind_id, kind in enumerate(MODEL_KINDS):
            row_id = pixel_id * len(MODEL_KINDS) + kind_id
            eigenvalues, eigenvectors = eigensystems[kind]
            W_t_cache: dict[float, torch.Tensor] = {}
            for col, t in enumerate(t_values):
                if t not in W_t_cache:
                    W_t_cache[t] = wiener_operator_from_eig(t, eigenvalues, eigenvectors)
                field = sensitivity_field(
                    operator=W_t_cache[t],
                    y=y,
                    x=x,
                    channel=channel,
                    num_channels=dataset.in_channels,
                    height=dataset.resolution,
                    width=dataset.resolution,
                ).detach().cpu()
                ax = axes[row_id, col]
                image = ax.imshow(field.sqrt(), cmap="turbo", vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.scatter([x], [y], s=8, c="red")
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(kind, fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    fig.savefig(out_dir / f"sensitivity_channel_{channel}.png", dpi=250)
    plt.close(fig)


@torch.no_grad()
def generate_wiener_samples(
    covariance_models: dict,
    dataset,
    num_samples: int,
    num_steps: int,
    sigma_min: float,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Generate samples with deterministic Euler for each Wiener covariance model."""
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(
        num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generator,
    )
    samples = {kind: initial_noise.clone() for kind in MODEL_KINDS}
    mean = covariance_models["mean"].to(device)
    eigensystems = {
        "global_wiener": (
            covariance_models["global_eigenvalues"].to(device),
            covariance_models["global_eigenvectors"].to(device),
        ),
        "patch_wiener": (
            covariance_models["patch_eigenvalues"].to(device),
            covariance_models["patch_eigenvectors"].to(device),
        ),
        "hybrid_wiener_orthogonal": (
            covariance_models["hybrid_eigenvalues"].to(device),
            covariance_models["hybrid_eigenvectors"].to(device),
        ),
    }

    step_size = 1.0 / num_steps
    for step in tqdm(range(num_steps), desc="Euler generation"):
        t = (step + 0.5) / num_steps
        denom = max(1.0 - t, sigma_min)
        for kind, (eigenvalues, eigenvectors) in eigensystems.items():
            W_t = wiener_operator_from_eig(t, eigenvalues, eigenvectors)
            x1_hat = affine_wiener_x1_hat(samples[kind], t, W_t, mean)
            velocity = (x1_hat - samples[kind]) / denom
            samples[kind] = samples[kind] + step_size * velocity

    return samples


def save_generation_grid(
    samples: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save generated images with rows=samples and columns=Wiener estimators."""
    num_samples = next(iter(samples.values())).shape[0]
    num_cols = len(MODEL_KINDS)
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(2.1 * num_cols, 2.1 * num_samples),
        squeeze=False,
    )

    for col, kind in enumerate(MODEL_KINDS):
        axes[0, col].set_title(kind, fontsize=14)
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
    """Run the hybrid Wiener covariance experiment."""
    args = parse_args()
    if args.dataset.lower().replace("-", "_") != "cifar10":
        raise ValueError("This dense prototype is currently intended for CIFAR-10 only.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = load_experiment_config(args.dataset, args.batch_size, subset_size=args.cov_subset_size)
    OmegaConf.set_readonly(cfg, False)
    cfg.dataset.num_workers = args.num_workers
    dataset = load_dataset(cfg)
    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)

    out_dir = (
        args.out_dir
        / f"patch_{args.patch_size:03d}_global_rank_{args.global_rank:04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"patch_size: {args.patch_size}")
    print(f"global_rank: {args.global_rank}")
    print(f"out_dir: {out_dir}")

    U, eigenvalues, _Vh, mean = load_or_compute_wiener_svd(dataset, device)
    covariance_models = build_or_load_covariance_models(
        dataset=dataset,
        U_full=U,
        eigenvalues_full=eigenvalues,
        mean=mean,
        args=args,
        device=device,
    )

    model = None
    if not args.skip_unet:
        model = load_fm_unet(checkpoint_path, device, args.dataset)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    t_values = parse_t_values(args.t_values)
    x1_all = collect_images(dataset, args.num_xt, device)
    eps_all = torch.randn(x1_all.shape, device=device, generator=generator)

    metrics = evaluate_models(
        covariance_models=covariance_models,
        x1_all=x1_all,
        eps_all=eps_all,
        model=model,
        t_values=t_values,
        xt_batch_size=args.xt_batch_size,
        sigma_min=args.sigma_min,
        device=device,
    )
    save_metrics(metrics, out_dir)
    save_sensitivity_plot(covariance_models, dataset, t_values, args.channel, out_dir)
    if not args.skip_generation:
        generated_samples = generate_wiener_samples(
            covariance_models=covariance_models,
            dataset=dataset,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            seed=args.seed,
            device=device,
        )
        save_generation_grid(
            samples=generated_samples,
            postprocess=dataset.postprocess,
            out_path=out_dir / "generated_samples.png",
        )

    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "patch_size": args.patch_size,
                "global_rank": args.global_rank,
                "cov_subset_size": args.cov_subset_size,
                "num_xt": args.num_xt,
                "num_samples": args.num_samples,
                "num_steps": args.num_steps,
                "t_values": t_values,
                "sigma_min": args.sigma_min,
                "cache_path": str(cache_path(args, dataset.name, dataset.resolution)),
                "diagnostics": covariance_models.get("diagnostics", {}),
            },
            handle,
            indent=2,
        )
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
