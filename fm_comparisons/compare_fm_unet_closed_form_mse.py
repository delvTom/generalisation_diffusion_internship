"""Compare FM U-Net velocity with empirical and affine velocity estimators.

This script compares a pretrained Flow Matching U-Net on CIFAR-10 against
three closed-form velocity estimators:

    x1_mask:       reconstruct x1 with the clean-image Wiener mask, then convert
                   to velocity.
    velocity_mask: reconstruct x1 with a mask built from the velocity sensitivity,
                   then convert to velocity.
    wiener_velocity: use the optimal affine Wiener denoiser, including its
                     dataset-mean bias, then convert it to velocity.

Inputs: CIFAR-10 or CelebA-HQ via the upstream dataset config, plus a matching
FM U-Net checkpoint. Outputs: time-indexed teacher MSE tables and plots in
``--out-dir``. Example (after setting up the external inputs)::

    python -m fm_comparisons.compare_fm_unet_closed_form_mse --dataset cifar10 --checkpoint path/to/fm_unet.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchcfm.models.unet.unet import UNetModelWrapper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from omegaconf import DictConfig

from local_diffusion.configuration import load_config
from local_diffusion.data import build_dataset
from local_diffusion.data.datasets import DatasetBundle
from local_diffusion.utils.wiener import (
    compute_wiener_filter,
    load_wiener_filter,
    save_wiener_filter,
)
from fm_comparisons.streaming_softmax import WeightedStreamingSoftmax


DEFAULT_CHECKPOINT = Path(
    "data/models/fm_unet/cifar10/fm_cifar10_weights_step_400000.pt"
)
DEFAULT_CELEBA_HQ_CHECKPOINT = Path(
    "data/models/fm_unet/celeba_hq/fm_celeba_hq_weights.pt"
)
MASK_KINDS = ("x1_mask", "velocity_mask")
ESTIMATOR_KINDS = (*MASK_KINDS, "wiener_velocity")


@dataclass(frozen=True)
class PixelCandidateBank:
    """Flattened candidate images cached once on the compute device."""

    images_flat: torch.Tensor  # [K, N]
    chunk_size: int

    def chunks(self):
        for start in range(0, self.images_flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.images_flat.shape[0])
            yield self.images_flat[start:end]

    @property
    def storage_megabytes(self) -> float:
        return (
            self.images_flat.numel() * self.images_flat.element_size() / (1024.0**2)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--xt-batch-size", type=int, default=None)
    parser.add_argument("--num-xt", type=int, default=128)
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--query-split", choices=("train", "test"), default="test")
    parser.add_argument("--candidate-split", choices=("train", "test"), default="train")
    parser.add_argument("--t-values", default="0.05,0.15,0.25,0.35,0.45,0.5,0.55,0.65,0.75,0.85,0.95")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_experiment_config(
    dataset_name: str,
    batch_size: int | None,
    subset_size: int | None,
    split: str | None = None,
    num_workers: int | None = None,
) -> DictConfig:
    """Load Artem's PCA-locality config with optional batch-size override."""
    overrides = ["metrics.wandb.enabled=false"]
    if batch_size is not None:
        overrides.append(f"dataset.batch_size={batch_size}")
    if subset_size is not None:
        overrides.append(f"dataset.subset_size={subset_size}")
    if split is not None:
        overrides.append(f"dataset.split={split}")
    if num_workers is not None:
        overrides.append(f"dataset.num_workers={num_workers}")
    return load_config(f"configs/pca_locality/{dataset_name}.yaml", overrides=overrides)


def parse_t_values(value: str) -> list[float]:
    """Parse comma-separated FM times in the open interval (0, 1)."""
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if any(t <= 0.0 or t >= 1.0 for t in values):
        raise ValueError("All t values must be strictly between 0 and 1.")
    return values


def dataset_output_slug(dataset_name: str) -> str:
    """Return a readable dataset name for experiment output directories."""
    normalized = dataset_name.lower().replace("-", "_")
    return "cifar_10" if normalized == "cifar10" else normalized


def default_checkpoint_for_dataset(dataset_name: str) -> Path:
    """Return the default pretrained FM U-Net checkpoint for a dataset."""
    normalized = dataset_name.lower().replace("-", "_")
    if normalized == "cifar10":
        return DEFAULT_CHECKPOINT
    if normalized == "celeba_hq":
        return DEFAULT_CELEBA_HQ_CHECKPOINT
    raise ValueError(f"No default FM U-Net checkpoint is known for dataset: {dataset_name}")


def load_dataset(cfg: DictConfig) -> DatasetBundle:
    """Load a dataset through Artem's config and dataset registry."""
    return build_dataset(cfg.dataset)


def collect_images(dataset: DatasetBundle, num_images: int, device: torch.device) -> torch.Tensor:
    """Collect the first num_images images from a DatasetBundle."""
    chunks: list[torch.Tensor] = []
    total = 0
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(images)
        total += images.shape[0]
        if total >= num_images:
            break

    if not chunks:
        raise RuntimeError("Dataset dataloader did not yield any images.")

    return torch.cat(chunks, dim=0)[:num_images].to(device)


def collect_random_images(
    dataset: DatasetBundle,
    num_images: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Collect a deterministic random subset and return its dataset indices."""
    dataset_size = len(dataset.dataset)
    if not 0 < num_images <= dataset_size:
        raise ValueError(f"num_images must lie in [1, {dataset_size}], got {num_images}.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)[:num_images].tolist()
    images = []
    for index in indices:
        sample = dataset.dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        images.append(image)
    return torch.stack(images, dim=0).to(device), indices


@torch.no_grad()
def build_pixel_candidate_bank(
    candidate_dataset: DatasetBundle,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PixelCandidateBank:
    """Load each empirical candidate once and cache it on the compute device."""
    chunks: list[torch.Tensor] = []
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        chunks.append(images.to(device=device, dtype=dtype).flatten(start_dim=1))
    if not chunks:
        raise RuntimeError("The candidate dataset is empty.")
    return PixelCandidateBank(
        images_flat=torch.cat(chunks, dim=0),
        chunk_size=int(candidate_dataset.dataloader.batch_size),
    )


def load_fm_unet(
    checkpoint_path: Path,
    device: torch.device,
    dataset_name: str,
) -> nn.Module:
    """Build a pretrained FM U-Net for the requested dataset."""
    normalized = dataset_name.lower().replace("-", "_")
    if normalized == "cifar10":
        model = UNetModelWrapper(
            dim=(3, 32, 32),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["ema_model"]
        model._fm_forward_order = "t_x"
    elif normalized == "celeba_hq":
        from fm_comparisons.fm_unet_celeba64 import create_unet_celeba64

        model = create_unet_celeba64().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["ema_model_state"]
        model._fm_forward_order = "x_t"
    else:
        raise ValueError(f"Unsupported FM U-Net dataset: {dataset_name}")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_or_compute_wiener_svd(
    dataset: DatasetBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Artem's cached Wiener SVD, or compute it from the dataset."""
    wiener_path = Path("data/models/wiener") / f"{dataset.name}_{dataset.resolution}"
    if (wiener_path / "U.pt").exists():
        return load_wiener_filter(wiener_path, device=device)

    covariance, mean = compute_wiener_filter(
        dataloader=dataset.dataloader,
        device=device,
        resolution=dataset.resolution,
        n_channels=dataset.in_channels,
    )
    U, eigenvalues, Vh = torch.linalg.svd(covariance)
    save_wiener_filter(U, eigenvalues, Vh, mean, wiener_path)
    return U, eigenvalues, Vh, mean


def build_fm_wiener_operator(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Build the clean-image Wiener operator for x_t = t x_1 + (1 - t) eps."""
    shrink = t * eigenvalues / (t * t * eigenvalues + (1.0 - t) ** 2)
    return (U * shrink.unsqueeze(0)) @ Vh


def build_mask_operator(wiener_operator: torch.Tensor, t: float, mask_kind: str) -> torch.Tensor:
    """Select the operator whose sensitivity field defines the mask."""
    if mask_kind == "x1_mask":
        return wiener_operator
    if mask_kind == "velocity_mask":
        identity = torch.eye(
            wiener_operator.shape[0],
            device=wiener_operator.device,
            dtype=wiener_operator.dtype,
        )
        return (wiener_operator - identity) / (1.0 - t)
    raise ValueError(f"Unknown mask kind: {mask_kind}")


def binarize_like_artem(
    operator: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize rows by the diagonal, then apply Artem's global threshold."""
    diagonal = torch.diagonal(operator).unsqueeze(1)
    diagonal = torch.where(diagonal.abs() < eps, torch.ones_like(diagonal), diagonal)
    normalized = operator / diagonal

    if threshold > 0:
        cutoff = normalized.abs().max() * threshold
        normalized = (normalized.abs() >= cutoff).float()

    return normalized.transpose(0, 1).contiguous()


@torch.no_grad()
def predict_closed_form_x1(
    xt: torch.Tensor,
    t: float,
    candidate_bank: PixelCandidateBank,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Predict x1 with Artem's dense streaming-softmax estimator in FM convention."""
    xt_flat = xt.flatten(start_dim=1)  # [B, N]
    t_tensor = torch.tensor(t, device=xt.device, dtype=xt.dtype)
    noise_var = (1.0 - t_tensor) ** 2
    first_moment = WeightedStreamingSoftmax(device=xt.device, dtype=xt.dtype)
    for candidates in candidate_bank.chunks():
        delta = (
            xt_flat.unsqueeze(1) - t_tensor * candidates.unsqueeze(0)
        ) ** 2  # [B, K, N]
        if mask.layout == torch.strided:
            masked_dist = torch.einsum("bkn,nm->bkm", delta, mask)
        else:
            batch_size, num_candidates, num_inputs = delta.shape
            flattened = delta.reshape(batch_size * num_candidates, num_inputs)
            masked_dist = torch.sparse.mm(mask.transpose(0, 1), flattened.T).T
            masked_dist = masked_dist.reshape(batch_size, num_candidates, -1)
        logits = -masked_dist / (2.0 * noise_var * temperature)
        first_moment.add(candidates, logits)

    prediction = first_moment.get_average()
    if prediction is None:
        raise RuntimeError("No candidate images were available for closed-form prediction.")
    return prediction.view_as(xt)


def fm_unet_velocity_forward(model: nn.Module, xt: torch.Tensor, t: float) -> torch.Tensor:
    """Forward the pretrained FM U-Net while preserving autograd if enabled."""
    t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=xt.dtype)  # [B]
    forward_order = getattr(model, "_fm_forward_order", "t_x")
    if forward_order == "t_x":
        return model(t_batch, xt)  # [B, C, H, W]
    if forward_order == "x_t":
        return model(xt, t_batch)  # [B, C, H, W]
    raise ValueError(f"Unknown FM U-Net forward order: {forward_order}")


@torch.no_grad()
def predict_fm_unet_velocity(model: nn.Module, xt: torch.Tensor, t: float) -> torch.Tensor:
    """Predict velocity with the pretrained FM U-Net without tracking gradients."""
    return fm_unet_velocity_forward(model, xt, t)


@torch.no_grad()
def predict_wiener_velocity(
    xt: torch.Tensor,
    t: float,
    wiener_operator: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Predict velocity with the affine Wiener denoiser, including its bias.

    In column-vector notation:
        x1_hat = W_t x_t + (I - t W_t) mu,
        u_t     = (x1_hat - x_t) / (1 - t).
    """
    xt_flat = xt.flatten(start_dim=1)  # [B, N]
    mean = mean.to(device=xt.device, dtype=xt.dtype)  # [N]

    # PyTorch stores each flattened image as a row, hence multiplication by W_t^T.
    mean_bias = mean - t * (mean.unsqueeze(0) @ wiener_operator.T).squeeze(0)  # [N]
    x1_hat = xt_flat @ wiener_operator.T + mean_bias.unsqueeze(0)  # [B, N]
    velocity = (x1_hat - xt_flat) / (1.0 - t)  # [B, N]
    return velocity.view_as(xt)


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return scalar mean squared error as a Python float."""
    return torch.mean((prediction - target) ** 2).item()


def run_mse_experiment(
    model: nn.Module,
    xt_dataset: DatasetBundle,
    candidate_bank: PixelCandidateBank,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    t_values: list[float],
    num_xt: int,
    xt_batch_size: int,
    temperature: float,
    mask_threshold: float,
    seed: int,
    device: torch.device,
) -> tuple[
    list[dict[str, float | str | int]],
    list[dict[str, float | int]],
    list[int],
]:
    """Compute velocity MSE between the FM U-Net and each closed-form estimator."""
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    x1_all, query_indices = collect_random_images(
        xt_dataset,
        num_xt,
        seed,
        device,
    )
    x0_all = torch.randn(x1_all.shape, device=device, generator=generator)
    results: list[dict[str, float | str | int]] = []
    paired_comparisons: list[dict[str, float | int]] = []

    for t in t_values:
        print(f"\n=== t={t:.2f} ===")
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
        masks = {
            mask_kind: binarize_like_artem(
                build_mask_operator(wiener_operator, t, mask_kind),
                mask_threshold,
            ).to_sparse_coo().coalesce()
            for mask_kind in MASK_KINDS
        }

        per_query_mse = {estimator: [] for estimator in ESTIMATOR_KINDS}
        for start in range(0, num_xt, xt_batch_size):
            end = min(start + xt_batch_size, num_xt)
            x1 = x1_all[start:end]
            x0 = x0_all[start:end]
            xt = (1.0 - t) * x0 + t * x1  # [B, C, H, W]
            unet_velocity = predict_fm_unet_velocity(model, xt, t)

            for mask_kind in MASK_KINDS:
                closed_form_x1 = predict_closed_form_x1(
                    xt=xt,
                    t=t,
                    candidate_bank=candidate_bank,
                    mask=masks[mask_kind],
                    temperature=temperature,
                )
                closed_form_velocity = (closed_form_x1 - xt) / (1.0 - t)
                errors = (
                    (closed_form_velocity - unet_velocity)
                    .square()
                    .flatten(start_dim=1)
                    .mean(dim=1)
                )
                per_query_mse[mask_kind].extend(errors.detach().cpu().tolist())

            wiener_velocity = predict_wiener_velocity(
                xt=xt,
                t=t,
                wiener_operator=wiener_operator,
                mean=mean,
            )
            errors = (
                (wiener_velocity - unet_velocity)
                .square()
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            per_query_mse["wiener_velocity"].extend(errors.detach().cpu().tolist())

        for estimator in ESTIMATOR_KINDS:
            values = torch.tensor(per_query_mse[estimator], dtype=torch.float64)
            mse = float(values.mean().item())
            standard_error = (
                float(values.std(unbiased=True).item() / math.sqrt(values.numel()))
                if values.numel() > 1
                else 0.0
            )
            row = {
                "t": t,
                "estimator": estimator,
                "mse_velocity_vs_fm_unet": mse,
                "standard_error": standard_error,
                "ci95_low": max(0.0, mse - 1.96 * standard_error),
                "ci95_high": mse + 1.96 * standard_error,
                "num_queries": values.numel(),
            }
            print(row)
            results.append(row)

        denoiser_values = torch.tensor(per_query_mse["x1_mask"], dtype=torch.float64)
        velocity_values = torch.tensor(per_query_mse["velocity_mask"], dtype=torch.float64)
        paired_delta = velocity_values - denoiser_values
        delta_mean = float(paired_delta.mean().item())
        delta_se = (
            float(paired_delta.std(unbiased=True).item() / math.sqrt(paired_delta.numel()))
            if paired_delta.numel() > 1
            else 0.0
        )
        paired_comparisons.append(
            {
                "t": t,
                "mse_velocity_mask_minus_denoiser_mask": delta_mean,
                "standard_error": delta_se,
                "ci95_low": delta_mean - 1.96 * delta_se,
                "ci95_high": delta_mean + 1.96 * delta_se,
                "relative_gain_velocity_mask": float(
                    ((denoiser_values.mean() - velocity_values.mean()) / denoiser_values.mean()).item()
                ),
                "num_queries": paired_delta.numel(),
            }
        )

    return results, paired_comparisons, query_indices


def save_metrics(results: list[dict[str, float | str | int]], out_path: Path) -> None:
    """Save metrics as a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_csv(rows: list[dict], out_path: Path) -> None:
    """Save homogeneous metric rows as a CSV table."""
    if not rows:
        raise ValueError("Cannot save an empty CSV table.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_mask_statistics(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    t_values: list[float],
    mask_threshold: float,
) -> list[dict[str, float | str]]:
    """Summarize how many input coordinates each sensitivity mask exposes."""
    rows: list[dict[str, float | str]] = []
    for t in t_values:
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
        for mask_kind in MASK_KINDS:
            mask = binarize_like_artem(
                build_mask_operator(wiener_operator, t, mask_kind),
                mask_threshold,
            )
            visible_per_output = mask.sum(dim=0)
            rows.append(
                {
                    "t": t,
                    "mask": mask_kind,
                    "mean_visible_inputs": float(visible_per_output.mean().item()),
                    "min_visible_inputs": float(visible_per_output.min().item()),
                    "max_visible_inputs": float(visible_per_output.max().item()),
                    "density": float(mask.mean().item()),
                }
            )
    return rows


def plot_mse(
    results: list[dict[str, float | str | int]],
    out_path: Path,
    estimators: tuple[str, ...] = ESTIMATOR_KINDS,
) -> None:
    """Plot velocity MSE curves for all closed-form estimators."""
    t_values = sorted({float(row["t"]) for row in results})
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    labels = {
        "x1_mask": "Masque du débruiteur affine",
        "velocity_mask": "Masque de la vitesse affine",
        "wiener_velocity": "Vitesse de Wiener affine",
    }
    for estimator in estimators:
        rows = {
            float(row["t"]): row
            for row in results
            if row["estimator"] == estimator
        }
        mse_values = [float(rows[t]["mse_velocity_vs_fm_unet"]) for t in t_values]
        ci_low = [float(rows[t]["ci95_low"]) for t in t_values]
        ci_high = [float(rows[t]["ci95_high"]) for t in t_values]
        line = ax.plot(
            t_values,
            mse_values,
            marker="o",
            linewidth=2.0,
            label=labels[estimator],
        )[0]
        ax.fill_between(t_values, ci_low, ci_high, color=line.get_color(), alpha=0.14)

    ax.set_title("Approximation du champ de vitesse du U-Net")
    ax.set_xlabel("Temps $t$")
    ax.set_ylabel("EQM de vitesse par pixel")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"fm_mask_comparison_corrected_{dataset_output_slug(args.dataset)}"
    )
    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)

    print(f"dataset: {args.dataset}")
    print(f"device: {device}")

    query_cfg = load_experiment_config(
        args.dataset,
        args.batch_size,
        subset_size=None,
        split=args.query_split,
        num_workers=args.num_workers,
    )
    candidate_cfg = load_experiment_config(
        args.dataset,
        args.batch_size,
        subset_size=args.num_xi,
        split=args.candidate_split,
        num_workers=args.num_workers,
    )
    pca_params = candidate_cfg.model.params or {}
    t_values = parse_t_values(args.t_values)
    temperature = float(pca_params.get("temperature", 1.0))
    mask_threshold = float(pca_params.get("mask_threshold", 0.02))
    xt_batch_size = args.xt_batch_size or int(query_cfg.sampling.batch_size)

    print(f"candidate_batch_size: {candidate_cfg.dataset.batch_size}")
    print(f"query_loader_batch_size: {query_cfg.dataset.batch_size}")
    print(f"xt_batch_size: {xt_batch_size}")
    print(f"num_xt: {args.num_xt}")
    print(f"num_xi: {args.num_xi}")
    print(f"query_split: {args.query_split}")
    print(f"candidate_split: {args.candidate_split}")
    print(f"t_values: {t_values}")
    print(f"temperature: {temperature}")
    print(f"mask_threshold: {mask_threshold}")

    dataset = load_dataset(query_cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    print(f"loaded dataset: {dataset.name}")
    print(f"resolution: {dataset.resolution}")
    print(f"channels: {dataset.in_channels}")
    print(f"num batches: {len(dataset.dataloader)}")

    model = load_fm_unet(checkpoint_path, device, args.dataset)
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(candidate_dataset, device)
    candidate_bank = build_pixel_candidate_bank(candidate_dataset, device)
    print("loaded FM U-Net and Wiener SVD")
    print(
        f"pixel candidate cache: {candidate_bank.images_flat.shape[0]} images, "
        f"{candidate_bank.storage_megabytes:.1f} MiB"
    )

    results, paired_comparisons, query_indices = run_mse_experiment(
        model=model,
        xt_dataset=dataset,
        candidate_bank=candidate_bank,
        U=U,
        eigenvalues=eigenvalues,
        Vh=Vh,
        mean=mean,
        t_values=t_values,
        num_xt=args.num_xt,
        xt_batch_size=xt_batch_size,
        temperature=temperature,
        mask_threshold=mask_threshold,
        seed=args.seed,
        device=device,
    )

    save_metrics(results, out_dir / "metrics.json")
    save_csv(results, out_dir / "metrics.csv")
    save_metrics(paired_comparisons, out_dir / "paired_mask_comparison.json")
    save_csv(paired_comparisons, out_dir / "paired_mask_comparison.csv")
    mask_statistics = compute_mask_statistics(
        U,
        eigenvalues,
        Vh,
        t_values,
        mask_threshold,
    )
    save_metrics(mask_statistics, out_dir / "mask_statistics.json")
    with (out_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "query_split": args.query_split,
                "candidate_split": args.candidate_split,
                "num_queries": args.num_xt,
                "num_candidates": len(candidate_dataset.dataset),
                "candidate_limit_argument": args.num_xi,
                "candidate_cache_megabytes": candidate_bank.storage_megabytes,
                "candidate_batch_size": int(candidate_cfg.dataset.batch_size),
                "query_batch_size": xt_batch_size,
                "num_workers": int(candidate_cfg.dataset.num_workers),
                "times": t_values,
                "temperature": temperature,
                "mask_threshold": mask_threshold,
                "seed": args.seed,
                "noise_seed": args.seed + 1,
                "query_dataset_indices": query_indices,
                "path": "x_t=(1-t)epsilon+t*x_1",
                "teacher": str(checkpoint_path),
                "metric": "mean squared velocity error over queries, channels, and pixels",
            },
            handle,
            indent=2,
        )
    plot_mse(results, out_dir / "mse.png")
    plot_mse(results, out_dir / "mse.pdf")
    plot_mse(results, out_dir / "mse_masks.png", estimators=MASK_KINDS)
    plot_mse(results, out_dir / "mse_masks.pdf", estimators=MASK_KINDS)
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
