"""Train a CIFAR-10 U-Net with the paper-style AsymFlow loss.

This script reuses the Atong01 CIFAR-10 Flow Matching U-Net architecture, but
starts from random weights and changes the velocity parametrization:

    x_t = t x_1 + (1 - t) eps
    u_A_hat = G_theta(x_t, t)
    u_hat = P u_A_hat + (I - P) (u_A_hat - x_t) / (1 - t)
    target = x_1 - eps

where P is a low-rank PCA projection.  The loss is applied to the reconstructed
full velocity u_hat, matching the AsymFlow paper's training principle.

Inputs: CIFAR-10 and global or patch PCA basis. Output: checkpoints and logs.
Example: python -m asymflow.train_asymflow_unet_cifar10 --rank 256
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALITY_ROOT = PROJECT_ROOT
LOCALITY_SRC = LOCALITY_ROOT / "src"


def parse_args() -> argparse.Namespace:
    """Parse command-line hyperparameters and return them as a Namespace."""
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 FM U-Net with an AsymFlow low-rank target."
    )
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument(
        "--projection-mode",
        choices=["global", "patchwise"],
        default="global",
        help="Use one global PCA projection or a shared PCA projection on image patches.",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument(
        "--patch-pca-dir",
        type=Path,
        default=Path("asymflow/results/patchwise_pca_cifar10"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=400_001)
    parser.add_argument("--save-step", type=int, default=20_000)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-min", type=float, default=5e-2)
    parser.add_argument(
        "--time-sampler",
        choices=["asymflow_logit_normal", "uniform"],
        default="asymflow_logit_normal",
    )
    parser.add_argument("--logit-normal-mean", type=float, default=0.8)
    parser.add_argument("--logit-normal-std", type=float, default=0.8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("asymflow/results/asymflow_unet_cifar10_paper_loss"),
    )
    return parser.parse_args()


def ensure_torchcfm_importable() -> None:
    """Require the separately installed Atong01 torchcfm package."""
    try:
        import torchcfm  # noqa: F401
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "torchcfm is required; install requirements-fm.txt first."
        ) from error


def ensure_locality_importable() -> None:
    """Make the locality repo source package importable in this repo."""
    if not LOCALITY_SRC.exists():
        raise FileNotFoundError(f"Could not find locality source directory: {LOCALITY_SRC}")
    sys.path.insert(0, str(LOCALITY_SRC))


@contextmanager
def working_directory(path: Path):
    """Temporarily run code from a different working directory."""
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def build_cifar10_unet(device: torch.device) -> nn.Module:
    """Build the Atong01 CIFAR-10 U-Net from random initialization."""
    ensure_torchcfm_importable()
    from torchcfm.models.unet.unet import UNetModelWrapper

    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    )
    return model.to(device)


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


def build_train_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    """Build a shuffled training DataLoader from the locality DatasetBundle."""
    return DataLoader(
        dataset.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def load_pca_basis(rank: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the first rank CIFAR-10 PCA directions and singular values."""
    svd_dir = LOCALITY_ROOT / "data" / "models" / "wiener" / "cifar10_32"
    U_path = svd_dir / "U.pt"
    LA_path = svd_dir / "LA.pt"
    if not U_path.exists() or not LA_path.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-10 Wiener SVD at {svd_dir}. "
            "Run the locality Wiener preprocessing first."
        )

    U = torch.load(U_path, map_location=device, weights_only=True)
    singular_values = torch.load(LA_path, map_location=device, weights_only=True)
    if rank < 0 or rank > U.shape[1]:
        raise ValueError(f"rank must be in [0, {U.shape[1]}], got {rank}.")
    return U[:, :rank].contiguous(), singular_values


def project_onto_basis(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project image batches onto span(basis) without forming P = basis @ basis.T."""
    if basis.shape[1] == 0:
        return torch.zeros_like(x)
    original_shape = x.shape
    flat = x.flatten(start_dim=1)  # [B, N]
    projected = (flat @ basis) @ basis.T  # [B, r] @ [r, N] -> [B, N]
    return projected.reshape(original_shape)


def load_patchwise_pca_basis(
    rank: int,
    patch_size: int,
    pca_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the first rank shared PCA directions for flattened image patches."""
    patch_dir = pca_dir / f"patch_size_{patch_size:03d}"
    basis_path = patch_dir / "basis_full.pt"
    eigenvalues_path = patch_dir / "eigenvalues.pt"
    if not basis_path.exists() or not eigenvalues_path.exists():
        raise FileNotFoundError(
            f"Missing patch-wise PCA files in {patch_dir}. "
            "Run compute_patchwise_pca_cifar10.py first."
        )

    full_basis = torch.load(basis_path, map_location=device, weights_only=True)
    eigenvalues = torch.load(eigenvalues_path, map_location=device, weights_only=True)
    if rank < 0 or rank > full_basis.shape[1]:
        raise ValueError(f"rank must be in [0, {full_basis.shape[1]}], got {rank}.")
    return full_basis[:, :rank].contiguous(), eigenvalues


def patchify_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B, C, H, W] images into [B, M, C * patch_size * patch_size] patches."""
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Image shape {(height, width)} is not divisible by patch_size={patch_size}.")

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.reshape(batch, -1, channels * patch_size * patch_size)


def unpatchify_images(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert [B, M, C * patch_size * patch_size] patches back to [B, C, H, W]."""
    batch = patches.shape[0]
    patches_y = height // patch_size
    patches_x = width // patch_size
    images = patches.reshape(batch, patches_y, patches_x, channels, patch_size, patch_size)
    images = images.permute(0, 3, 1, 4, 2, 5).contiguous()
    return images.reshape(batch, channels, height, width)


def project_patchwise_onto_basis(
    x: torch.Tensor,
    basis: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Project each image patch onto a shared PCA subspace."""
    if basis.shape[1] == 0:
        return torch.zeros_like(x)

    batch, channels, height, width = x.shape
    patches = patchify_images(x, patch_size)  # [B, M, Dp]
    projected = (patches @ basis) @ basis.T  # [B, M, r] @ [r, Dp] -> [B, M, Dp]
    return unpatchify_images(projected, patch_size, channels, height, width)


def project_low_rank(
    x: torch.Tensor,
    basis: torch.Tensor,
    projection_mode: str,
    patch_size: int,
) -> torch.Tensor:
    """Apply the chosen low-rank projection to an image batch."""
    if projection_mode == "global":
        return project_onto_basis(x, basis)
    if projection_mode == "patchwise":
        return project_patchwise_onto_basis(x, basis, patch_size)
    raise ValueError(f"Unknown projection_mode: {projection_mode}")


def reconstruct_full_velocity(
    asymflow_output: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    basis: torch.Tensor,
    projection_mode: str,
    patch_size: int,
    sigma_min: float,
) -> torch.Tensor:
    """Convert predicted u_A into the full velocity used in the FM loss.

    With our convention x_t = t x_1 + (1 - t) eps, the noise coefficient is
    sigma_t = 1 - t.  This is the convention-flipped version of Eq. (5) in the
    AsymFlow paper.  The projection P can be global or shared patch-wise.
    """
    sigma = (1.0 - t).reshape(-1, 1, 1, 1).clamp(min=sigma_min)
    projected_output = project_low_rank(asymflow_output, basis, projection_mode, patch_size)
    complement_argument = (asymflow_output - xt) / sigma
    projected_complement_argument = project_low_rank(
        complement_argument, basis, projection_mode, patch_size
    )
    complement_velocity = complement_argument - projected_complement_argument
    return projected_output + complement_velocity


def sample_fm_times(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sampler: str,
    logit_normal_mean: float,
    logit_normal_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample FM times in our convention x_t = t x_1 + (1 - t) eps."""
    if sampler == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)

    if sampler != "asymflow_logit_normal":
        raise ValueError(f"Unknown time sampler: {sampler}")

    # The AsymFlow paper samples the noise coefficient sigma from a logit-normal.
    # Our convention uses t as the data coefficient, so t = 1 - sigma.
    sigma = torch.sigmoid(
        logit_normal_mean
        + logit_normal_std
        * torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    )
    return 1.0 - sigma


def make_asymflow_batch(
    x1: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one FM training batch and its full-velocity target.

    Returns
    -------
    xt:
        Interpolated images, shape [B, C, H, W].
    t:
        FM times, shape [B].
    target:
        Full Flow Matching velocity target x1 - eps, shape [B, C, H, W].
    """
    eps = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = sample_fm_times(
        batch_size=x1.shape[0],
        device=x1.device,
        dtype=x1.dtype,
        sampler=args.time_sampler,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        generator=generator,
    )
    t_view = t.reshape(-1, 1, 1, 1)

    xt = t_view * x1 + (1.0 - t_view) * eps
    target = x1 - eps
    return xt, t, target


def clone_ema_model(model: nn.Module) -> nn.Module:
    """Create a frozen EMA copy of the model."""
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update EMA parameters with ema <- decay * ema + (1 - decay) * model."""
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Return a run directory that encodes the main screening hyperparameters."""
    if args.projection_mode == "patchwise":
        return (
            args.out_dir
            / f"patch_{args.patch_size:02d}"
            / f"rank_{args.rank:04d}_steps_{args.total_steps:06d}"
        )
    return args.out_dir / f"global_rank_{args.rank:04d}_steps_{args.total_steps:06d}"


def json_ready_args(args: argparse.Namespace) -> dict:
    """Convert argparse values to JSON-friendly values."""
    output = vars(args).copy()
    for key, value in output.items():
        if isinstance(value, Path):
            output[key] = str(value)
    return output


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    args: argparse.Namespace,
) -> None:
    """Save model, EMA, optimizer, scheduler, and config state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": json_ready_args(args),
        },
        path,
    )


def save_losses(run_dir: Path, loss_history: list[dict]) -> None:
    """Save the scalar loss history as JSON and a PNG curve."""
    metrics_path = run_dir / "losses.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(loss_history, handle, indent=2)

    if not loss_history:
        return

    steps = [item["step"] for item in loss_history]
    losses = [item["loss"] for item in loss_history]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("AsymFlow CIFAR-10 training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss.png", dpi=160)
    plt.close()


def run_training(
    model: nn.Module,
    ema_model: nn.Module,
    dataloader: DataLoader,
    basis: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> list[dict]:
    """Run training with loss on the reconstructed full velocity."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, float(step + 1) / float(max(args.warmup_steps, 1))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    use_amp = device.type == "cuda" and not args.no_amp

    start_step = 0
    loss_history: list[dict] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])

    model.train()
    data_iter = iter(dataloader)
    progress = trange(start_step + 1, args.total_steps + 1, initial=start_step, total=args.total_steps)
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x1 = batch[0] if isinstance(batch, (tuple, list)) else batch
        x1 = x1.to(device, non_blocking=True)
        xt, t, target = make_asymflow_batch(x1, args)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            asymflow_output = model(t, xt)
            prediction = reconstruct_full_velocity(
                asymflow_output=asymflow_output,
                xt=xt,
                t=t,
                basis=basis,
                projection_mode=args.projection_mode,
                patch_size=args.patch_size,
                sigma_min=args.sigma_min,
            )
            loss = F.mse_loss(prediction, target)

        scaler.scale(loss).backward()
        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema_model(ema_model, model, args.ema_decay)

        if step % args.log_step == 0 or step == 1:
            loss_value = float(loss.detach().cpu())
            loss_history.append({"step": step, "loss": loss_value})
            progress.set_postfix(loss=f"{loss_value:.4f}")
            save_losses(run_dir, loss_history)

        if step % args.save_step == 0 or step == args.total_steps:
            save_checkpoint(
                run_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                args=args,
            )
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                args=args,
            )

    return loss_history


def main() -> None:
    """Train the CIFAR-10 U-Net with the paper-style AsymFlow loss."""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cifar10_unet(device)
    ema_model = clone_ema_model(model)
    dataset = load_cifar10_dataset(args.batch_size)
    dataloader = build_train_dataloader(dataset, args.batch_size, args.num_workers)
    if args.projection_mode == "patchwise":
        basis, spectrum = load_patchwise_pca_basis(
            rank=args.rank,
            patch_size=args.patch_size,
            pca_dir=args.patch_pca_dir,
            device=device,
        )
    else:
        basis, spectrum = load_pca_basis(args.rank, device)
    explained_variance = spectrum[: args.rank].sum() / spectrum.sum()
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready_args(args), handle, indent=2)

    print("dataset: cifar10")
    print(f"device: {device}")
    print(f"projection_mode: {args.projection_mode}")
    print(f"rank: {args.rank}")
    if args.projection_mode == "patchwise":
        print(f"patch_size: {args.patch_size}")
        print(f"patch_pca_dir: {args.patch_pca_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"grad_clip: {args.grad_clip}")
    print(f"total_steps: {args.total_steps}")
    print(f"save_step: {args.save_step}")
    print(f"sigma_min: {args.sigma_min}")
    print(f"time_sampler: {args.time_sampler}")
    print(f"logit_normal_mean: {args.logit_normal_mean}")
    print(f"logit_normal_std: {args.logit_normal_std}")
    print(f"num_parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"resolution: {dataset.resolution}")
    print(f"channels: {dataset.in_channels}")
    print(f"num_images: {len(dataset.dataset)}")
    print(f"num_batches: {len(dataloader)}")
    print(f"basis_shape: {tuple(basis.shape)}")
    print(f"explained_variance_at_rank: {explained_variance.item():.4f}")
    print(f"run_dir: {run_dir}")

    run_training(
        model=model,
        ema_model=ema_model,
        dataloader=dataloader,
        basis=basis,
        args=args,
        device=device,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
