"""Screen CIFAR-10 AsymFlow ranks with short training runs and FID.

For each rank in a grid, this script:

1. trains the AsymFlow U-Net for a reduced number of steps;
2. loads the EMA checkpoint;
3. generates samples in batches with the reconstructed full velocity;
4. computes FID against real CIFAR-10 images;
5. saves a JSON table and an FID-vs-rank plot.

This is meant as the first-stage rank calibration before spending full
400k-step training runs on the best ranks.

Input: CIFAR-10 and matching PCA basis. Output: checkpoints, FID JSON, plot.
Example: python -m asymflow.sweep_asymflow_rank_cifar10 --projection-mode global
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from asymflow.generate_asymflow_unet_cifar10 import (
    euler_generate,
    load_asymflow_model,
    save_image_grid,
)
from asymflow.train_asymflow_unet_cifar10 import (
    load_cifar10_dataset,
    load_patchwise_pca_basis,
    load_pca_basis,
)


DEFAULT_RANKS = "0,8,16,32,64,128,256"
DEFAULT_OUT_DIR = Path("asymflow/results/asymflow_rank_sweep_cifar10_paper_loss")
DEFAULT_TRAIN_OUT_ROOT = Path("asymflow/results/asymflow_unet_cifar10_paper_loss")
DEFAULT_PATCHWISE_OUT_ROOT = Path("asymflow/results/asymflow_patchwise_rank_sweep_cifar10")
DEFAULT_PATCHWISE_TRAIN_ROOT = Path("asymflow/results/asymflow_patchwise_unet_cifar10_paper_loss")


def parse_args() -> argparse.Namespace:
    """Parse rank-sweep, training, sampling, and FID settings."""
    parser = argparse.ArgumentParser(
        description="Train short AsymFlow CIFAR-10 runs for multiple ranks and compute FID."
    )
    parser.add_argument(
        "--ranks",
        default=None,
        help=(
            "Comma-separated ranks. If omitted, global uses DEFAULT_RANKS and "
            "patchwise uses a patch-size-specific default grid."
        ),
    )
    parser.add_argument(
        "--projection-mode",
        choices=["global", "patchwise"],
        default="global",
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument(
        "--patch-sizes",
        default=None,
        help="Comma-separated patch sizes for patch-wise sweeps, e.g. 4,8,16.",
    )
    parser.add_argument(
        "--patch-pca-dir",
        type=Path,
        default=Path("asymflow/results/patchwise_pca_cifar10"),
    )
    parser.add_argument("--train-steps", type=int, default=50_000)
    parser.add_argument("--save-step", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-sigma-min", type=float, default=5e-2)
    parser.add_argument(
        "--time-sampler",
        choices=["asymflow_logit_normal", "uniform"],
        default="asymflow_logit_normal",
    )
    parser.add_argument("--logit-normal-mean", type=float, default=0.8)
    parser.add_argument("--logit-normal-std", type=float, default=0.8)
    parser.add_argument("--num-fid-samples", type=int, default=10_000)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--sampling-sigma-min", type=float, default=4e-2)
    parser.add_argument("--fid-real-batch-size", type=int, default=256)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--train-out-root",
        type=Path,
        default=DEFAULT_TRAIN_OUT_ROOT,
    )
    return parser.parse_args()


def parse_nonnegative_int_list(value: str, name: str) -> list[int]:
    """Parse a comma-separated list of non-negative integers."""
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError(f"{name} must contain non-negative integers.")
    return values


def parse_ranks(value: str | None, projection_mode: str) -> list[int] | None:
    """Parse comma-separated ranks, or return None for patch-wise defaults."""
    if value is None:
        if projection_mode == "patchwise":
            return None
        value = DEFAULT_RANKS
    return parse_nonnegative_int_list(value, "Ranks")


def parse_patch_sizes(args: argparse.Namespace) -> list[int]:
    """Return the patch-size grid for the current projection mode."""
    if args.projection_mode != "patchwise":
        return [args.patch_size]
    if args.patch_sizes is None:
        patch_sizes = [4, 8, 16]
    else:
        patch_sizes = parse_nonnegative_int_list(args.patch_sizes, "Patch sizes")

    for patch_size in patch_sizes:
        if patch_size <= 0 or 32 % patch_size != 0:
            raise ValueError(f"patch_size must divide 32, got {patch_size}.")
    return patch_sizes


def default_patchwise_ranks(patch_size: int) -> list[int]:
    """Return the default local-rank grid for one patch size."""
    patch_dim = 3 * patch_size * patch_size
    if patch_size == 4:
        ranks = [0, 2, 4, 8, 32, patch_dim]
    elif patch_size == 8:
        ranks = [0, 4, 8, 32, 64, patch_dim]
    elif patch_size == 16:
        ranks = [0, 8, 32, 128, patch_dim]
    else:
        ranks = [0, max(1, patch_dim // 64), max(1, patch_dim // 32), max(1, patch_dim // 8), patch_dim]
    return sorted(set(ranks))


def ranks_for_patch(ranks: list[int] | None, patch_size: int) -> list[int]:
    """Return explicit ranks or the default grid for this patch size."""
    if ranks is None:
        return default_patchwise_ranks(patch_size)
    return valid_ranks_for_patch(ranks, patch_size)


def valid_ranks_for_patch(ranks: list[int], patch_size: int) -> list[int]:
    """Keep ranks that fit inside one flattened RGB patch."""
    patch_dim = 3 * patch_size * patch_size
    valid = [rank for rank in ranks if rank <= patch_dim]
    skipped = [rank for rank in ranks if rank > patch_dim]
    if skipped:
        print(f"[patch_size={patch_size}] skipping invalid ranks {skipped}; patch_dim={patch_dim}")
    return valid


def ensure_patchwise_pca_exists(patch_pca_dir: Path, patch_size: int) -> None:
    """Fail early if the shared patch-wise PCA was not precomputed."""
    patch_dir = patch_pca_dir / f"patch_size_{patch_size:03d}"
    missing = [
        path
        for path in [
            patch_dir / "basis_full.pt",
            patch_dir / "eigenvalues.pt",
            patch_dir / "metadata.json",
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing patch-wise PCA for patch_size={patch_size} in {patch_dir}. "
            f"Run: python -m asymflow.compute_patchwise_pca_cifar10 --patch-size {patch_size}"
        )


def resolve_output_roots(args: argparse.Namespace) -> None:
    """Move patch-wise training runs to a separate default folder."""
    if args.projection_mode != "patchwise":
        return
    if args.train_out_root == DEFAULT_TRAIN_OUT_ROOT:
        args.train_out_root = DEFAULT_PATCHWISE_TRAIN_ROOT


def output_dir_for_patch(
    args: argparse.Namespace,
    base_out_dir: Path,
    patch_size: int,
    multiple_patch_sizes: bool,
) -> Path:
    """Return the metrics directory for one patch size."""
    if args.projection_mode != "patchwise":
        return base_out_dir
    if base_out_dir == DEFAULT_OUT_DIR:
        return DEFAULT_PATCHWISE_OUT_ROOT / f"patch_{patch_size:02d}"
    if multiple_patch_sizes:
        return base_out_dir / f"patch_{patch_size:02d}"
    return base_out_dir


def train_run_dir(args: argparse.Namespace, rank: int) -> Path:
    """Return the training output directory used by train_asymflow_unet_cifar10.py."""
    if args.projection_mode == "patchwise":
        return (
            args.train_out_root
            / f"patch_{args.patch_size:02d}"
            / f"rank_{rank:04d}_steps_{args.train_steps:06d}"
        )
    return args.train_out_root / f"global_rank_{rank:04d}_steps_{args.train_steps:06d}"


def checkpoint_path_for_rank(args: argparse.Namespace, rank: int) -> Path:
    """Return the latest checkpoint path for a rank."""
    return train_run_dir(args, rank) / "checkpoints" / "latest.pt"


def run_training_for_rank(args: argparse.Namespace, rank: int) -> Path:
    """Launch one short training run unless its checkpoint already exists."""
    checkpoint_path = checkpoint_path_for_rank(args, rank)
    if args.skip_existing and checkpoint_path.exists():
        print(f"[rank={rank}] found existing checkpoint: {checkpoint_path}")
        return checkpoint_path

    command = [
        sys.executable,
        "-m",
        "asymflow.train_asymflow_unet_cifar10",
        "--rank",
        str(rank),
        "--projection-mode",
        args.projection_mode,
        "--patch-size",
        str(args.patch_size),
        "--patch-pca-dir",
        str(args.patch_pca_dir),
        "--total-steps",
        str(args.train_steps),
        "--save-step",
        str(args.save_step),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--ema-decay",
        str(args.ema_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--grad-clip",
        str(args.grad_clip),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--sigma-min",
        str(args.train_sigma_min),
        "--time-sampler",
        args.time_sampler,
        "--logit-normal-mean",
        str(args.logit_normal_mean),
        "--logit-normal-std",
        str(args.logit_normal_std),
        "--out-dir",
        str(args.train_out_root),
    ]
    print(f"[rank={rank}] training command: {' '.join(command)}")
    subprocess.run(command, check=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Training finished but checkpoint is missing: {checkpoint_path}")
    return checkpoint_path


def to_fid_images(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] image tensors to [0, 1] tensors for torchmetrics FID."""
    return ((images.detach() + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def update_fid_with_real_images(
    fid: FrechetInceptionDistance,
    dataset,
    num_images: int,
    device: torch.device,
) -> None:
    """Feed real CIFAR-10 images into the FID metric."""
    seen = 0
    for batch in tqdm(dataset.dataloader, desc="FID real images"):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device, non_blocking=True)
        remaining = num_images - seen
        if images.shape[0] > remaining:
            images = images[:remaining]
        fid.update(to_fid_images(images), real=True)
        seen += images.shape[0]
        if seen >= num_images:
            break

    if seen < num_images:
        raise RuntimeError(f"Dataset only yielded {seen} real images, expected {num_images}.")


@torch.no_grad()
def update_fid_with_generated_images(
    fid: FrechetInceptionDistance,
    checkpoint_path: Path,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Generate images in batches, update FID, and return a small preview batch."""
    model, _checkpoint = load_asymflow_model(
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=True,
    )
    if args.projection_mode == "patchwise":
        basis, _spectrum = load_patchwise_pca_basis(
            rank=rank,
            patch_size=args.patch_size,
            pca_dir=args.patch_pca_dir,
            device=device,
        )
    else:
        basis, _spectrum = load_pca_basis(rank, device)
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)

    preview: torch.Tensor | None = None
    remaining = args.num_fid_samples
    progress = tqdm(total=args.num_fid_samples, desc=f"FID generated rank={rank}")
    while remaining > 0:
        batch_size = min(args.sample_batch_size, remaining)
        initial_noise = torch.randn(
            batch_size,
            3,
            32,
            32,
            device=device,
            generator=generator,
        )
        samples = euler_generate(
            model=model,
            initial_noise=initial_noise,
            basis=basis,
            projection_mode=args.projection_mode,
            patch_size=args.patch_size,
            num_steps=args.num_sampling_steps,
            sigma_min=args.sampling_sigma_min,
            show_progress=False,
        )
        fid.update(to_fid_images(samples), real=False)
        if preview is None:
            preview = samples[: min(64, samples.shape[0])].detach().cpu()
        remaining -= batch_size
        progress.update(batch_size)
    progress.close()

    if preview is None:
        raise RuntimeError("No generated images were produced.")
    return preview


def compute_fid_for_rank(
    checkpoint_path: Path,
    rank: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Compute FID for one trained rank."""
    fid = FrechetInceptionDistance(feature=args.fid_feature, normalize=True).to(device)
    real_dataset = load_cifar10_dataset(batch_size=args.fid_real_batch_size)
    update_fid_with_real_images(
        fid=fid,
        dataset=real_dataset,
        num_images=args.num_fid_samples,
        device=device,
    )
    preview = update_fid_with_generated_images(
        fid=fid,
        checkpoint_path=checkpoint_path,
        rank=rank,
        args=args,
        device=device,
    )
    return float(fid.compute().detach().cpu()), preview


def load_existing_results(out_dir: Path) -> list[dict]:
    """Load existing FID results if present."""
    results_path = out_dir / "fid_results.json"
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_key(result: dict) -> tuple[str, int | None, int]:
    """Return the identity of one FID result row."""
    return (
        str(result.get("projection_mode", "global")),
        result.get("patch_size"),
        int(result["rank"]),
    )


def upsert_result(results: list[dict], result: dict) -> list[dict]:
    """Insert or replace one result with the same projection mode, patch size, and rank."""
    key = result_key(result)
    merged = [item for item in results if result_key(item) != key]
    merged.append(result)
    return sorted(
        merged,
        key=lambda item: (
            str(item.get("projection_mode", "global")),
            -1 if item.get("patch_size") is None else int(item["patch_size"]),
            int(item["rank"]),
        ),
    )


def save_results(results: list[dict], out_dir: Path) -> None:
    """Save sweep metrics as JSON and plot FID versus rank."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fid_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    ordered = sorted(results, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in ordered]
    fids = [float(item["fid"]) for item in ordered]
    x_positions = list(range(len(ranks)))

    projection_mode = str(ordered[0].get("projection_mode", "global")) if ordered else "global"
    patch_size = ordered[0].get("patch_size") if ordered else None
    title = "CIFAR-10 AsymFlow FID vs rank"
    if projection_mode == "patchwise" and patch_size is not None:
        title += f" (patch={patch_size})"

    plt.figure(figsize=(max(7, 0.8 * len(ranks) + 2), 4.5))
    plt.plot(x_positions, fids, marker="o", linewidth=2)
    plt.xticks(x_positions, [str(rank) for rank in ranks], rotation=35, ha="right")
    plt.xlabel("rank")
    plt.ylabel("FID")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fid_vs_rank.png", dpi=180)
    plt.close()


def main() -> None:
    """Run the rank sweep end to end."""
    args = parse_args()
    ranks = parse_ranks(args.ranks, args.projection_mode)
    patch_sizes = parse_patch_sizes(args)
    multiple_patch_sizes = args.projection_mode == "patchwise" and len(patch_sizes) > 1
    base_out_dir = args.out_dir
    resolve_output_roots(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("dataset: cifar10")
    print(f"device: {device}")
    print(f"projection_mode: {args.projection_mode}")
    if args.projection_mode == "patchwise":
        print(f"patch_sizes: {patch_sizes}")
        print(f"patch_pca_dir: {args.patch_pca_dir}")
    print(f"ranks: {ranks if ranks is not None else 'patch-wise defaults'}")
    print(f"train_steps: {args.train_steps}")
    print(f"lr: {args.lr}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"grad_clip: {args.grad_clip}")
    print(f"num_fid_samples: {args.num_fid_samples}")
    print(f"num_sampling_steps: {args.num_sampling_steps}")
    print(f"train_sigma_min: {args.train_sigma_min}")
    print(f"sampling_sigma_min: {args.sampling_sigma_min}")
    print(f"time_sampler: {args.time_sampler}")
    print(f"logit_normal_mean: {args.logit_normal_mean}")
    print(f"logit_normal_std: {args.logit_normal_std}")
    print(f"base_out_dir: {base_out_dir}")
    print(f"train_out_root: {args.train_out_root}")

    for patch_size in patch_sizes:
        args.patch_size = patch_size
        args.out_dir = output_dir_for_patch(
            args=args,
            base_out_dir=base_out_dir,
            patch_size=patch_size,
            multiple_patch_sizes=multiple_patch_sizes,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)

        if args.projection_mode == "patchwise":
            ensure_patchwise_pca_exists(args.patch_pca_dir, patch_size)
            active_ranks = ranks_for_patch(ranks, patch_size)
            print(f"\n=== patch_size={patch_size} | ranks={active_ranks} | out_dir={args.out_dir} ===")
        else:
            active_ranks = ranks

        if not active_ranks:
            print(f"[patch_size={patch_size}] no valid ranks, skipping.")
            continue

        results = load_existing_results(args.out_dir)
        for rank in active_ranks:
            checkpoint_path = run_training_for_rank(args, rank)
            fid_value, preview = compute_fid_for_rank(
                checkpoint_path=checkpoint_path,
                rank=rank,
                args=args,
                device=device,
            )
            rank_out_dir = args.out_dir / f"{args.projection_mode}_rank_{rank:04d}"
            rank_out_dir.mkdir(parents=True, exist_ok=True)
            save_image_grid(preview, rank_out_dir / "preview_samples.png")

            result = {
                "rank": rank,
                "projection_mode": args.projection_mode,
                "patch_size": args.patch_size if args.projection_mode == "patchwise" else None,
                "fid": fid_value,
                "checkpoint": str(checkpoint_path),
                "train_steps": args.train_steps,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "grad_clip": args.grad_clip,
                "num_fid_samples": args.num_fid_samples,
                "num_sampling_steps": args.num_sampling_steps,
                "train_sigma_min": args.train_sigma_min,
                "sampling_sigma_min": args.sampling_sigma_min,
                "time_sampler": args.time_sampler,
                "logit_normal_mean": args.logit_normal_mean,
                "logit_normal_std": args.logit_normal_std,
            }
            results = upsert_result(results, result)
            save_results(results, args.out_dir)
            print(f"[patch_size={patch_size} rank={rank}] FID={fid_value:.4f}")


if __name__ == "__main__":
    main()
