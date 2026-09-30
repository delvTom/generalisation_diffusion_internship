"""Generate CIFAR-10 samples with a PCA-hybrid sampler for Wiener_asym.

Wiener_asym_r is the affine closed-form estimator of the raw AsymFlow target

    u_A = x_1 - P_r eps,

where P_r is the projector onto the first r PCA directions.  This target has a
hybrid meaning:

    P_r u_A       = P_r (x_1 - eps)  is velocity-like,
    (I-P_r) u_A   = (I-P_r) x_1      is clean-image-like.

Using Wiener_asym_r directly as a full velocity is therefore not coherent.
This script therefore uses a PCA-hybrid sampler:

    P_r x_next is advanced with Euler using P_r Wiener_asym_r.
    (I-P_r) x_next is updated as a denoising step using
    (I-P_r) Wiener_asym_r as a clean-image prediction.

This rule interpolates between a denoiser-style sampler at r=0 and a velocity
Euler sampler at r=d.
Inputs: CIFAR-10 and PCA cache; output: sample grids and run metadata.
Example: python -m fm_comparisons.generate_wiener_asym_hybrid_sampler_cifar10 --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from compare_wiener_asym_direct_cifar10 import (
    dataset_output_slug,
    load_dataset,
    load_experiment_config,
    load_or_compute_wiener_svd,
    parse_ranks,
    predict_wiener_asym,
    project_noise_on_pca_rank,
)


def parse_args() -> argparse.Namespace:
    """Parse generation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ranks", default="0,8,32,256,3072")
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def project_flat(x: torch.Tensor, U: torch.Tensor, rank: int) -> torch.Tensor:
    """Project a batch of image tensors onto the first rank PCA directions."""
    flat = x.flatten(start_dim=1)
    projected = project_noise_on_pca_rank(flat, U, rank)
    return projected.view_as(x)


@torch.no_grad()
def pca_hybrid_step(
    xt: torch.Tensor,
    t: float,
    next_t: float,
    dt: float,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Advance one PCA-hybrid step using velocity in P and denoising in I-P."""
    wiener_asym = predict_wiener_asym(
        xt=xt,
        t=t,
        U=U,
        eigenvalues=eigenvalues,
        mean=mean,
        rank=rank,
    )

    p_xt = project_flat(xt, U, rank)
    p_field = project_flat(wiener_asym, U, rank)
    p_next = p_xt + dt * p_field

    xt_perp = xt - p_xt
    clean_perp = wiener_asym - project_flat(wiener_asym, U, rank)
    noise_perp = (xt_perp - t * clean_perp) / (1.0 - t)
    perp_next = next_t * clean_perp + (1.0 - next_t) * noise_perp

    return p_next + perp_next


@torch.no_grad()
def generate_for_rank(
    initial_noise: torch.Tensor,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean: torch.Tensor,
    rank: int,
    num_steps: int,
) -> torch.Tensor:
    """Generate samples for one rank with the PCA-hybrid sampler."""
    samples = initial_noise.clone()
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        next_t = min(1.0, (step + 1) / num_steps)
        print(f"rank={rank} | step {step + 1:03d}/{num_steps} | t={t:.4f}")

        samples = pca_hybrid_step(
            xt=samples,
            t=t,
            next_t=next_t,
            dt=dt,
            U=U,
            eigenvalues=eigenvalues,
            mean=mean,
            rank=rank,
        )

    return samples


def save_generation_grid(
    samples_by_rank: dict[int, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save a grid with columns=ranks and rows=shared noises."""
    first_rank = next(iter(samples_by_rank))
    num_samples = samples_by_rank[first_rank].shape[0]
    ranks = list(samples_by_rank.keys())

    fig, axes = plt.subplots(
        num_samples,
        len(ranks),
        figsize=(2.1 * len(ranks), 2.1 * num_samples),
        squeeze=False,
    )

    for col, rank in enumerate(ranks):
        axes[0, col].set_title(f"r={rank}", fontsize=14)
        images = postprocess(samples_by_rank[rank].detach().cpu())
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


def save_config(args: argparse.Namespace, ranks: list[int], out_path: Path) -> None:
    """Save generation metadata."""
    payload = vars(args).copy()
    payload["out_dir"] = str(payload["out_dir"]) if payload["out_dir"] is not None else None
    payload["ranks"] = ranks
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Run the PCA-hybrid Wiener_asym generation experiment."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"wiener_asym_hybrid_sampler_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_experiment_config(args.dataset, args.batch_size, args.num_workers)
    dataset = load_dataset(cfg)
    U, eigenvalues, _Vh, mean = load_or_compute_wiener_svd(dataset, device)
    ranks = parse_ranks(args.ranks, max_rank=U.shape[1])

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"ranks: {ranks}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"out_dir: {out_dir}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generator,
    )

    samples_by_rank = {
        rank: generate_for_rank(
            initial_noise=initial_noise,
            U=U,
            eigenvalues=eigenvalues,
            mean=mean,
            rank=rank,
            num_steps=args.num_steps,
        )
        for rank in ranks
    }

    save_generation_grid(
        samples_by_rank=samples_by_rank,
        postprocess=dataset.postprocess,
        out_path=out_dir / "generated_samples.png",
    )
    save_config(args, ranks, out_dir / "config.json")
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
