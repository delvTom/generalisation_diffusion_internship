"""Generate CIFAR-10 samples with PCA hybrids of Artem and Wiener estimators.

For a PCA projector P_r, this script compares two full-velocity fields:

Version A:
    u_A = P_r u_Artem + (I-P_r) (x1_Wiener - x_t) / (1 - t)

Version B:
    u_B = P_r u_Wiener + (I-P_r) (x1_Artem - x_t) / (1 - t)

where x1_Artem is Artem's x1_mask estimator, x1_Wiener is the affine Wiener
denoiser with the dataset-mean bias, and u_Artem/u_Wiener are the corresponding
velocity estimates.  The goal is to test whether low PCA frequencies prefer a
velocity-like estimator while the orthogonal complement prefers a denoiser-like
estimator.
Inputs: CIFAR-10, PCA cache and teacher checkpoint; output: sample grids.
Example: python -m fm_comparisons.generate_pca_hybrid_artem_wiener_cifar10 --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from compare_fm_unet_closed_form_mse import (  # noqa: E402
    binarize_like_artem,
    build_fm_wiener_operator,
    build_mask_operator,
    dataset_output_slug,
    load_dataset,
    load_experiment_config,
    load_or_compute_wiener_svd,
    predict_closed_form_x1,
    predict_wiener_velocity,
)
from local_diffusion.configuration import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse generation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--ranks", default="0,8,32,256,3072")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_generation_config(dataset_name: str, batch_size: int | None, num_workers: int | None, subset_size: int | None):
    """Load the locality dataset config with optional loader overrides."""
    overrides = ["metrics.wandb.enabled=false"]
    if batch_size is not None:
        overrides.append(f"dataset.batch_size={batch_size}")
    if num_workers is not None:
        overrides.append(f"dataset.num_workers={num_workers}")
    if subset_size is not None:
        overrides.append(f"dataset.subset_size={subset_size}")
    return load_config(f"configs/pca_locality/{dataset_name}.yaml", overrides=overrides)


def parse_ranks(value: str, max_rank: int) -> list[int]:
    """Parse comma-separated PCA ranks and keep valid values."""
    ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ranks or any(rank < 0 for rank in ranks):
        raise ValueError("Ranks must be non-negative integers.")
    valid = sorted(set(rank for rank in ranks if rank <= max_rank))
    skipped = sorted(set(rank for rank in ranks if rank > max_rank))
    if skipped:
        print(f"skipping invalid ranks {skipped}; max rank is {max_rank}")
    return valid


def project_pca(x: torch.Tensor, U: torch.Tensor, rank: int) -> torch.Tensor:
    """Project image batches onto the first rank PCA directions."""
    if rank == 0:
        return torch.zeros_like(x)
    flat = x.flatten(start_dim=1)
    basis = U[:, :rank]
    projected = (flat @ basis) @ basis.T
    return projected.view_as(x)


@torch.no_grad()
def predict_wiener_x1(
    xt: torch.Tensor,
    t: float,
    wiener_operator: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """Predict x1 with the affine Wiener denoiser and its mean bias."""
    xt_flat = xt.flatten(start_dim=1)
    mean = mean.to(device=xt.device, dtype=xt.dtype)
    mean_bias = mean - t * (mean.unsqueeze(0) @ wiener_operator.T).squeeze(0)
    x1_hat = xt_flat @ wiener_operator.T + mean_bias.unsqueeze(0)
    return x1_hat.view_as(xt)


@torch.no_grad()
def predict_artem_velocity(
    xt: torch.Tensor,
    t: float,
    candidate_dataset,
    mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict Artem x1_mask and convert it to velocity."""
    x1_hat = predict_closed_form_x1(
        xt=xt,
        t=t,
        candidate_dataset=candidate_dataset,
        mask=mask,
        temperature=temperature,
    )
    velocity = (x1_hat - xt) / (1.0 - t)
    return x1_hat, velocity


@torch.no_grad()
def build_hybrid_velocities(
    xt_a: torch.Tensor,
    xt_b: torch.Tensor,
    t: float,
    rank: int,
    U: torch.Tensor,
    wiener_operator: torch.Tensor,
    mean: torch.Tensor,
    candidate_dataset,
    x1_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build version-A and version-B hybrid velocities for one rank."""
    x1_wiener_a = predict_wiener_x1(xt_a, t, wiener_operator, mean)
    _x1_artem_a, velocity_artem_a = predict_artem_velocity(
        xt=xt_a,
        t=t,
        candidate_dataset=candidate_dataset,
        mask=x1_mask,
        temperature=temperature,
    )
    velocity_a = project_pca(velocity_artem_a, U, rank) + (
        (x1_wiener_a - xt_a) - project_pca(x1_wiener_a - xt_a, U, rank)
    ) / (1.0 - t)

    x1_artem_b, _velocity_artem_b = predict_artem_velocity(
        xt=xt_b,
        t=t,
        candidate_dataset=candidate_dataset,
        mask=x1_mask,
        temperature=temperature,
    )
    velocity_wiener_b = predict_wiener_velocity(
        xt=xt_b,
        t=t,
        wiener_operator=wiener_operator,
        mean=mean,
    )
    velocity_b = project_pca(velocity_wiener_b, U, rank) + (
        (x1_artem_b - xt_b) - project_pca(x1_artem_b - xt_b, U, rank)
    ) / (1.0 - t)

    return velocity_a, velocity_b


@torch.no_grad()
def euler_generate(
    initial_noise: torch.Tensor,
    candidate_dataset,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    ranks: list[int],
    num_steps: int,
    temperature: float,
    mask_threshold: float,
) -> dict[str, torch.Tensor]:
    """Generate samples for all requested ranks and both hybrid versions."""
    samples: dict[str, torch.Tensor] = {}
    for rank in ranks:
        samples[f"A P-Artem_perp-Wiener r={rank}"] = initial_noise.clone()
        samples[f"B P-Wiener_perp-Artem r={rank}"] = initial_noise.clone()

    step_size = 1.0 / num_steps
    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
        x1_mask = binarize_like_artem(
            build_mask_operator(wiener_operator, t, "x1_mask"),
            mask_threshold,
        )

        for rank in ranks:
            key_a = f"A P-Artem_perp-Wiener r={rank}"
            key_b = f"B P-Wiener_perp-Artem r={rank}"
            velocity_a, velocity_b = build_hybrid_velocities(
                xt_a=samples[key_a],
                xt_b=samples[key_b],
                t=t,
                rank=rank,
                U=U,
                wiener_operator=wiener_operator,
                mean=mean,
                candidate_dataset=candidate_dataset,
                x1_mask=x1_mask,
                temperature=temperature,
            )
            samples[key_a] = samples[key_a] + step_size * velocity_a
            samples[key_b] = samples[key_b] + step_size * velocity_b

    return samples


def save_generation_grid(
    samples: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save a grid with rows=shared noises and columns=hybrid estimators."""
    num_samples = next(iter(samples.values())).shape[0]
    names = list(samples.keys())
    fig, axes = plt.subplots(
        num_samples,
        len(names),
        figsize=(2.2 * len(names), 2.1 * num_samples),
        squeeze=False,
    )

    for col, name in enumerate(names):
        axes[0, col].set_title(name, fontsize=11)
        images = postprocess(samples[name].detach().cpu())
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
    """Run the PCA hybrid Artem/Wiener generation experiment."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"pca_hybrid_artem_wiener_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_generation_config(args.dataset, args.batch_size, args.num_workers, subset_size=None)
    candidate_cfg = load_generation_config(
        args.dataset,
        args.batch_size,
        args.num_workers,
        subset_size=args.num_xi,
    )
    pca_params = cfg.model.params or {}
    temperature = float(pca_params.get("temperature", 1.0))
    mask_threshold = float(pca_params.get("mask_threshold", 0.02))

    dataset = load_dataset(cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    U, eigenvalues, Vh, mean = load_or_compute_wiener_svd(candidate_dataset, device)
    ranks = parse_ranks(args.ranks, max_rank=U.shape[1])

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"ranks: {ranks}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"num_xi: {args.num_xi}")
    print(f"num_workers: {cfg.dataset.num_workers}")
    print(f"temperature: {temperature}")
    print(f"mask_threshold: {mask_threshold}")
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

    samples = euler_generate(
        initial_noise=initial_noise,
        candidate_dataset=candidate_dataset,
        U=U,
        eigenvalues=eigenvalues,
        Vh=Vh,
        mean=mean,
        ranks=ranks,
        num_steps=args.num_steps,
        temperature=temperature,
        mask_threshold=mask_threshold,
    )
    save_generation_grid(
        samples=samples,
        postprocess=dataset.postprocess,
        out_path=out_dir / "generated_samples.png",
    )
    save_config(args, ranks, out_dir / "config.json")
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
