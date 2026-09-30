"""Compare Wiener_asym directly with the pretrained FM U-Net velocity.

Wiener_asym is the optimal affine estimator of the raw AsymFlow target

    u_A = x_1 - P eps

under the Flow Matching convention

    x_t = t x_1 + (1 - t) eps.

Here P is the orthogonal projector onto the first PCA directions of the data.
For each rank and time t, we build

    Wiener_asym_r(x_t,t) = mu + A_t (x_t - t mu)

then compare it directly with the pretrained Flow Matching U-Net velocity.
This direct comparison mixes parametrizations when rank is neither full nor
zero, but it is useful as a first diagnostic of how the raw asymmetric affine
field relates to the learned U-Net velocity field.

Generation also uses Wiener_asym_r directly as the Euler vector field, without
converting it back into a full-rank velocity.
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

from fm_comparisons.compare_fm_unet_closed_form_mse import (  # noqa: E402
    collect_images,
    dataset_output_slug,
    default_checkpoint_for_dataset,
    load_fm_unet,
    load_dataset,
    load_or_compute_wiener_svd,
    parse_t_values,
    predict_fm_unet_velocity,
)
from local_diffusion.configuration import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse experiment hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-xt", type=int, default=128)
    parser.add_argument("--xt-batch-size", type=int, default=None)
    parser.add_argument("--ranks", default="0,8,16,32,64,128,256,3072")
    parser.add_argument(
        "--t-values",
        default="0.05,0.15,0.25,0.35,0.45,0.5,0.55,0.65,0.75,0.85,0.95",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument(
        "--generation-ranks",
        default=None,
        help="Comma-separated ranks to show in the generation plot. Defaults to --ranks.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_experiment_config(
    dataset_name: str,
    batch_size: int | None,
    num_workers: int | None,
):
    """Load Artem's dataset config with lightweight CLI overrides."""
    overrides = ["metrics.wandb.enabled=false"]
    if batch_size is not None:
        overrides.append(f"dataset.batch_size={batch_size}")
    if num_workers is not None:
        overrides.append(f"dataset.num_workers={num_workers}")
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


def build_pca_indicator(rank: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return p_i = 1 on the first rank PCA directions and 0 elsewhere."""
    indicator = torch.zeros(dim, device=device, dtype=dtype)
    if rank > 0:
        indicator[:rank] = 1.0
    return indicator


def build_asymflow_affine_diagonal(
    eigenvalues: torch.Tensor,
    rank: int,
    t: float,
) -> torch.Tensor:
    """Build the diagonal of A_t in the PCA basis.

    If Sigma = U diag(lambda_i) U^T and P = U diag(p_i) U^T, then

        A_t = U diag((t lambda_i - (1-t) p_i)
                    / (t^2 lambda_i + (1-t)^2)) U^T.
    """
    p_indicator = build_pca_indicator(
        rank=rank,
        dim=eigenvalues.numel(),
        device=eigenvalues.device,
        dtype=eigenvalues.dtype,
    )
    denominator = t * t * eigenvalues + (1.0 - t) ** 2
    numerator = t * eigenvalues - (1.0 - t) * p_indicator
    return numerator / denominator


def apply_pca_diagonal_operator(
    x: torch.Tensor,
    U: torch.Tensor,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    """Apply U diag(diagonal) U.T to a batch of image vectors stored as rows."""
    coeffs = x @ U  # [B, N] @ [N, N] -> [B, N]
    return (coeffs * diagonal.unsqueeze(0)) @ U.T


def project_noise_on_pca_rank(
    eps_flat: torch.Tensor,
    U: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Compute P z where P is the PCA projector of the requested rank."""
    if rank == 0:
        return torch.zeros_like(eps_flat)
    basis = U[:, :rank]
    return (eps_flat @ basis) @ basis.T


@torch.no_grad()
def predict_wiener_asym(
    xt: torch.Tensor,
    t: float,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Predict Wiener_asym_r(x_t,t), the affine estimator of the raw AsymFlow target."""
    xt_flat = xt.flatten(start_dim=1)
    mean = mean.to(device=xt.device, dtype=xt.dtype)
    centered_xt = xt_flat - t * mean.unsqueeze(0)
    diagonal = build_asymflow_affine_diagonal(eigenvalues, rank, t)
    prediction = mean.unsqueeze(0) + apply_pca_diagonal_operator(centered_xt, U, diagonal)
    return prediction.view_as(xt)


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return scalar MSE as a Python float."""
    return torch.mean((prediction - target) ** 2).item()


def run_experiment(
    model: torch.nn.Module,
    x1_all: torch.Tensor,
    eps_all: torch.Tensor,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean: torch.Tensor,
    ranks: list[int],
    t_values: list[float],
    xt_batch_size: int,
) -> list[dict[str, float | int]]:
    """Compare raw affine AsymFlow predictions directly with FM U-Net velocity."""
    num_xt = x1_all.shape[0]
    results: list[dict[str, float | int]] = []

    for t in t_values:
        print(f"\n=== t={t:.2f} ===")
        for rank in ranks:
            mse_values: list[float] = []
            for start in range(0, num_xt, xt_batch_size):
                end = min(start + xt_batch_size, num_xt)
                x1 = x1_all[start:end]
                eps = eps_all[start:end]
                xt = t * x1 + (1.0 - t) * eps
                unet_velocity = predict_fm_unet_velocity(model, xt, t)
                wiener_asym = predict_wiener_asym(
                    xt=xt,
                    t=t,
                    U=U,
                    eigenvalues=eigenvalues,
                    mean=mean,
                    rank=rank,
                )
                mse_values.append(mean_squared_error(wiener_asym, unet_velocity))

            mse = sum(mse_values) / len(mse_values)
            row = {"t": t, "rank": rank, "mse_wiener_asym_vs_unet_velocity": mse}
            print(row)
            results.append(row)

    return results


def save_metrics(results: list[dict[str, float | int]], out_path: Path) -> None:
    """Save metrics as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def plot_mse(results: list[dict[str, float | int]], out_path: Path) -> None:
    """Plot direct MSE to the FM U-Net velocity as a function of t."""
    t_values = sorted({float(row["t"]) for row in results})
    ranks = sorted({int(row["rank"]) for row in results})

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rank in ranks:
        rows = {float(row["t"]): row for row in results if int(row["rank"]) == rank}
        values = [float(rows[t]["mse_wiener_asym_vs_unet_velocity"]) for t in t_values]
        ax.plot(t_values, values, marker="o", linewidth=2.0, label=f"rank={rank}")

    ax.set_title("FM U-Net velocity vs Wiener_asym")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_mse_vs_rank(results: list[dict[str, float | int]], out_path: Path) -> None:
    """Plot direct MSE to the FM U-Net velocity as a function of rank."""
    t_values = sorted({float(row["t"]) for row in results})
    ranks = sorted({int(row["rank"]) for row in results})

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for t in t_values:
        rows = {int(row["rank"]): row for row in results if float(row["t"]) == t}
        values = [float(rows[rank]["mse_wiener_asym_vs_unet_velocity"]) for rank in ranks]
        ax.plot(ranks, values, marker="o", linewidth=2.0, label=f"t={t:.2f}")

    ax.set_title("FM U-Net velocity vs Wiener_asym")
    ax.set_xlabel("rank r")
    ax.set_ylabel("MSE")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


@torch.no_grad()
def euler_generate(
    initial_noise: torch.Tensor,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    mean: torch.Tensor,
    ranks: list[int],
    num_steps: int,
) -> dict[str, torch.Tensor]:
    """Generate images by using the raw Wiener_asym output directly as a field."""
    samples = {f"Wiener_asym r={rank}": initial_noise.clone() for rank in ranks}
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        for rank in ranks:
            key = f"Wiener_asym r={rank}"
            raw_field = predict_wiener_asym(
                xt=samples[key],
                t=t,
                U=U,
                eigenvalues=eigenvalues,
                mean=mean,
                rank=rank,
            )
            samples[key] = samples[key] + step_size * raw_field

    return samples


def save_generation_grid(
    samples: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save a grid with rows=shared noises and columns=Wiener_asym ranks."""
    num_samples = next(iter(samples.values())).shape[0]
    estimator_names = list(samples.keys())
    num_cols = len(estimator_names)

    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(2.1 * num_cols, 2.1 * num_samples),
        squeeze=False,
    )

    for col, name in enumerate(estimator_names):
        axes[0, col].set_title(name, fontsize=14)
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


def main() -> None:
    """Run the experiment."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"wiener_asym_direct_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_experiment_config(args.dataset, args.batch_size, args.num_workers)
    dataset = load_dataset(cfg)
    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)
    model = load_fm_unet(checkpoint_path, device, args.dataset)
    U, eigenvalues, _Vh, mean = load_or_compute_wiener_svd(dataset, device)

    t_values = parse_t_values(args.t_values)
    ranks = parse_ranks(args.ranks, max_rank=U.shape[1])
    generation_ranks = parse_ranks(args.generation_ranks or args.ranks, max_rank=U.shape[1])

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"num_xt: {args.num_xt}")
    xt_batch_size = args.xt_batch_size or int(cfg.sampling.batch_size)
    print(f"xt_batch_size: {xt_batch_size}")
    print(f"num_workers: {cfg.dataset.num_workers}")
    print(f"ranks: {ranks}")
    print(f"generation_ranks: {generation_ranks}")
    print(f"t_values: {t_values}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"out_dir: {out_dir}")

    mse_generator = torch.Generator(device=device).manual_seed(args.seed)
    x1_all = collect_images(dataset, args.num_xt, device)
    eps_all = torch.randn(x1_all.shape, device=device, generator=mse_generator)

    results = run_experiment(
        model=model,
        x1_all=x1_all,
        eps_all=eps_all,
        U=U,
        eigenvalues=eigenvalues,
        mean=mean,
        ranks=ranks,
        t_values=t_values,
        xt_batch_size=xt_batch_size,
    )
    save_metrics(results, out_dir / "metrics.json")
    plot_mse(results, out_dir / "mse_vs_t.png")
    plot_mse_vs_rank(results, out_dir / "mse_vs_rank.png")

    generation_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generation_generator,
    )
    generated_samples = euler_generate(
        initial_noise=initial_noise,
        U=U,
        eigenvalues=eigenvalues,
        mean=mean,
        ranks=generation_ranks,
        num_steps=args.num_steps,
    )
    save_generation_grid(
        samples=generated_samples,
        postprocess=dataset.postprocess,
        out_path=out_dir / "generated_samples.png",
    )
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
