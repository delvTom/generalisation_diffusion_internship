"""Generate samples with an Artem-like estimator of AsymFlow's target.

For a PCA projector P_r, the raw AsymFlow target is

    u_A = x_1 - P_r eps,

with the Flow Matching convention

    x_t = t x_1 + (1 - t) eps.

This script builds an Artem-style local estimator of u_A.  The local masks are
not taken from the usual clean-image Wiener filter: they are induced by the
linear Wiener_asym estimator of u_A.  The predicted u_A is then converted back
to a full-rank velocity through

    u = P_r u_A + ((I - P_r) u_A - (I - P_r) x_t) / (1 - t).

The goal is to test the simplest "one asymmetric mask" version before moving
to separate masks for P_r u and (I - P_r) x_1.
Inputs: CIFAR-10, PCA cache and teacher checkpoint; output: sample grids.
Example: python -m fm_comparisons.generate_asym_artem_mask_cifar10 --help
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
    WeightedStreamingSoftmax,
    binarize_like_artem,
    collect_images,
    dataset_output_slug,
    default_checkpoint_for_dataset,
    load_dataset,
    load_fm_unet,
    load_or_compute_wiener_svd,
    parse_t_values,
    predict_fm_unet_velocity,
)
from compare_wiener_asym_direct_cifar10 import (  # noqa: E402
    build_asymflow_affine_diagonal,
    load_experiment_config,
    parse_ranks,
)
from local_diffusion.configuration import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse experiment options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--num-xt", type=int, default=128)
    parser.add_argument("--xt-batch-size", type=int, default=None)
    parser.add_argument("--ranks", default="0,8,32,256,3072")
    parser.add_argument("--generation-ranks", default=None)
    parser.add_argument(
        "--t-values",
        default="0.05,0.15,0.25,0.35,0.45,0.5,0.55,0.65,0.75,0.85,0.95",
    )
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mse", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_locality_config(
    dataset_name: str,
    batch_size: int | None,
    num_workers: int | None,
    subset_size: int | None,
):
    """Load Artem's dataset config with optional loader/subset overrides."""
    overrides = ["metrics.wandb.enabled=false"]
    if batch_size is not None:
        overrides.append(f"dataset.batch_size={batch_size}")
    if num_workers is not None:
        overrides.append(f"dataset.num_workers={num_workers}")
    if subset_size is not None:
        overrides.append(f"dataset.subset_size={subset_size}")
    return load_config(f"configs/pca_locality/{dataset_name}.yaml", overrides=overrides)


def build_wiener_asym_operator(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    rank: int,
    t: float,
) -> torch.Tensor:
    """Build the linear sensitivity of the affine Wiener_asym estimator."""
    diagonal = build_asymflow_affine_diagonal(eigenvalues, rank, t)
    return (U * diagonal.unsqueeze(0)) @ U.T


def project_flat(x_flat: torch.Tensor, U: torch.Tensor, rank: int) -> torch.Tensor:
    """Project flattened row-vectors onto the first rank PCA directions."""
    if rank == 0:
        return torch.zeros_like(x_flat)
    if rank >= U.shape[1]:
        return x_flat
    basis = U[:, :rank]
    return (x_flat @ basis) @ basis.T


def projector_matrix(U: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the dense PCA projector P_r in pixel coordinates."""
    dim = U.shape[0]
    if rank == 0:
        return torch.zeros(dim, dim, device=U.device, dtype=U.dtype)
    if rank >= U.shape[1]:
        return torch.eye(dim, device=U.device, dtype=U.dtype)
    basis = U[:, :rank]
    return basis @ basis.T


@torch.no_grad()
def predict_asym_artem_uA(
    xt: torch.Tensor,
    t: float,
    candidate_dataset,
    mask: torch.Tensor,
    U: torch.Tensor,
    rank: int,
    temperature: float,
) -> torch.Tensor:
    """Predict u_A with an Artem-style local softmax and an asymmetric mask.

    For a candidate clean image x_i, the compatible noise is

        eps_i = (x_t - t x_i) / (1 - t),

    and the local asymmetric target for output coordinate q is

        u_A,i^q = x_i^q - [P_r M_q (x_t - t x_i)]^q / (1 - t).

    The mask has the same orientation as predict_closed_form_x1: mask[p, q]
    tells whether input coordinate p is used to predict output coordinate q.
    """
    xt_flat = xt.flatten(start_dim=1)  # [B, N]
    t_tensor = torch.tensor(t, device=xt.device, dtype=xt.dtype)
    noise_var = (1.0 - t_tensor) ** 2
    first_moment = WeightedStreamingSoftmax(device=xt.device, dtype=xt.dtype)

    mask = mask.to(device=xt.device, dtype=xt.dtype).clone()  # [input p, output q]
    mask.fill_diagonal_(1.0)
    p_matrix = projector_matrix(U, rank).to(device=xt.device, dtype=xt.dtype)  # [q, p]

    # Row q is P_r[q, p] * M_q[p].  Since mask is stored as [p, q], we use mask.T.
    masked_projector = p_matrix * mask.T  # [output q, input p]
    masked_projected_xt = xt_flat @ masked_projector.T  # [B, N]

    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        candidates = images.to(xt.device, dtype=xt.dtype).flatten(start_dim=1)  # [K, N]
        masked_projected_candidates = candidates @ masked_projector.T  # [K, N]

        values = candidates + (t_tensor / (1.0 - t_tensor)) * masked_projected_candidates
        delta = (xt_flat.unsqueeze(1) - t_tensor * candidates.unsqueeze(0)) ** 2

        # mask is [input p, output q], so this is sum_p delta[b,k,p] * mask[p,q].
        masked_dist = torch.einsum("bkn,nm->bkm", delta, mask)
        logits = -masked_dist / (2.0 * noise_var * temperature)
        first_moment.add(values, logits)

    prediction = first_moment.get_average()
    if prediction is None:
        raise RuntimeError("No candidate images were available for asymmetric prediction.")

    prediction = prediction - masked_projected_xt / (1.0 - t_tensor)
    return prediction.view_as(xt)


@torch.no_grad()
def asym_uA_to_velocity(
    uA: torch.Tensor,
    xt: torch.Tensor,
    U: torch.Tensor,
    rank: int,
    t: float,
) -> torch.Tensor:
    """Convert an estimate of u_A into a full-rank FM velocity."""
    uA_flat = uA.flatten(start_dim=1)
    xt_flat = xt.flatten(start_dim=1)
    p_uA = project_flat(uA_flat, U, rank)
    p_xt = project_flat(xt_flat, U, rank)
    q_uA = uA_flat - p_uA
    q_xt = xt_flat - p_xt
    velocity = p_uA + (q_uA - q_xt) / (1.0 - t)
    return velocity.view_as(xt)


@torch.no_grad()
def predict_asym_artem_velocity(
    xt: torch.Tensor,
    t: float,
    candidate_dataset,
    mask: torch.Tensor,
    U: torch.Tensor,
    rank: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict u_A with the asymmetric mask, then convert it to velocity."""
    uA = predict_asym_artem_uA(
        xt=xt,
        t=t,
        candidate_dataset=candidate_dataset,
        mask=mask,
        U=U,
        rank=rank,
        temperature=temperature,
    )
    velocity = asym_uA_to_velocity(uA, xt, U, rank, t)
    return uA, velocity


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return scalar MSE as a Python float."""
    return torch.mean((prediction - target) ** 2).item()


def run_mse_experiment(
    model: torch.nn.Module,
    xt_dataset,
    candidate_dataset,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    ranks: list[int],
    t_values: list[float],
    num_xt: int,
    xt_batch_size: int,
    temperature: float,
    mask_threshold: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Compare the asymmetric Artem-like velocity with the FM U-Net velocity."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x1_all = collect_images(xt_dataset, num_xt, device)
    eps_all = torch.randn(x1_all.shape, device=device, generator=generator)
    results: list[dict[str, float | int]] = []

    for t in t_values:
        print(f"\n=== MSE t={t:.2f} ===")
        for rank in ranks:
            operator = build_wiener_asym_operator(U, eigenvalues, rank, t)
            mask = binarize_like_artem(operator, mask_threshold)
            mse_values: list[float] = []

            for start in range(0, num_xt, xt_batch_size):
                end = min(start + xt_batch_size, num_xt)
                x1 = x1_all[start:end]
                eps = eps_all[start:end]
                xt = t * x1 + (1.0 - t) * eps
                unet_velocity = predict_fm_unet_velocity(model, xt, t)
                _uA, velocity = predict_asym_artem_velocity(
                    xt=xt,
                    t=t,
                    candidate_dataset=candidate_dataset,
                    mask=mask,
                    U=U,
                    rank=rank,
                    temperature=temperature,
                )
                mse_values.append(mean_squared_error(velocity, unet_velocity))

            mse = sum(mse_values) / len(mse_values)
            row = {"t": t, "rank": rank, "mse_velocity_vs_fm_unet": mse}
            print(row)
            results.append(row)

    return results


def save_metrics(results: list[dict[str, float | int]], out_path: Path) -> None:
    """Save metrics as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def plot_mse(results: list[dict[str, float | int]], out_path: Path) -> None:
    """Plot MSE curves against the pretrained FM U-Net."""
    t_values = sorted({float(row["t"]) for row in results})
    ranks = sorted({int(row["rank"]) for row in results})

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rank in ranks:
        rows = {float(row["t"]): row for row in results if int(row["rank"]) == rank}
        values = [float(rows[t]["mse_velocity_vs_fm_unet"]) for t in t_values]
        ax.plot(t_values, values, marker="o", linewidth=2.0, label=f"rank={rank}")

    ax.set_title("FM U-Net velocity vs asym Artem-like mask")
    ax.set_xlabel("t")
    ax.set_ylabel("MSE")
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
    candidate_dataset,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    ranks: list[int],
    num_steps: int,
    temperature: float,
    mask_threshold: float,
) -> dict[str, torch.Tensor]:
    """Generate samples with the asymmetric Artem-like velocity estimator."""
    samples = {f"asym_artem r={rank}": initial_noise.clone() for rank in ranks}
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        for rank in ranks:
            key = f"asym_artem r={rank}"
            operator = build_wiener_asym_operator(U, eigenvalues, rank, t)
            mask = binarize_like_artem(operator, mask_threshold)
            _uA, velocity = predict_asym_artem_velocity(
                xt=samples[key],
                t=t,
                candidate_dataset=candidate_dataset,
                mask=mask,
                U=U,
                rank=rank,
                temperature=temperature,
            )
            samples[key] = samples[key] + step_size * velocity

    return samples


def save_generation_grid(samples: dict[str, torch.Tensor], postprocess, out_path: Path) -> None:
    """Save a grid with columns=ranks and rows=shared initial noises."""
    names = list(samples.keys())
    num_samples = next(iter(samples.values())).shape[0]
    fig, axes = plt.subplots(
        num_samples,
        len(names),
        figsize=(2.1 * len(names), 2.1 * num_samples),
        squeeze=False,
    )

    for col, name in enumerate(names):
        axes[0, col].set_title(name, fontsize=13)
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


def save_config(args: argparse.Namespace, ranks: list[int], generation_ranks: list[int], out_path: Path) -> None:
    """Save run metadata."""
    payload = vars(args).copy()
    payload["out_dir"] = str(payload["out_dir"]) if payload["out_dir"] is not None else None
    payload["ranks"] = ranks
    payload["generation_ranks"] = generation_ranks
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Run the asymmetric Artem-like mask experiment."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"asym_artem_mask_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_locality_config(args.dataset, args.batch_size, args.num_workers, subset_size=None)
    candidate_cfg = load_locality_config(
        args.dataset,
        args.batch_size,
        args.num_workers,
        subset_size=args.num_xi,
    )
    dataset = load_dataset(cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    U, eigenvalues, _Vh, _mean = load_or_compute_wiener_svd(dataset, device)

    pca_params = cfg.model.params or {}
    temperature = float(pca_params.get("temperature", 1.0))
    mask_threshold = float(pca_params.get("mask_threshold", 0.02))
    t_values = parse_t_values(args.t_values)
    ranks = parse_ranks(args.ranks, max_rank=U.shape[1])
    generation_ranks = parse_ranks(args.generation_ranks or args.ranks, max_rank=U.shape[1])
    xt_batch_size = args.xt_batch_size or int(cfg.sampling.batch_size)

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"num_xi: {args.num_xi}")
    print(f"num_xt: {args.num_xt}")
    print(f"xt_batch_size: {xt_batch_size}")
    print(f"temperature: {temperature}")
    print(f"mask_threshold: {mask_threshold}")
    print(f"ranks: {ranks}")
    print(f"generation_ranks: {generation_ranks}")
    print(f"out_dir: {out_dir}")

    if not args.skip_mse:
        checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)
        model = load_fm_unet(checkpoint_path, device, args.dataset)
        print(f"checkpoint: {checkpoint_path}")
        results = run_mse_experiment(
            model=model,
            xt_dataset=dataset,
            candidate_dataset=candidate_dataset,
            U=U,
            eigenvalues=eigenvalues,
            ranks=ranks,
            t_values=t_values,
            num_xt=args.num_xt,
            xt_batch_size=xt_batch_size,
            temperature=temperature,
            mask_threshold=mask_threshold,
            seed=args.seed,
            device=device,
        )
        save_metrics(results, out_dir / "metrics.json")
        plot_mse(results, out_dir / "mse.png")

    if not args.skip_generation:
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
            ranks=generation_ranks,
            num_steps=args.num_steps,
            temperature=temperature,
            mask_threshold=mask_threshold,
        )
        save_generation_grid(samples, dataset.postprocess, out_dir / "generated_samples.png")

    save_config(args, ranks, generation_ranks, out_dir / "config.json")
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
