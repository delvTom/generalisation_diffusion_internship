"""Generate samples with the exact local kappa closed-form and Artem masks.

This script studies the locality-constrained closed-form associated with the
kappa spectral probability path. The locality sets are fixed by Artem's
classical clean-image Wiener mask: for each output pixel q, we extract the
coordinates selected by the x1-mask M_t^q.

Unlike the standard FM path, the spectral path has covariance

    Gamma_t = U diag(alpha_i(t)^2) U^T,

which is not diagonal in pixel space for kappa != 1. Therefore the local
posterior weights use the exact local Mahalanobis covariance

    C_t^q = S_t^q Gamma_t (S_t^q)^T,

and the local target uses the exact conditional mean of x_t given the observed
coordinates S_t^q x_t. The extractor S_t^q is implemented by the selected
coordinate indices of Artem's binary mask; it is never materialized as a dense
matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from omegaconf import read_write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
EXPERIMENTS_ROOT = PROJECT_ROOT / "fm_comparisons"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (  # noqa: E402
    binarize_like_artem,
    build_fm_wiener_operator,
    build_mask_operator,
    dataset_output_slug,
    load_dataset,
    load_experiment_config,
    load_or_compute_wiener_svd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--kappas", default="-2,-1,-0.5,0,0.5,1,1.5,2")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--q-chunk-size", type=int, default=16)
    parser.add_argument("--cholesky-jitter", type=float, default=1e-5)
    parser.add_argument(
        "--neighbor-t",
        type=float,
        default=0.98,
        help="Euler evaluation time used to inspect local posterior neighbors.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_kappas(value: str) -> list[float]:
    """Parse comma-separated kappa values."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_path_config(dataset_name: str, batch_size: int | None, subset_size: int | None, num_workers: int | None):
    """Load Artem's PCA-locality config with local overrides."""
    cfg = load_experiment_config(dataset_name, batch_size, subset_size)
    if num_workers is not None:
        with read_write(cfg):
            cfg.dataset.num_workers = num_workers
    return cfg


def spectral_stds(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return lambda_i from covariance eigenvalues lambda_i^2."""
    return torch.sqrt(torch.clamp(eigenvalues, min=eps))


def geometric_mean(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return exp(mean(log(values))) in a numerically stable way."""
    return torch.exp(torch.log(torch.clamp(values, min=eps)).mean())


def kappa_schedule(
    t: float,
    kappa: float,
    lambdas: torch.Tensor,
    lambda_star: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return beta_i(t), alpha_i(t), and beta_dot_i(t).

    Shapes:
        lambdas: [N]
        beta: [N]
        alpha: [N]
        beta_dot: [N]
    """
    t_tensor = torch.tensor(t, device=lambdas.device, dtype=lambdas.dtype)
    ratio = torch.clamp(lambdas / lambda_star, min=1e-12)
    r = ratio ** (kappa - 1.0)
    denom = 1.0 - t_tensor + t_tensor * r
    beta = t_tensor * r / denom
    alpha = (1.0 - t_tensor) / denom
    beta_dot = r / (denom * denom)
    return beta, alpha, beta_dot


def stable_cholesky(matrix: torch.Tensor, jitter: float, max_tries: int = 6) -> torch.Tensor:
    """Compute a Cholesky factor with trace-scaled diagonal jitter."""
    dim = matrix.shape[0]
    eye = torch.eye(dim, device=matrix.device, dtype=matrix.dtype)
    scale = torch.diagonal(matrix).mean().abs().clamp_min(1e-12)
    current_jitter = jitter
    for _ in range(max_tries):
        regularized = matrix + current_jitter * scale * eye
        factor, info = torch.linalg.cholesky_ex(regularized)
        if int(info.item()) == 0:
            return factor
        current_jitter *= 10.0
    return torch.linalg.cholesky(matrix + current_jitter * scale * eye)


def solve_cholesky(rhs: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """Return C^{-1} rhs from a Cholesky factor C=LL^T."""
    return torch.cholesky_solve(rhs, chol)


def quadratic_form_cholesky(residual: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    """Compute residual^T C^{-1} residual for residual shape [B, K, R]."""
    batch, candidates, local_dim = residual.shape
    flat = residual.reshape(batch * candidates, local_dim)
    solved = solve_cholesky(flat.T, chol).T
    return (flat * solved).sum(dim=1).view(batch, candidates)


@torch.no_grad()
def precompute_candidate_path_terms(
    candidate_spectral: torch.Tensor,
    U: torch.Tensor,
    beta: torch.Tensor,
    beta_dot: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute B_t x_i and dot(B_t) x_i for every candidate image.

    Both pixel-space quantities are reused for every output coordinate q. This
    avoids repeating the same spectral projections once per local mask.

    Returns:
        candidate_means: B_t x_i, shape [K, N].
        candidate_targets: dot(B_t) x_i, shape [K, N].
    """
    candidate_means = torch.empty_like(candidate_spectral)
    candidate_targets = torch.empty_like(candidate_spectral)
    pixel_basis = U.T

    for start in range(0, candidate_spectral.shape[0], batch_size):
        end = min(start + batch_size, candidate_spectral.shape[0])
        z_candidates = candidate_spectral[start:end]  # [K_batch, N]
        candidate_means[start:end] = (z_candidates * beta.unsqueeze(0)) @ pixel_basis
        candidate_targets[start:end] = (
            z_candidates * beta_dot.unsqueeze(0)
        ) @ pixel_basis

    return candidate_means, candidate_targets


@torch.no_grad()
def collect_candidate_spectral_coordinates(
    candidate_dataset,
    U: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Collect candidate images once in PCA coordinates.

    Returns:
        z_candidates: tensor of shape [K, N], where K is the number of
            candidate training images and N is the flattened image dimension.
    """
    chunks: list[torch.Tensor] = []
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        candidates = images.to(device=device, dtype=dtype).flatten(start_dim=1)  # [K, N]
        chunks.append(candidates @ U)  # [K, N]

    if not chunks:
        raise RuntimeError("Candidate dataloader did not yield any images.")

    return torch.cat(chunks, dim=0)


def build_artem_x1_mask_indices(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    t: float,
    threshold: float,
) -> list[torch.Tensor]:
    """Build coordinate extractors from Artem's clean-image binary mask.

    The returned list contains one tensor I_q per output coordinate q. Each
    tensor stores the input coordinates selected by the mask M_t^q.
    """
    wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
    mask = binarize_like_artem(
        build_mask_operator(wiener_operator, t, "x1_mask"),
        threshold,
    )  # [input p, output q]
    diagonal = torch.arange(mask.shape[0], device=mask.device)
    mask[diagonal, diagonal] = 1.0
    return [
        torch.nonzero(mask[:, q], as_tuple=False).flatten().long()
        for q in range(mask.shape[1])
    ]


class LocalStreamingSoftmax:
    """Streaming softmax for local logits and local scalar values."""

    def __init__(self, batch_size: int, num_outputs: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_logits = torch.full(
            (batch_size, num_outputs),
            -float("inf"),
            device=device,
            dtype=dtype,
        )
        self.sum_weights = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)
        self.sum_weighted = torch.zeros((batch_size, num_outputs), device=device, dtype=dtype)

    def add_column(self, column: int, logits: torch.Tensor, values: torch.Tensor) -> None:
        """Add logits and values for one output coordinate in the current chunk.

        Args:
            column: local output-column index in the q chunk.
            logits: posterior logits, shape [B, K].
            values: local target values, shape [B, K].
        """
        chunk_max = logits.max(dim=1).values
        new_max = torch.maximum(self.max_logits[:, column], chunk_max)
        old_scale = torch.exp(self.max_logits[:, column] - new_max)
        weights = torch.exp(logits - new_max.unsqueeze(1))

        self.sum_weights[:, column] = (
            self.sum_weights[:, column] * old_scale + weights.sum(dim=1)
        )
        self.sum_weighted[:, column] = (
            self.sum_weighted[:, column] * old_scale + (weights * values).sum(dim=1)
        )
        self.max_logits[:, column] = new_max

    def average(self) -> torch.Tensor:
        """Return the local weighted averages, shape [B, Q]."""
        return self.sum_weighted / self.sum_weights.clamp_min(1e-12)


@torch.no_grad()
def predict_local_kappa_velocity(
    xt: torch.Tensor,
    t: float,
    kappa: float,
    candidate_spectral: torch.Tensor,
    candidate_batch_size: int,
    mask_indices: list[torch.Tensor],
    U: torch.Tensor,
    lambdas: torch.Tensor,
    lambda_star: torch.Tensor,
    q_chunk_size: int,
    temperature: float,
    cholesky_jitter: float,
    collect_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Predict the exact local kappa closed-form velocity.

    For each output coordinate q, I_q is the support of Artem's x1-mask. The
    posterior weights use C_q = Gamma[I_q, I_q], and the local target is

        e_q^T G (x_i - E[x_t | S_q x_t = S_q x, x_i]).
    """
    xt_shape = xt.shape
    xt_flat = xt.flatten(start_dim=1)  # [B, N]
    batch_size, num_pixels = xt_flat.shape
    beta, alpha, beta_dot = kappa_schedule(t, kappa, lambdas, lambda_star)

    print("    precomputing candidate path terms", flush=True)
    candidate_means, candidate_targets = precompute_candidate_path_terms(
        candidate_spectral=candidate_spectral,
        U=U,
        beta=beta,
        beta_dot=beta_dot,
        batch_size=candidate_batch_size,
    )

    gamma = (U * alpha.square().unsqueeze(0)) @ U.T  # [N, N]
    output = torch.empty_like(xt_flat)
    if collect_diagnostics:
        num_candidates = candidate_spectral.shape[0]
        aggregate_distances = torch.zeros(
            (batch_size, num_candidates),
            device=xt.device,
            dtype=xt.dtype,
        )
        local_best_distances = torch.full_like(xt_flat, float("inf"))
        local_best_indices = torch.full(
            xt_flat.shape,
            -1,
            device=xt.device,
            dtype=torch.long,
        )
        local_max_weights = torch.empty_like(xt_flat)
    else:
        aggregate_distances = None
        local_best_distances = None
        local_best_indices = None
        local_max_weights = None

    for q_start in range(0, num_pixels, q_chunk_size):
        q_end = min(q_start + q_chunk_size, num_pixels)
        q_values = list(range(q_start, q_end))
        stream = LocalStreamingSoftmax(
            batch_size=batch_size,
            num_outputs=len(q_values),
            device=xt.device,
            dtype=xt.dtype,
        )

        local_data: list[dict[str, Any]] = []
        for q in q_values:
            indices = mask_indices[q]
            local_gamma = gamma.index_select(0, indices).index_select(1, indices)
            chol = stable_cholesky(local_gamma, jitter=cholesky_jitter)

            # e_q^T G Gamma[:, I_q] C_q^{-1}
            cross = (U[q] * beta_dot * alpha) @ U.index_select(0, indices).T
            local_a = solve_cholesky(cross.unsqueeze(1), chol).squeeze(1)
            local_data.append(
                {
                    "q": q,
                    "indices": indices,
                    "chol": chol,
                    "a": local_a,
                    "xt_local": xt_flat.index_select(1, indices),
                }
            )

        for start in range(0, candidate_means.shape[0], candidate_batch_size):
            end = min(start + candidate_batch_size, candidate_means.shape[0])
            mean_chunk = candidate_means[start:end]  # [K, N]
            target_chunk = candidate_targets[start:end, q_start:q_end]  # [K, Q]

            for local_q, data in enumerate(local_data):
                mu_local = mean_chunk.index_select(1, data["indices"])  # [K, R]
                residual = data["xt_local"].unsqueeze(1) - mu_local.unsqueeze(0)  # [B, K, R]
                distances = quadratic_form_cholesky(residual, data["chol"])
                logits = -0.5 * distances / temperature

                correction = torch.einsum("bkr,r->bk", residual, data["a"])
                values = target_chunk[:, local_q].unsqueeze(0) - correction
                stream.add_column(local_q, logits, values)

                if collect_diagnostics:
                    assert aggregate_distances is not None
                    assert local_best_distances is not None
                    assert local_best_indices is not None
                    q = q_start + local_q
                    aggregate_distances[:, start:end] += distances
                    chunk_distances, chunk_offsets = distances.min(dim=1)
                    improved = chunk_distances < local_best_distances[:, q]
                    local_best_distances[improved, q] = chunk_distances[improved]
                    local_best_indices[improved, q] = start + chunk_offsets[improved]

        output[:, q_start:q_end] = stream.average()
        if collect_diagnostics:
            assert local_max_weights is not None
            local_max_weights[:, q_start:q_end] = (
                1.0 / stream.sum_weights.clamp_min(1e-12)
            )
        print(
            f"    q {q_start:05d}-{q_end - 1:05d} / {num_pixels - 1:05d}",
            flush=True,
        )

    diagnostics = None
    if collect_diagnostics:
        assert aggregate_distances is not None
        assert local_best_distances is not None
        assert local_best_indices is not None
        assert local_max_weights is not None
        diagnostics = {
            "aggregate_distances": aggregate_distances / num_pixels,
            "local_best_distances": local_best_distances,
            "local_best_indices": local_best_indices,
            "local_max_weights": local_max_weights,
        }

    return output.view(xt_shape), diagnostics


@torch.no_grad()
def euler_generate(
    initial_noise: torch.Tensor,
    kappas: list[float],
    candidate_spectral: torch.Tensor,
    candidate_batch_size: int,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    Vh: torch.Tensor,
    num_steps: int,
    mask_threshold: float,
    q_chunk_size: int,
    temperature: float,
    cholesky_jitter: float,
    neighbor_t: float,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]], float]:
    """Run Euler generation and inspect posteriors at the closest requested time."""
    lambdas = spectral_stds(eigenvalues).to(device=initial_noise.device, dtype=initial_noise.dtype)
    lambda_star = geometric_mean(lambdas)
    samples = {f"kappa={kappa:g}": initial_noise.clone() for kappa in kappas}
    posterior_diagnostics: dict[str, dict[str, torch.Tensor]] = {}
    step_size = 1.0 / num_steps
    neighbor_step = min(
        range(num_steps),
        key=lambda step: abs((step + 0.5) / num_steps - neighbor_t),
    )
    resolved_neighbor_t = (neighbor_step + 0.5) / num_steps

    print(f"lambda_star: {lambda_star.item():.6g}")
    print(
        f"posterior diagnostic: step={neighbor_step + 1}, "
        f"t={resolved_neighbor_t:.6f}"
    )
    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        mask_indices = build_artem_x1_mask_indices(
            U=U,
            eigenvalues=eigenvalues,
            Vh=Vh,
            t=t,
            threshold=mask_threshold,
        )
        sizes = torch.tensor([idx.numel() for idx in mask_indices], device="cpu")
        print(
            "  mask sizes | "
            f"min={int(sizes.min())} median={float(sizes.float().median()):.1f} "
            f"max={int(sizes.max())}",
            flush=True,
        )

        for kappa in kappas:
            key = f"kappa={kappa:g}"
            print(f"  predicting {key}")
            velocity, diagnostics = predict_local_kappa_velocity(
                xt=samples[key],
                t=t,
                kappa=kappa,
                candidate_spectral=candidate_spectral,
                candidate_batch_size=candidate_batch_size,
                mask_indices=mask_indices,
                U=U,
                lambdas=lambdas,
                lambda_star=lambda_star,
                q_chunk_size=q_chunk_size,
                temperature=temperature,
                cholesky_jitter=cholesky_jitter,
                collect_diagnostics=step == neighbor_step,
            )
            if diagnostics is not None:
                posterior_diagnostics[key] = diagnostics
            samples[key] = samples[key] + step_size * velocity

    return samples, posterior_diagnostics, resolved_neighbor_t


def save_final_grid(samples: dict[str, torch.Tensor], postprocess, out_path: Path) -> None:
    """Save a grid with rows=samples and columns=kappa values."""
    num_samples = next(iter(samples.values())).shape[0]
    keys = list(samples.keys())
    fig, axes = plt.subplots(
        num_samples,
        len(keys),
        figsize=(2.1 * len(keys), 2.1 * num_samples),
        squeeze=False,
    )

    for col, key in enumerate(keys):
        axes[0, col].set_title(key, fontsize=15)
        images = postprocess(samples[key].detach().cpu())
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


@torch.no_grad()
def find_pixel_nearest_neighbors(
    samples: dict[str, torch.Tensor],
    candidate_dataset,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Find each final sample's nearest candidate in normalized pixel L2."""
    keys = list(samples)
    batch_size = samples[keys[0]].shape[0]
    image_shape = samples[keys[0]].shape[1:]
    queries = torch.cat(
        [samples[key].flatten(start_dim=1) for key in keys],
        dim=0,
    )  # [num_kappas * B, N]
    query_norms = queries.square().sum(dim=1, keepdim=True)
    best_distances = torch.full(
        (queries.shape[0],),
        float("inf"),
        device=queries.device,
        dtype=queries.dtype,
    )
    best_indices = torch.full(
        (queries.shape[0],),
        -1,
        device=queries.device,
        dtype=torch.long,
    )
    best_neighbors = torch.empty_like(queries)

    dataset_offset = 0
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        candidates = images.to(
            device=queries.device,
            dtype=queries.dtype,
        ).flatten(start_dim=1)
        candidate_norms = candidates.square().sum(dim=1).unsqueeze(0)
        distances = query_norms + candidate_norms - 2.0 * (queries @ candidates.T)
        chunk_distances, chunk_offsets = distances.min(dim=1)
        improved = chunk_distances < best_distances
        if improved.any():
            best_distances[improved] = chunk_distances[improved]
            best_indices[improved] = dataset_offset + chunk_offsets[improved]
            best_neighbors[improved] = candidates[chunk_offsets[improved]]
        dataset_offset += candidates.shape[0]

    neighbors: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    indices: dict[str, torch.Tensor] = {}
    for key_index, key in enumerate(keys):
        start = key_index * batch_size
        end = start + batch_size
        neighbors[key] = best_neighbors[start:end].view(batch_size, *image_shape)
        distances[key] = best_distances[start:end].clamp_min(0.0)
        indices[key] = best_indices[start:end]
    return neighbors, distances, indices


@torch.no_grad()
def reconstruct_candidates(
    candidate_spectral: torch.Tensor,
    indices: torch.Tensor,
    U: torch.Tensor,
    image_shape: torch.Size,
) -> torch.Tensor:
    """Reconstruct complete candidate images selected by one index per sample."""
    flat = candidate_spectral.index_select(0, indices) @ U.T
    return flat.view(indices.shape[0], *image_shape)


@torch.no_grad()
def reconstruct_local_candidate_mosaic(
    candidate_spectral: torch.Tensor,
    local_indices: torch.Tensor,
    U: torch.Tensor,
    image_shape: torch.Size,
    q_chunk_size: int,
) -> torch.Tensor:
    """Use the candidate selected by each q to reconstruct its output coordinate."""
    batch_size, num_pixels = local_indices.shape
    flat = torch.empty(
        (batch_size, num_pixels),
        device=candidate_spectral.device,
        dtype=candidate_spectral.dtype,
    )
    for q_start in range(0, num_pixels, q_chunk_size):
        q_end = min(q_start + q_chunk_size, num_pixels)
        selected = candidate_spectral[local_indices[:, q_start:q_end]]  # [B, Q, N]
        flat[:, q_start:q_end] = (
            selected * U[q_start:q_end].unsqueeze(0)
        ).sum(dim=2)
    return flat.view(batch_size, *image_shape)


@torch.no_grad()
def summarize_posterior_diagnostics(
    samples: dict[str, torch.Tensor],
    diagnostics: dict[str, dict[str, torch.Tensor]],
    candidate_spectral: torch.Tensor,
    U: torch.Tensor,
    q_chunk_size: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, list[dict[str, Any]]]]:
    """Materialize posterior neighbors and summarize local candidate selections."""
    visuals: dict[str, dict[str, torch.Tensor]] = {}
    records: dict[str, list[dict[str, Any]]] = {}

    for key, sample_diagnostics in diagnostics.items():
        generated = samples[key]
        image_shape = generated.shape[1:]
        aggregate_distances = sample_diagnostics["aggregate_distances"]
        local_indices = sample_diagnostics["local_best_indices"]
        local_max_weights = sample_diagnostics["local_max_weights"]

        aggregate_values, aggregate_indices = aggregate_distances.min(dim=1)
        aggregate_neighbors = reconstruct_candidates(
            candidate_spectral,
            aggregate_indices,
            U,
            image_shape,
        )

        modal_indices: list[torch.Tensor] = []
        modal_counts: list[int] = []
        unique_counts: list[int] = []
        top_candidates: list[list[dict[str, Any]]] = []
        for sample_indices in local_indices:
            values, counts = torch.unique(sample_indices, return_counts=True)
            order = torch.argsort(counts, descending=True)
            values = values[order]
            counts = counts[order]
            modal_indices.append(values[0])
            modal_counts.append(int(counts[0].item()))
            unique_counts.append(int(values.numel()))
            top_candidates.append(
                [
                    {
                        "index": int(candidate_index.item()),
                        "fraction": float(count.item() / sample_indices.numel()),
                    }
                    for candidate_index, count in zip(values[:5], counts[:5])
                ]
            )

        modal_indices_tensor = torch.stack(modal_indices)
        modal_neighbors = reconstruct_candidates(
            candidate_spectral,
            modal_indices_tensor,
            U,
            image_shape,
        )
        local_mosaic = reconstruct_local_candidate_mosaic(
            candidate_spectral,
            local_indices,
            U,
            image_shape,
            q_chunk_size=q_chunk_size,
        )
        agreement = (
            local_indices
            == modal_indices_tensor.unsqueeze(1)
        ).view(generated.shape[0], *image_shape).float().mean(dim=1)

        visuals[key] = {
            "aggregate_neighbor": aggregate_neighbors,
            "modal_neighbor": modal_neighbors,
            "local_mosaic": local_mosaic,
            "agreement": agreement,
        }

        key_records: list[dict[str, Any]] = []
        for sample_index in range(generated.shape[0]):
            key_records.append(
                {
                    "sample": sample_index,
                    "aggregate_neighbor_index": int(
                        aggregate_indices[sample_index].item()
                    ),
                    "aggregate_mean_distance": float(
                        aggregate_values[sample_index].item()
                    ),
                    "modal_local_candidate_index": int(
                        modal_indices_tensor[sample_index].item()
                    ),
                    "modal_local_fraction": float(
                        modal_counts[sample_index] / local_indices.shape[1]
                    ),
                    "num_unique_local_candidates": unique_counts[sample_index],
                    "mean_local_max_weight": float(
                        local_max_weights[sample_index].mean().item()
                    ),
                    "generated_to_aggregate_mse": float(
                        (
                            generated[sample_index]
                            - aggregate_neighbors[sample_index]
                        ).square().mean().item()
                    ),
                    "generated_to_modal_mse": float(
                        (
                            generated[sample_index]
                            - modal_neighbors[sample_index]
                        ).square().mean().item()
                    ),
                    "generated_to_local_mosaic_mse": float(
                        (
                            generated[sample_index]
                            - local_mosaic[sample_index]
                        ).square().mean().item()
                    ),
                    "top_local_candidates": top_candidates[sample_index],
                }
            )
        records[key] = key_records

    return visuals, records


def save_posterior_neighbor_grid(
    samples: dict[str, torch.Tensor],
    visuals: dict[str, dict[str, torch.Tensor]],
    postprocess,
    posterior_t: float,
    out_path: Path,
) -> None:
    """Save generated images beside their aggregate posterior neighbors."""
    keys = list(samples.keys())
    num_samples = next(iter(samples.values())).shape[0]
    fig, axes = plt.subplots(
        num_samples,
        2 * len(keys),
        figsize=(3.6 * len(keys), 1.8 * num_samples),
        squeeze=False,
    )
    for key_index, key in enumerate(keys):
        generated = postprocess(samples[key].detach().cpu())
        neighbors = postprocess(visuals[key]["aggregate_neighbor"].detach().cpu())
        for sample_index in range(num_samples):
            for offset, (images, label) in enumerate(
                ((generated, f"{key}\ngen"), (neighbors, "posterior\nneighbor"))
            ):
                ax = axes[sample_index, 2 * key_index + offset]
                ax.imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
                if sample_index == 0:
                    ax.set_title(label, fontsize=12)

    fig.suptitle(f"Local posterior neighbors at t={posterior_t:.4f}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_pixel_neighbor_grid(
    samples: dict[str, torch.Tensor],
    neighbors: dict[str, torch.Tensor],
    distances: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
) -> None:
    """Save final generations beside their nearest dataset images in pixel L2."""
    keys = list(samples)
    num_samples = next(iter(samples.values())).shape[0]
    num_pixels = next(iter(samples.values()))[0].numel()
    fig, axes = plt.subplots(
        num_samples,
        2 * len(keys),
        figsize=(3.6 * len(keys), 1.8 * num_samples),
        squeeze=False,
    )
    for key_index, key in enumerate(keys):
        generated = postprocess(samples[key].detach().cpu())
        nearest = postprocess(neighbors[key].detach().cpu())
        for sample_index in range(num_samples):
            for offset, images in enumerate((generated, nearest)):
                ax = axes[sample_index, 2 * key_index + offset]
                ax.imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
            if sample_index == 0:
                axes[sample_index, 2 * key_index].set_title(
                    f"{key}\ngen",
                    fontsize=12,
                )
                axes[sample_index, 2 * key_index + 1].set_title(
                    "pixel-L2\nneighbor",
                    fontsize=12,
                )
            pixel_mse = distances[key][sample_index].item() / num_pixels
            axes[sample_index, 2 * key_index + 1].set_xlabel(
                f"MSE={pixel_mse:.3g}",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_local_diagnostic_grids(
    samples: dict[str, torch.Tensor],
    visuals: dict[str, dict[str, torch.Tensor]],
    records: dict[str, list[dict[str, Any]]],
    postprocess,
    posterior_t: float,
    out_dir: Path,
) -> None:
    """Save one detailed local-posterior diagnostic grid per kappa."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, generated_tensor in samples.items():
        generated = postprocess(generated_tensor.detach().cpu())
        aggregate = postprocess(
            visuals[key]["aggregate_neighbor"].detach().cpu()
        )
        modal = postprocess(visuals[key]["modal_neighbor"].detach().cpu())
        mosaic = postprocess(visuals[key]["local_mosaic"].detach().cpu())
        agreement = visuals[key]["agreement"].detach().cpu()
        num_samples = generated.shape[0]
        fig, axes = plt.subplots(
            num_samples,
            5,
            figsize=(9.5, 1.9 * num_samples),
            squeeze=False,
        )
        titles = (
            "generated",
            "aggregate posterior NN",
            "modal local candidate",
            "local candidate mosaic",
            "modal agreement",
        )
        for sample_index in range(num_samples):
            image_columns = (
                generated[sample_index],
                aggregate[sample_index],
                modal[sample_index],
                mosaic[sample_index],
            )
            for column, image in enumerate(image_columns):
                axes[sample_index, column].imshow(
                    image.permute(1, 2, 0).numpy()
                )
            axes[sample_index, 4].imshow(
                agreement[sample_index].numpy(),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            fraction = records[key][sample_index]["modal_local_fraction"]
            axes[sample_index, 0].set_ylabel(
                f"sample {sample_index}\nmodal={fraction:.1%}",
                fontsize=9,
            )
            for column in range(5):
                axes[sample_index, column].set_xticks([])
                axes[sample_index, column].set_yticks([])
                if sample_index == 0:
                    axes[sample_index, column].set_title(
                        titles[column],
                        fontsize=11,
                    )

        fig.suptitle(f"{key}: local posteriors at t={posterior_t:.4f}", fontsize=14)
        fig.tight_layout()
        filename = (
            key.replace("=", "_")
            .replace("-", "minus_")
            .replace(".", "p")
        )
        fig.savefig(out_dir / f"{filename}.png", dpi=250)
        plt.close(fig)


def save_config(args: argparse.Namespace, out_path: Path, extra: dict[str, Any]) -> None:
    """Save generation arguments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload.update(extra)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.neighbor_t < 1.0:
        raise ValueError("--neighbor-t must be strictly between 0 and 1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kappas = parse_kappas(args.kappas)
    out_dir = args.out_dir or Path("spectral_paths/results") / (
        f"kappa_local_artem_generation_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_path_config(args.dataset, args.batch_size, subset_size=None, num_workers=args.num_workers)
    candidate_cfg = load_path_config(
        args.dataset,
        args.batch_size,
        subset_size=args.num_xi,
        num_workers=args.num_workers,
    )
    dataset = load_dataset(cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    # The spectral path uses full-dataset PCA statistics even when --num-xi
    # restricts the empirical mixture used by the closed-form estimator.
    U, eigenvalues, Vh, _mean = load_or_compute_wiener_svd(dataset, device)
    candidate_spectral = collect_candidate_spectral_coordinates(
        candidate_dataset=candidate_dataset,
        U=U,
        device=device,
        dtype=U.dtype,
    )
    candidate_batch_size = int(candidate_cfg.dataset.batch_size)

    pca_params = cfg.model.params or {}
    mask_threshold = (
        args.mask_threshold
        if args.mask_threshold is not None
        else float(pca_params.get("mask_threshold", 0.02))
    )

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"kappas: {kappas}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"num_xi: {args.num_xi}")
    print(f"batch_size: {cfg.dataset.batch_size}")
    print(f"candidate_spectral_shape: {tuple(candidate_spectral.shape)}")
    print(f"num_workers: {cfg.dataset.num_workers}")
    print(f"temperature: {args.temperature}")
    print(f"mask_threshold: {mask_threshold}")
    print(f"q_chunk_size: {args.q_chunk_size}")
    print(f"cholesky_jitter: {args.cholesky_jitter}")
    print(f"neighbor_t: {args.neighbor_t}")
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

    samples, posterior_diagnostics, resolved_neighbor_t = euler_generate(
        initial_noise=initial_noise,
        kappas=kappas,
        candidate_spectral=candidate_spectral,
        candidate_batch_size=candidate_batch_size,
        U=U,
        eigenvalues=eigenvalues,
        Vh=Vh,
        num_steps=args.num_steps,
        mask_threshold=mask_threshold,
        q_chunk_size=args.q_chunk_size,
        temperature=args.temperature,
        cholesky_jitter=args.cholesky_jitter,
        neighbor_t=args.neighbor_t,
    )

    posterior_visuals, posterior_records = summarize_posterior_diagnostics(
        samples=samples,
        diagnostics=posterior_diagnostics,
        candidate_spectral=candidate_spectral,
        U=U,
        q_chunk_size=args.q_chunk_size,
    )
    pixel_neighbors, pixel_distances, pixel_indices = find_pixel_nearest_neighbors(
        samples=samples,
        candidate_dataset=candidate_dataset,
    )
    for key, key_records in posterior_records.items():
        num_pixels = samples[key][0].numel()
        for sample_index, record in enumerate(key_records):
            record["pixel_neighbor_index"] = int(
                pixel_indices[key][sample_index].item()
            )
            record["pixel_squared_l2"] = float(
                pixel_distances[key][sample_index].item()
            )
            record["pixel_mse"] = float(
                pixel_distances[key][sample_index].item() / num_pixels
            )

    save_final_grid(samples, dataset.postprocess, out_dir / "final_samples.png")
    save_pixel_neighbor_grid(
        samples=samples,
        neighbors=pixel_neighbors,
        distances=pixel_distances,
        postprocess=dataset.postprocess,
        out_path=out_dir / "pixel_l2_neighbors.png",
    )
    save_posterior_neighbor_grid(
        samples=samples,
        visuals=posterior_visuals,
        postprocess=dataset.postprocess,
        posterior_t=resolved_neighbor_t,
        out_path=out_dir / "posterior_neighbors.png",
    )
    save_local_diagnostic_grids(
        samples=samples,
        visuals=posterior_visuals,
        records=posterior_records,
        postprocess=dataset.postprocess,
        posterior_t=resolved_neighbor_t,
        out_dir=out_dir / "local_posterior_diagnostics",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "posterior_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "posterior_t": resolved_neighbor_t,
                "aggregate_distance": "mean_q d_{i,q}(x_t,t)",
                "local_selection": "argmin_i d_{i,q}(x_t,t)",
                "results": posterior_records,
            },
            f,
            indent=2,
        )
    torch.save(
        {key: value.detach().cpu() for key, value in samples.items()},
        out_dir / "final_samples.pt",
    )
    torch.save(
        {
            "posterior_t": resolved_neighbor_t,
            "diagnostics": {
                key: {
                    name: tensor.detach().cpu()
                    for name, tensor in values.items()
                }
                for key, values in posterior_diagnostics.items()
            },
        },
        out_dir / "posterior_diagnostics.pt",
    )
    save_config(
        args,
        out_dir / "generation_config.json",
        extra={
            "resolved_mask_threshold": mask_threshold,
            "candidate_spectral_shape": list(candidate_spectral.shape),
            "candidate_batch_size": candidate_batch_size,
            "resolved_neighbor_t": resolved_neighbor_t,
        },
    )
    print(f"saved plot to {out_dir / 'final_samples.png'}")
    print(f"saved posterior diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
