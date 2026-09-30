"""Full and mask-restricted affine Wiener denoisers in RGB wavelet coordinates."""

from __future__ import annotations

import numpy as np
import torch

from wavelet_empirical_cifar.masks import site_sensitivity
from wavelet_score.mask_groups import WaveletMaskGroup


def empirical_moments(
    references: np.ndarray, *, device: torch.device, batch_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    if references.ndim != 2 or len(references) < 2:
        raise ValueError("references must be [N,d] with N >= 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mean = np.mean(references, axis=0, dtype=np.float64)
    dimension = references.shape[1]
    gram = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    for start in range(0, len(references), batch_size):
        batch = torch.as_tensor(
            np.ascontiguousarray(references[start : start + batch_size]),
            dtype=torch.float64,
            device=device,
        )
        batch -= torch.as_tensor(mean, dtype=torch.float64, device=device)
        gram.addmm_(batch.T, batch)
    covariance = (gram / len(references)).cpu().numpy()
    return mean, 0.5 * (covariance + covariance.T)


def full_denoiser_matrix(
    covariance: np.ndarray,
    time: float,
    *,
    device: torch.device,
) -> np.ndarray:
    if not 0.0 < time < 1.0 or covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("invalid covariance or flow time")
    matrix = torch.as_tensor(covariance, dtype=torch.float64, device=device)
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min_(0.0)
    gains = time * values / (time * time * values + (1.0 - time) ** 2)
    result = (vectors * gains.unsqueeze(0)) @ vectors.T
    return result.cpu().numpy()


def covariance_eigensystem(covariance: np.ndarray, device: torch.device):
    matrix = torch.as_tensor(covariance, dtype=torch.float64, device=device)
    values, vectors = torch.linalg.eigh(matrix)
    return values.clamp_min_(0.0), vectors


def denoiser_from_eigensystem(values, vectors, time: float) -> np.ndarray:
    if not 0.0 < time < 1.0:
        raise ValueError("time must be inside (0,1)")
    gains = time * values / (time*time*values + (1-time)**2)
    return ((vectors * gains.unsqueeze(0)) @ vectors.T).cpu().numpy()


def sensitivity_stability(
    first: np.ndarray, second: np.ndarray, scalar_dimension: int, budget: int,
    band_sizes: np.ndarray | None = None,
) -> float:
    """Mean selected-site overlap divided by budget for two fitted masks."""
    from wavelet_empirical_cifar.masks import top_sensitivity_sites

    scores_a = site_sensitivity(first, scalar_dimension)
    scores_b = site_sensitivity(second, scalar_dimension)
    overlaps = []
    for target in range(scalar_dimension):
        target_budget = min(budget, int(band_sizes[target])) if band_sizes is not None else budget
        left = set(top_sensitivity_sites(scores_a[target], target, target_budget))
        right = set(top_sensitivity_sites(scores_b[target], target, target_budget))
        overlaps.append(len(left & right) / target_budget)
    return float(np.mean(overlaps))


def restricted_wiener_clean(
    queries: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    time: float,
    groups: tuple[WaveletMaskGroup, ...],
    *,
    device: torch.device,
    group_batch_size: int = 256,
) -> np.ndarray:
    """Re-fit the affine conditional mean on each mask's covariance submatrix."""
    if queries.ndim != 2 or queries.shape[1] != len(mean):
        raise ValueError("queries do not match the fitted mean")
    if covariance.shape != (len(mean), len(mean)) or not 0 <= time < 1:
        raise ValueError("invalid covariance or time")
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be positive")

    outputs = np.empty_like(queries, dtype=np.float64)
    buckets: dict[tuple[int, int], list[WaveletMaskGroup]] = {}
    for group in groups:
        buckets.setdefault((len(group.input_indices), len(group.output_indices)), []).append(group)

    covariance_t = torch.as_tensor(covariance, dtype=torch.float64, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float64, device=device)
    query_t = torch.as_tensor(np.ascontiguousarray(queries), dtype=torch.float64, device=device)
    for bucket in buckets.values():
        for start in range(0, len(bucket), group_batch_size):
            selected = bucket[start : start + group_batch_size]
            inputs = torch.as_tensor(
                np.stack([group.input_indices for group in selected]), device=device
            )
            targets = torch.as_tensor(
                np.stack([group.output_indices for group in selected]), device=device
            )
            observed_covariance = covariance_t[inputs[:, :, None], inputs[:, None, :]]
            cross_covariance = covariance_t[inputs[:, :, None], targets[:, None, :]]
            normal = time * time * observed_covariance
            normal.diagonal(dim1=-2, dim2=-1).add_((1.0 - time) ** 2)
            rhs = time * cross_covariance
            coefficients = torch.linalg.solve(normal, rhs)
            centered_queries = query_t[:, inputs] - time * mean_t[inputs]
            predictions = mean_t[targets].unsqueeze(0) + torch.einsum(
                "bgi,gio->bgo", centered_queries, coefficients
            )
            index = targets.reshape(-1).cpu().numpy()
            outputs[:, index] = predictions.reshape(len(queries), -1).cpu().numpy()
    return outputs
