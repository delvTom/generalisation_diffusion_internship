"""Empirical and optimal-affine velocity baselines for standard Flow Matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class WaveletWienerVelocity:
    """Optimal affine predictor E[V_Y | Y_t] from clean wavelet moments."""

    mean: torch.Tensor  # [d]
    eigenvectors: torch.Tensor  # [d,d], columns are eigenvectors
    eigenvalues: torch.Tensor  # [d]

    @classmethod
    def fit(
        cls,
        Y_0: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> WaveletWienerVelocity:
        if not isinstance(Y_0, np.ndarray) or Y_0.ndim != 2 or len(Y_0) < 2:
            raise ValueError("Y_0 must have shape [N,d] with N >= 2.")
        values = torch.as_tensor(Y_0, device=device, dtype=dtype)
        mean = values.mean(dim=0)
        centered = values - mean
        # Match fm_comparisons/src/local_diffusion Wiener caches exactly.
        covariance = centered.T @ centered / (len(values) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        return cls(
            mean=mean,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues.clamp_min(0.0),
        )

    def predict(self, Y_t: torch.Tensor, t: float) -> torch.Tensor:
        """Evaluate mu + Cov(V,Y_t) Cov(Y_t)^-1 (Y_t-t*mu)."""
        if Y_t.ndim != 2 or Y_t.shape[1] != self.mean.numel():
            raise ValueError("Y_t must have shape [N,d] compatible with the fit.")
        if not 0.0 <= t < 1.0:
            raise ValueError("Wiener velocity evaluation requires t in [0,1).")
        t_tensor = torch.as_tensor(t, device=Y_t.device, dtype=Y_t.dtype)
        one_minus_t = 1.0 - t_tensor
        numerator = t_tensor * self.eigenvalues - one_minus_t
        denominator = t_tensor.square() * self.eigenvalues + one_minus_t.square()
        gains = numerator / denominator
        centered_coordinates = (Y_t - t_tensor * self.mean) @ self.eigenvectors
        return self.mean + (centered_coordinates * gains) @ self.eigenvectors.T


@torch.no_grad()
def empirical_optimal_velocity(
    queries: torch.Tensor,
    clean_candidates: torch.Tensor,
    t: float,
    candidate_chunk_size: int = 512,
) -> torch.Tensor:
    """Evaluate the empirical conditional optimum by streaming its softmax."""
    if queries.ndim != 2 or clean_candidates.ndim != 2:
        raise ValueError("queries and clean_candidates must have shapes [B,d] and [K,d].")
    if queries.shape[1] != clean_candidates.shape[1] or len(clean_candidates) == 0:
        raise ValueError("Queries and candidates must share a non-empty dimension d.")
    if not 0.0 <= t < 1.0:
        raise ValueError("The empirical optimum requires t in [0,1).")
    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive.")

    one_minus_t = torch.as_tensor(1.0 - t, device=queries.device, dtype=queries.dtype)
    running_max = torch.full(
        (len(queries),), -torch.inf, device=queries.device, dtype=queries.dtype
    )
    running_denominator = torch.zeros_like(running_max)
    running_numerator = torch.zeros_like(queries)

    for start in range(0, len(clean_candidates), candidate_chunk_size):
        candidates = clean_candidates[start : start + candidate_chunk_size]
        residual = queries[:, None, :] - t * candidates[None, :, :]
        logits = -residual.square().sum(dim=-1) / (2.0 * one_minus_t.square())
        chunk_max = logits.max(dim=1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        weights = torch.exp(logits - new_max[:, None])
        running_denominator = running_denominator * old_scale + weights.sum(dim=1)
        running_numerator = (
            running_numerator * old_scale[:, None] + weights @ candidates
        )
        running_max = new_max

    posterior_clean_mean = running_numerator / running_denominator[:, None]
    return (posterior_clean_mean - queries) / one_minus_t
