"""Exact scalar empirical conditional FM velocity under receptive masks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def brute_posterior_weights(
    references: np.ndarray,
    queries: np.ndarray,
    t: float,
    visible_indices: tuple[int, ...],
) -> np.ndarray:
    """Materialized reference implementation for tests only."""
    if references.ndim != 2 or queries.ndim != 2 or references.shape[1] != queries.shape[1]:
        raise ValueError("references and queries must be [N,d] matrices with matching d.")
    if not 0.0 <= t < 1.0:
        raise ValueError("t must lie in [0,1).")
    indices = np.asarray(visible_indices, dtype=np.int64)
    residual = queries[:, None, indices] - t * references[None, :, indices]
    logits = -np.sum(residual**2, axis=-1) / (2.0 * (1.0 - t) ** 2)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def brute_masked_velocity(
    references: np.ndarray,
    queries: np.ndarray,
    t: float,
    target_index: int,
    visible_indices: tuple[int, ...],
) -> np.ndarray:
    weights = brute_posterior_weights(references, queries, t, visible_indices)
    expected_clean = weights @ references[:, target_index]
    return (expected_clean - queries[:, target_index]) / (1.0 - t)


@dataclass(slots=True)
class ScalarMaskedEmpiricalVelocity:
    """Streaming exact empirical optimum for one target coefficient."""

    reference_chunk_size: int = 1024
    query_batch_size: int = 128
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float64
    references_: torch.Tensor | None = field(default=None, init=False, repr=False)

    def fit(self, references: np.ndarray) -> ScalarMaskedEmpiricalVelocity:
        if references.ndim != 2 or len(references) == 0:
            raise ValueError("references must have shape [N,d] with N positive.")
        if references.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("references must use float32 or float64.")
        if self.dtype != torch.float64:
            raise TypeError("The controlled reference estimator requires torch.float64.")
        if min(self.reference_chunk_size, self.query_batch_size) < 1:
            raise ValueError("Chunk sizes must be positive.")
        self.references_ = torch.as_tensor(
            np.ascontiguousarray(references), device=self.device, dtype=self.dtype
        )
        return self

    @torch.no_grad()
    def predict(
        self,
        queries: np.ndarray,
        t: float,
        target_index: int,
        visible_indices: tuple[int, ...],
    ) -> np.ndarray:
        if self.references_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        references = self.references_
        if queries.ndim != 2 or queries.shape[1] != references.shape[1]:
            raise ValueError("queries must have shape [Q,d] matching the reference bank.")
        if not 0.0 <= t < 1.0:
            raise ValueError("t must lie in [0,1).")
        if not 0 <= target_index < references.shape[1]:
            raise ValueError("target_index lies outside the coefficient vector.")
        indices_array = np.asarray(visible_indices, dtype=np.int64)
        if (
            indices_array.ndim != 1
            or not len(indices_array)
            or indices_array[0] != target_index
            or len(np.unique(indices_array)) != len(indices_array)
            or np.any(indices_array < 0)
            or np.any(indices_array >= references.shape[1])
        ):
            raise ValueError("visible_indices must be unique, valid, and target-first.")

        result = np.empty(len(queries), dtype=np.float64)
        visible = torch.as_tensor(indices_array, device=self.device, dtype=torch.long)
        variance = torch.as_tensor((1.0 - t) ** 2, device=self.device, dtype=self.dtype)
        for query_start in range(0, len(queries), self.query_batch_size):
            query_stop = min(query_start + self.query_batch_size, len(queries))
            query = torch.as_tensor(
                queries[query_start:query_stop], device=self.device, dtype=self.dtype
            )
            query_visible = query[:, visible]
            running_max = torch.full(
                (len(query),), -torch.inf, device=self.device, dtype=self.dtype
            )
            denominator = torch.zeros_like(running_max)
            clean_numerator = torch.zeros_like(running_max)
            for reference_start in range(0, len(references), self.reference_chunk_size):
                reference_stop = min(
                    reference_start + self.reference_chunk_size, len(references)
                )
                reference = references[reference_start:reference_stop]
                residual = query_visible[:, None] - t * reference[None, :, visible]
                logits = -residual.square().sum(dim=-1) / (2.0 * variance)
                chunk_max = logits.max(dim=1).values
                new_max = torch.maximum(running_max, chunk_max)
                previous_scale = torch.exp(running_max - new_max)
                weights = torch.exp(logits - new_max[:, None])
                denominator = denominator * previous_scale + weights.sum(dim=1)
                clean_numerator = (
                    clean_numerator * previous_scale
                    + weights @ reference[:, target_index]
                )
                running_max = new_max
            if not (
                torch.isfinite(running_max).all()
                and torch.isfinite(denominator).all()
                and torch.isfinite(clean_numerator).all()
                and torch.all(denominator > 0)
            ):
                raise FloatingPointError("The streaming posterior became non-finite.")
            expected_clean = clean_numerator / denominator
            velocity = (expected_clean - query[:, target_index]) / (1.0 - t)
            result[query_start:query_stop] = velocity.cpu().numpy()
        return result
