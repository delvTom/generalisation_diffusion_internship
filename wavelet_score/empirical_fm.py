"""Exact unconstrained empirical Flow Matching optimum with streaming references."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class EmpiricalFMVelocity:
    reference_chunk_size: int = 512
    query_batch_size: int = 16
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    references_: torch.Tensor | None = field(default=None, init=False, repr=False)

    def fit(self, clean_references: np.ndarray):
        if clean_references.ndim != 2 or len(clean_references) == 0:
            raise ValueError("clean_references must have shape [N,d].")
        self.references_ = torch.as_tensor(
            np.ascontiguousarray(clean_references), device=self.device, dtype=self.dtype
        )
        return self

    @torch.no_grad()
    def predict_clean(self, states: np.ndarray, t: float) -> np.ndarray:
        if self.references_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        if states.ndim != 2 or states.shape[1] != self.references_.shape[1]:
            raise ValueError("states must have shape [B,d].")
        if not 0.0 <= t < 1.0:
            raise ValueError("t must lie in [0,1).")
        if t == 0.0:
            mean = self.references_.mean(dim=0).cpu().numpy()
            return np.broadcast_to(mean, states.shape).copy()
        result = np.empty(states.shape, dtype=np.float32)
        variance = torch.as_tensor((1.0 - t) ** 2, device=self.device, dtype=self.dtype)
        for query_start in range(0, len(states), self.query_batch_size):
            query_stop = min(query_start + self.query_batch_size, len(states))
            query = torch.as_tensor(
                states[query_start:query_stop], device=self.device, dtype=self.dtype
            )
            running_max = torch.full((len(query),), -torch.inf, device=self.device, dtype=self.dtype)
            denominator = torch.zeros_like(running_max)
            numerator = torch.zeros_like(query)
            for reference_start in range(0, len(self.references_), self.reference_chunk_size):
                reference_stop = min(
                    reference_start + self.reference_chunk_size, len(self.references_)
                )
                references = self.references_[reference_start:reference_stop]
                residual = query[:, None] - t * references[None]
                logits = -residual.square().sum(dim=-1) / (2.0 * variance)
                chunk_max = logits.max(dim=1).values
                new_max = torch.maximum(running_max, chunk_max)
                old_scale = torch.exp(running_max - new_max)
                weights = torch.exp(logits - new_max[:, None])
                denominator = denominator * old_scale + weights.sum(dim=1)
                numerator = numerator * old_scale[:, None] + weights @ references
                running_max = new_max
            result[query_start:query_stop] = (numerator / denominator[:, None]).cpu().numpy()
        return result

    def predict_velocity(self, states: np.ndarray, t: float) -> np.ndarray:
        return (self.predict_clean(states, t) - states) / (1.0 - t)
