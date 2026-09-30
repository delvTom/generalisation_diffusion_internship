"""Batch-invariant posterior averaging over an empirical candidate bank.

This local implementation replaces the upstream batch-wise normalization used
by the FM comparison scripts. The candidate posterior itself follows the
empirical estimator studied by Lukoianov et al.; see THIRD_PARTY.md.
"""

from __future__ import annotations

import torch


class WeightedStreamingSoftmax:
    """Accumulate a softmax-weighted mean across arbitrary candidate chunks."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.log_normalizer: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None

    def add(self, values: torch.Tensor, logits: torch.Tensor) -> None:
        if logits.ndim != 3 or values.ndim != 2:
            raise ValueError("Expected logits [B,K,D] and values [K,D].")
        if values.shape != logits.shape[1:] or values.shape[0] == 0:
            raise ValueError("Candidate values must match the nonempty chunk.")
        if self.mean is not None and self.mean.shape != (logits.shape[0], logits.shape[2]):
            raise ValueError("All chunks must have the same batch and output dimensions.")

        values = values.to(device=self.device, dtype=self.dtype)
        logits = logits.to(device=self.device, dtype=self.dtype)
        chunk_log_z = torch.logsumexp(logits, dim=1)
        chunk_mean = torch.einsum("bkd,kd->bd", torch.softmax(logits, dim=1), values)
        if self.mean is None or self.log_normalizer is None:
            self.mean = chunk_mean
            self.log_normalizer = chunk_log_z
            return

        new_log_z = torch.logaddexp(self.log_normalizer, chunk_log_z)
        self.mean = (
            self.mean * torch.exp(self.log_normalizer - new_log_z)
            + chunk_mean * torch.exp(chunk_log_z - new_log_z)
        )
        self.log_normalizer = new_log_z

    def get_average(self) -> torch.Tensor | None:
        return self.mean
