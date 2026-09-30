"""Common interface for the coupled and isotropic-noise spectral paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .schedule import SequentialLinearSchedule, TimeLike


PathKind = Literal["coupled", "isotropic"]


@dataclass(frozen=True)
class PathCoefficients:
    """Diagonal coefficients of one spectral probability path."""

    alpha: torch.Tensor
    beta: torch.Tensor
    dot_alpha: torch.Tensor
    dot_beta: torch.Tensor


@dataclass(frozen=True)
class SpectralPath:
    """A spectral path sharing ``beta`` but choosing a noise schedule."""

    schedule: SequentialLinearSchedule
    kind: PathKind

    def __post_init__(self) -> None:
        if self.kind not in ("coupled", "isotropic"):
            raise ValueError(f"Unknown spectral path: {self.kind}")

    def coefficients(
        self,
        t: TimeLike,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> PathCoefficients:
        """Return ``alpha``, ``beta`` and their time derivatives."""
        beta = self.schedule.beta(t, device=device, dtype=dtype)
        dot_beta = self.schedule.dot_beta(t, device=device, dtype=dtype)

        if self.kind == "coupled":
            alpha = 1.0 - beta
            dot_alpha = -dot_beta
        else:
            time = torch.as_tensor(t, device=beta.device, dtype=beta.dtype)
            alpha = (1.0 - time).unsqueeze(-1).expand_as(beta)
            dot_alpha = -torch.ones_like(beta)

        return PathCoefficients(
            alpha=alpha,
            beta=beta,
            dot_alpha=dot_alpha,
            dot_beta=dot_beta,
        )

    def forward(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        t: TimeLike,
    ) -> torch.Tensor:
        """Construct ``z_t = alpha(t) z0 + beta(t) z1``.

        ``z0`` and ``z1`` must have shape ``[B, D]``. A vector of ``T`` times
        returns a tensor with shape ``[T, B, D]``.
        """
        if z0.shape != z1.shape or z0.ndim != 2:
            raise ValueError("z0 and z1 must have the same shape [B, D].")
        if z0.shape[-1] != self.schedule.dimension:
            raise ValueError("The state dimension does not match the schedule.")

        coefficients = self.coefficients(
            t,
            device=z0.device,
            dtype=z0.dtype,
        )
        if coefficients.beta.ndim == 1:
            return coefficients.alpha * z0 + coefficients.beta * z1
        return (
            coefficients.alpha.unsqueeze(1) * z0.unsqueeze(0)
            + coefficients.beta.unsqueeze(1) * z1.unsqueeze(0)
        )
