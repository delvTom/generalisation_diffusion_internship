"""Optimal affine velocity field for a diagonal spectral path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paths import SpectralPath
from .schedule import TimeLike


@dataclass(frozen=True)
class AffineCoefficients:
    """Per-component coefficients of the affine velocity field."""

    beta: torch.Tensor
    dot_beta: torch.Tensor
    variance: torch.Tensor
    slope: torch.Tensor
    bias: torch.Tensor


@dataclass
class AffineSpectralVelocity:
    """Optimal affine approximation of the spectral FM velocity.

    Parameters
    ----------
    path:
        Coupled or isotropic-noise probability path.
    lambda_sq:
        Population covariance eigenvalues, shape ``[D]``.
    mu_z:
        Non-centered data mean in PCA coordinates, shape ``[D]``.
    variance_floor:
        Numerical floor used only in divisions and square roots.
    """

    path: SpectralPath
    lambda_sq: torch.Tensor
    mu_z: torch.Tensor
    variance_floor: float = 1e-12

    def __post_init__(self) -> None:
        dimension = self.path.schedule.dimension
        if self.lambda_sq.shape != (dimension,):
            raise ValueError("lambda_sq must have shape [D].")
        if self.mu_z.shape != (dimension,):
            raise ValueError("mu_z must have shape [D].")
        if self.lambda_sq.device != self.mu_z.device:
            raise ValueError("lambda_sq and mu_z must use the same device.")
        if self.lambda_sq.dtype != self.mu_z.dtype:
            raise ValueError("lambda_sq and mu_z must use the same dtype.")

    def coefficients(self, t: TimeLike) -> AffineCoefficients:
        """Return variance, slope and bias for every spectral component."""
        path_coefficients = self.path.coefficients(
            t,
            device=self.lambda_sq.device,
            dtype=self.lambda_sq.dtype,
        )
        alpha = path_coefficients.alpha
        beta = path_coefficients.beta
        dot_alpha = path_coefficients.dot_alpha
        dot_beta = path_coefficients.dot_beta

        variance = alpha.square() + self.lambda_sq * beta.square()
        safe_variance = variance.clamp_min(self.variance_floor)
        slope = (
            dot_alpha * alpha
            + self.lambda_sq * dot_beta * beta
        ) / safe_variance
        bias = (dot_beta - slope * beta) * self.mu_z
        return AffineCoefficients(
            beta=beta,
            dot_beta=dot_beta,
            variance=variance,
            slope=slope,
            bias=bias,
        )

    def field(
        self,
        z: torch.Tensor,
        t: float | torch.Tensor,
        *,
        centered_form: bool = True,
    ) -> torch.Tensor:
        """Evaluate the affine velocity on states ``z`` of shape ``[B, D]``."""
        if z.ndim != 2 or z.shape[-1] != self.path.schedule.dimension:
            raise ValueError("z must have shape [B, D].")
        coefficients = self.coefficients(t)
        if coefficients.slope.ndim != 1:
            raise ValueError("field expects one scalar time per evaluation.")

        if centered_form:
            moving_mean = coefficients.beta * self.mu_z
            moving_mean_velocity = coefficients.dot_beta * self.mu_z
            return (
                coefficients.slope * (z - moving_mean)
                + moving_mean_velocity
            )
        return coefficients.slope * z + coefficients.bias

    def exact_state(self, z0: torch.Tensor, t: TimeLike) -> torch.Tensor:
        """Evaluate the exact affine ODE trajectory at arbitrary times.

        Parameters
        ----------
        z0:
            Initial Gaussian samples, shape ``[B, D]``.
        t:
            Scalar time or tensor with shape ``[T]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]`` for one time or ``[T, B, D]`` for ``T`` times.
        """
        if z0.ndim != 2 or z0.shape[-1] != self.path.schedule.dimension:
            raise ValueError("z0 must have shape [B, D].")
        coefficients = self.coefficients(t)
        moving_mean = coefficients.beta * self.mu_z
        scale = torch.sqrt(coefficients.variance.clamp_min(self.variance_floor))
        if moving_mean.ndim == 1:
            return moving_mean + scale * z0
        return moving_mean.unsqueeze(1) + scale.unsqueeze(1) * z0.unsqueeze(0)

    def exact_transition(
        self,
        z_from: torch.Tensor,
        t_from: float,
        t_to: float,
    ) -> torch.Tensor:
        """Apply the exact affine flow map from ``t_from`` to ``t_to``."""
        if not 0.0 <= t_from <= t_to <= 1.0:
            raise ValueError("Expected 0 <= t_from <= t_to <= 1.")
        if z_from.ndim != 2:
            raise ValueError("z_from must have shape [B, D].")

        source = self.coefficients(t_from)
        target = self.coefficients(t_to)
        source_mean = source.beta * self.mu_z
        target_mean = target.beta * self.mu_z
        variance_ratio = target.variance.clamp_min(self.variance_floor) / (
            source.variance.clamp_min(self.variance_floor)
        )
        return target_mean + torch.sqrt(variance_ratio) * (z_from - source_mean)
