"""Analytic affine optimum for the standard noise-to-data FM path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class AffineFMWiener:
    """Plug-in population optimum using empirical clean-data moments.

    The clean distribution is the uniform empirical distribution over the fit
    images, while the Gaussian noise expectation is integrated analytically.
    """

    mean_: torch.Tensor  # [d]
    eigenvalues_: torch.Tensor  # [d], covariance eigenvalues
    eigenvectors_: torch.Tensor  # [d,d]

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        clean_coefficients: np.ndarray,
        *,
        device: torch.device,
        batch_size: int = 2048,
        moment_dtype: torch.dtype = torch.float32,
        solve_dtype: torch.dtype = torch.float64,
    ) -> "AffineFMWiener":
        if clean_coefficients.ndim != 2 or len(clean_coefficients) < 2:
            raise ValueError("clean_coefficients must have shape [N,d] with N >= 2.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        num_samples, dimension = clean_coefficients.shape
        sum_y = torch.zeros(dimension, device=device, dtype=solve_dtype)
        second_moment = torch.zeros(
            (dimension, dimension), device=device, dtype=solve_dtype
        )
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            batch = torch.as_tensor(
                clean_coefficients[start:stop], device=device, dtype=moment_dtype
            )
            sum_y.add_(batch.sum(dim=0, dtype=solve_dtype))
            second_moment.add_((batch.T @ batch).to(solve_dtype))
        mean = sum_y / num_samples
        covariance = second_moment / num_samples - torch.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        scale = float(covariance.abs().max().item())
        tolerance = torch.finfo(moment_dtype).eps * max(dimension, 1) * max(scale, 1.0)
        if float(eigenvalues.min().item()) < -10.0 * tolerance:
            raise RuntimeError(
                "The empirical covariance has a significantly negative eigenvalue; "
                "increase moment precision."
            )
        eigenvalues.clamp_(min=0.0)
        return cls(mean, eigenvalues, eigenvectors)

    @property
    def device(self) -> torch.device:
        return self.mean_.device

    @torch.no_grad()
    def predict_tensor(self, Y_t: torch.Tensor, t: float) -> torch.Tensor:
        """Predict wavelet velocity for Y_t using the full affine operator."""
        if Y_t.ndim != 2 or Y_t.shape[1] != self.mean_.numel():
            raise ValueError("Y_t has an incompatible shape.")
        if not 0.0 < float(t) < 1.0:
            raise ValueError("t must lie strictly between zero and one.")
        Y_t = Y_t.to(device=self.device, dtype=self.mean_.dtype)
        time = torch.as_tensor(float(t), device=self.device, dtype=self.mean_.dtype)
        noise_scale = 1.0 - time
        denominator = time.square() * self.eigenvalues_ + noise_scale.square()
        gains = (time * self.eigenvalues_ - noise_scale) / denominator
        centered = Y_t - time * self.mean_
        spectral = centered @ self.eigenvectors_
        return self.mean_ + (spectral * gains) @ self.eigenvectors_.T

    @torch.no_grad()
    def mse(
        self,
        Y_t: np.ndarray,
        targets: np.ndarray,
        t: float,
        *,
        batch_size: int = 2048,
    ) -> float:
        if Y_t.shape != targets.shape or Y_t.ndim != 2:
            raise ValueError("Y_t and targets must have the same matrix shape.")
        squared_error = torch.zeros((), device=self.device, dtype=torch.float64)
        for start in range(0, len(Y_t), batch_size):
            stop = min(start + batch_size, len(Y_t))
            inputs = torch.as_tensor(Y_t[start:stop], device=self.device)
            expected = torch.as_tensor(
                targets[start:stop], device=self.device, dtype=self.mean_.dtype
            )
            prediction = self.predict_tensor(inputs, t)
            squared_error.add_((prediction - expected).square().sum(dtype=torch.float64))
        return float((squared_error / targets.size).item())
