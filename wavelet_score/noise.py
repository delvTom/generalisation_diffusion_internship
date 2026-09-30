"""Variance-exploding Gaussian noise in pixel and wavelet coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_score.indexing import WaveletLayout
from wavelet_score.transforms import WaveletTransform2D


@dataclass(frozen=True, slots=True)
class VENoiseBatch:
    """One VE corruption represented in both pixel and wavelet spaces."""

    images_0: np.ndarray  # [N,H,W]
    epsilon: np.ndarray  # [N,H,W]
    images_t: np.ndarray  # [N,H,W]
    Y_0: np.ndarray  # [N,d]
    Eta: np.ndarray  # [N,d]
    Y_t: np.ndarray  # [N,d]
    sigma: float
    layout: WaveletLayout

    def __post_init__(self) -> None:
        if self.images_0.shape != self.epsilon.shape or self.images_0.shape != self.images_t.shape:
            raise ValueError("Pixel-space arrays must have identical [N,H,W] shapes.")
        if self.Y_0.shape != self.Eta.shape or self.Y_0.shape != self.Y_t.shape:
            raise ValueError("Wavelet-space arrays must have identical [N,d] shapes.")
        expected_wavelet_shape = self.images_0.shape[0], self.layout.num_coefficients
        if self.Y_0.shape != expected_wavelet_shape:
            raise ValueError(f"Expected wavelet arrays of shape {expected_wavelet_shape}.")
        if self.sigma < 0.0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and non-negative.")


def sample_ve_noise(
    images_0: np.ndarray,
    sigma: float,
    transform: WaveletTransform2D,
    generator: np.random.Generator,
) -> VENoiseBatch:
    """Sample images_t = images_0 + sigma*epsilon and analyze every term."""
    if not isinstance(images_0, np.ndarray) or images_0.ndim != 3:
        raise ValueError("images_0 must be a numpy array with shape [N,H,W].")
    if images_0.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("images_0 must use float32 or float64.")
    if not np.isscalar(sigma):
        raise TypeError("sigma must be a scalar, not a batched array.")
    sigma_float = float(sigma)
    if sigma_float < 0.0 or not np.isfinite(sigma_float):
        raise ValueError("sigma must be finite and non-negative.")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator.")

    epsilon = generator.standard_normal(images_0.shape, dtype=images_0.dtype)
    images_t = images_0 + sigma_float * epsilon
    Y_0, layout = transform.analyze(images_0)
    Eta, noise_layout = transform.analyze(epsilon)
    Y_t, noisy_layout = transform.analyze(images_t)
    if noise_layout != layout or noisy_layout != layout:
        raise RuntimeError("Clean, noise, and noisy images produced different layouts.")

    return VENoiseBatch(
        images_0=images_0,
        epsilon=epsilon,
        images_t=images_t,
        Y_0=Y_0,
        Eta=Eta,
        Y_t=Y_t,
        sigma=sigma_float,
        layout=layout,
    )
