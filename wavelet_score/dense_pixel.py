"""Literal dense pixel-space ridge regression from shared wavelet features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelet_score.indexing import WaveletLayout
from wavelet_score.transforms import WaveletTransform2D


def shared_polynomial_features(Y_t: np.ndarray, degree: int) -> np.ndarray:
    """Build [1,{y_i**p}_{i,p}] with shape [N,1+d*degree]."""
    if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 2:
        raise ValueError("Y_t must be a numpy array with shape [N,d].")
    if Y_t.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Y_t must use float32 or float64.")
    if Y_t.shape[0] == 0 or Y_t.shape[1] == 0:
        raise ValueError("Y_t must contain samples and coefficients.")
    if degree < 1:
        raise ValueError("degree must be at least one.")

    num_samples, dimension = Y_t.shape
    features = np.empty((num_samples, 1 + dimension * degree), dtype=Y_t.dtype)
    features[:, 0] = 1.0
    powers = np.empty((num_samples, dimension, degree), dtype=Y_t.dtype)
    powers[..., 0] = Y_t
    for power_index in range(1, degree):
        powers[..., power_index] = powers[..., power_index - 1] * Y_t
    features[:, 1:] = powers.reshape(num_samples, dimension * degree)
    return features


def synthesis_matrix(transform: WaveletTransform2D, layout: WaveletLayout) -> np.ndarray:
    """Materialize S such that flattened images X = Y @ S."""
    identity = np.eye(layout.num_coefficients, dtype=np.float64)
    atoms = transform.synthesize(identity, layout)
    return atoms.reshape(layout.num_coefficients, layout.num_coefficients)


def independent_weight_mask(dimension: int, degree: int) -> np.ndarray:
    """Return allowed shared-feature weights for independent wavelet outputs."""
    if dimension <= 0 or degree < 1:
        raise ValueError("dimension and degree must be positive.")
    mask = np.zeros((1 + dimension * degree, dimension), dtype=bool)
    mask[0, :] = True
    for coefficient_index in range(dimension):
        start = 1 + coefficient_index * degree
        mask[start : start + degree, coefficient_index] = True
    return mask


def off_mask_weight_fraction(weights_wavelet: np.ndarray, degree: int) -> float:
    """Measure the Frobenius-energy fraction violating independent coupling."""
    if weights_wavelet.ndim != 2:
        raise ValueError("weights_wavelet must have shape [1+d*degree,d].")
    dimension = weights_wavelet.shape[1]
    expected_features = 1 + dimension * degree
    if weights_wavelet.shape[0] != expected_features:
        raise ValueError(f"Expected {expected_features} feature rows.")
    allowed = independent_weight_mask(dimension, degree)
    total_energy = float(np.sum(weights_wavelet**2))
    if total_energy == 0.0:
        return 0.0
    return float(np.sum(weights_wavelet[~allowed] ** 2) / total_energy)


@dataclass(slots=True)
class DensePixelPolynomialRegressor:
    """Fit Equation (23) literally with a shared feature map and pixel outputs."""

    degree: int
    ridge: float
    weights_pixel_: np.ndarray | None = field(default=None, init=False, repr=False)
    weights_wavelet_: np.ndarray | None = field(default=None, init=False, repr=False)
    layout_: WaveletLayout | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be at least one.")
        if self.ridge <= 0.0 or not np.isfinite(self.ridge):
            raise ValueError("A finite positive ridge is required for the dense system.")

    def fit(
        self,
        Y_t: np.ndarray,
        images_0: np.ndarray,
        transform: WaveletTransform2D,
        layout: WaveletLayout,
    ) -> DensePixelPolynomialRegressor:
        """Fit W_pix in X_hat_0 = A W_pix and compute W_wav = W_pix S^T."""
        if images_0.ndim != 3 or images_0.shape[0] != Y_t.shape[0]:
            raise ValueError("images_0 must have shape [N,H,W] with the same N as Y_t.")
        if images_0.shape[1:] != layout.image_shape:
            raise ValueError("images_0 and layout use different image shapes.")
        features = shared_polynomial_features(Y_t, self.degree).astype(np.float64, copy=False)
        targets = images_0.reshape(images_0.shape[0], -1).astype(np.float64, copy=False)
        num_samples = features.shape[0]
        gram = features.T @ features / num_samples
        gram += self.ridge * np.eye(gram.shape[0], dtype=np.float64)
        rhs = features.T @ targets / num_samples
        self.weights_pixel_ = np.linalg.solve(gram, rhs)
        S = synthesis_matrix(transform, layout)
        self.weights_wavelet_ = self.weights_pixel_ @ S.T
        self.layout_ = layout
        return self

    def predict_pixels(self, Y_t: np.ndarray) -> np.ndarray:
        """Predict clean images [N,H,W] from noisy wavelet coefficients."""
        if self.weights_pixel_ is None or self.layout_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        features = shared_polynomial_features(Y_t, self.degree).astype(np.float64, copy=False)
        if features.shape[1] != self.weights_pixel_.shape[0]:
            raise ValueError("Y_t has a different dimension than the fitted data.")
        pixels = features @ self.weights_pixel_
        return pixels.reshape(Y_t.shape[0], *self.layout_.image_shape)

    def predict_wavelets(self, Y_t: np.ndarray) -> np.ndarray:
        """Predict clean coefficients using the transformed dense pixel weights."""
        if self.weights_wavelet_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        features = shared_polynomial_features(Y_t, self.degree).astype(np.float64, copy=False)
        return features @ self.weights_wavelet_
