"""Semantic independent polynomial models for Flow Matching velocity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_score.independent import IndependentPolynomialRidgeRegressor
from wavelet_score.indexing import WaveletLayout
from wavelet_score.transforms import WaveletTransform2D


@dataclass(slots=True)
class IndependentPolynomialVelocityRegressor(IndependentPolynomialRidgeRegressor):
    """Predict V_Y coefficient-wise at one fixed Flow Matching time."""

    def fit(
        self,
        Y_t: np.ndarray,
        velocity_target_wavelets: np.ndarray,
        batch_size: int = 1024,
    ) -> IndependentPolynomialVelocityRegressor:
        IndependentPolynomialRidgeRegressor.fit(
            self, Y_t, velocity_target_wavelets, batch_size=batch_size
        )
        return self

    def predict_wavelets(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict V_Y with shape [N,d]."""
        return self.predict(Y_t, batch_size=batch_size)

    def predict_pixels(
        self,
        Y_t: np.ndarray,
        transform: WaveletTransform2D,
        layout: WaveletLayout,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Predict V_Y then synthesize V_X without clipping or score conversion."""
        predicted_velocity_wavelets = self.predict_wavelets(Y_t, batch_size=batch_size)
        return transform.synthesize(predicted_velocity_wavelets, layout)
