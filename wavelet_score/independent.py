"""Independent polynomial ridge regression with semantic task wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def polynomial_features(Y_t: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, y, ..., y**degree] with shape [N,d,degree+1]."""
    if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 2:
        raise ValueError("Y_t must be a numpy array with shape [N,d].")
    if Y_t.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Y_t must use float32 or float64.")
    if Y_t.shape[0] == 0 or Y_t.shape[1] == 0:
        raise ValueError("Y_t must contain samples and coefficients.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    features = np.empty((*Y_t.shape, degree + 1), dtype=Y_t.dtype)
    features[..., 0] = 1.0
    for power in range(1, degree + 1):
        features[..., power] = features[..., power - 1] * Y_t
    return features


@dataclass(slots=True)
class IndependentPolynomialRidgeRegressor:
    """Fit an arbitrary [N,d] target independently from each input coefficient."""

    degree: int
    ridge: float = 0.0
    coefficients_: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("degree must be non-negative.")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative.")

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 1024,
    ) -> IndependentPolynomialRidgeRegressor:
        """Fit coefficients [d,D+1] from paired input and target batches."""
        self._validate_pair(inputs, targets)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        num_samples, dimension = inputs.shape
        num_features = self.degree + 1
        gram = np.zeros((dimension, num_features, num_features), dtype=np.float64)
        rhs = np.zeros((dimension, num_features), dtype=np.float64)

        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            features = polynomial_features(inputs[start:stop], self.degree).astype(
                np.float64, copy=False
            )
            target_batch = targets[start:stop].astype(np.float64, copy=False)
            gram += np.einsum("ndp,ndq->dpq", features, features, optimize=True)
            rhs += np.einsum("ndp,nd->dp", features, target_batch, optimize=True)

        gram /= num_samples
        rhs /= num_samples
        if self.ridge > 0.0:
            identity = np.eye(num_features, dtype=np.float64)
            gram += self.ridge * identity[None, :, :]

        try:
            self.coefficients_ = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "Independent normal equations are singular; use a positive ridge."
            ) from error
        return self

    def predict(self, inputs: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict the fitted target with shape [N,d]."""
        if self.coefficients_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        if not isinstance(inputs, np.ndarray) or inputs.ndim != 2:
            raise ValueError("inputs must have shape [N,d].")
        if inputs.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("inputs must use float32 or float64.")
        if inputs.shape[1] != self.coefficients_.shape[0]:
            raise ValueError("inputs have a different coefficient dimension than the fit data.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions = np.empty(inputs.shape, dtype=np.float64)
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            features = polynomial_features(inputs[start:stop], self.degree).astype(
                np.float64, copy=False
            )
            predictions[start:stop] = np.einsum(
                "ndp,dp->nd", features, self.coefficients_, optimize=True
            )
        return predictions

    @staticmethod
    def _validate_pair(inputs: np.ndarray, targets: np.ndarray) -> None:
        for name, array in (("inputs", inputs), ("targets", targets)):
            if not isinstance(array, np.ndarray) or array.ndim != 2:
                raise ValueError(f"{name} must be a numpy array with shape [N,d].")
            if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError(f"{name} must use float32 or float64.")
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have identical [N,d] shapes.")
        if inputs.shape[0] == 0 or inputs.shape[1] == 0:
            raise ValueError("Training arrays must contain samples and coefficients.")


@dataclass(slots=True)
class IndependentPolynomialRegressor(IndependentPolynomialRidgeRegressor):
    """Semantic wrapper for clean-coefficient denoising: Y_t -> Y_0."""

    def fit(
        self,
        Y_t: np.ndarray,
        Y_0: np.ndarray,
        batch_size: int = 1024,
    ) -> IndependentPolynomialRegressor:
        IndependentPolynomialRidgeRegressor.fit(
            self, Y_t, Y_0, batch_size=batch_size
        )
        return self
