"""Periodic same-subband local polynomial regression for wavelet denoising."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelet_score.indexing import SubbandDescriptor, WaveletLayout
from wavelet_score.local_features import evaluate_local_monomials, local_monomial_terms


def extract_subband_maps(Y_t: np.ndarray, descriptor: SubbandDescriptor) -> np.ndarray:
    """Extract one flattened subband and reshape it to [N,h,w]."""
    if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 2:
        raise ValueError("Y_t must have shape [N,d].")
    return Y_t[:, descriptor.flat_slice].reshape(Y_t.shape[0], *descriptor.shape)


def validate_local_radius(layout: WaveletLayout, radius: int) -> None:
    """Reject neighborhoods that alias offsets on a small periodic subband."""
    if radius < 1:
        raise ValueError("radius must be at least one.")
    neighborhood_width = 2 * radius + 1
    for descriptor in layout.subbands:
        if neighborhood_width > min(descriptor.shape):
            raise ValueError(
                f"radius={radius} aliases periodic neighbors in subband "
                f"{descriptor.subband_index} with shape {descriptor.shape}."
            )


@dataclass(slots=True)
class LocalCoupledPolynomialRegressor:
    """Fit a local ridge-polynomial model at every wavelet coefficient."""

    degree: int
    radius: int
    ridge: float = 0.0
    coefficients_: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    layout_: WaveletLayout | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be at least one.")
        if self.radius < 1:
            raise ValueError("radius must be at least one.")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative.")

    def fit(
        self,
        Y_t: np.ndarray,
        Y_0: np.ndarray,
        layout: WaveletLayout,
        batch_size: int = 1024,
    ) -> LocalCoupledPolynomialRegressor:
        """Fit one [P,P] ridge system for each subband position."""
        self._validate_arrays(Y_t, Y_0, layout)
        validate_local_radius(layout, self.radius)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        num_samples = Y_t.shape[0]
        terms = local_monomial_terms(self.degree, self.radius)
        num_features = len(terms)
        self.coefficients_ = {}

        for descriptor in layout.subbands:
            gram = np.zeros(
                (descriptor.size, num_features, num_features), dtype=np.float64
            )
            rhs = np.zeros((descriptor.size, num_features), dtype=np.float64)

            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                noisy_maps = extract_subband_maps(Y_t[start:stop], descriptor)
                features = evaluate_local_monomials(noisy_maps, terms).astype(
                    np.float64, copy=False
                )
                targets = Y_0[start:stop, descriptor.flat_slice].astype(
                    np.float64, copy=False
                )
                gram += np.einsum("nkp,nkq->kpq", features, features, optimize=True)
                rhs += np.einsum("nkp,nk->kp", features, targets, optimize=True)

            gram /= num_samples
            rhs /= num_samples
            if self.ridge > 0.0:
                gram += self.ridge * np.eye(num_features, dtype=np.float64)[None]
            try:
                self.coefficients_[descriptor.subband_index] = np.linalg.solve(
                    gram, rhs[..., None]
                )[..., 0]
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    f"Local systems are singular in subband {descriptor.subband_index}; "
                    "use a positive ridge."
                ) from error

        self.layout_ = layout
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict clean wavelet coefficients with shape [N,d]."""
        if self.layout_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        self._validate_arrays(Y_t, None, self.layout_)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions = np.empty(Y_t.shape, dtype=np.float64)
        terms = local_monomial_terms(self.degree, self.radius)
        for start in range(0, Y_t.shape[0], batch_size):
            stop = min(start + batch_size, Y_t.shape[0])
            for descriptor in self.layout_.subbands:
                maps = extract_subband_maps(Y_t[start:stop], descriptor)
                features = evaluate_local_monomials(maps, terms).astype(
                    np.float64, copy=False
                )
                predictions[start:stop, descriptor.flat_slice] = np.einsum(
                    "nkp,kp->nk",
                    features,
                    self.coefficients_[descriptor.subband_index],
                    optimize=True,
                )
        return predictions

    @staticmethod
    def _validate_arrays(
        Y_t: np.ndarray,
        Y_0: np.ndarray | None,
        layout: WaveletLayout,
    ) -> None:
        if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 2:
            raise ValueError("Y_t must have shape [N,d].")
        if Y_t.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("Y_t must use float32 or float64.")
        if Y_t.shape[0] == 0 or Y_t.shape[1] != layout.num_coefficients:
            raise ValueError("Y_t is empty or incompatible with layout.")
        if Y_0 is not None:
            if not isinstance(Y_0, np.ndarray) or Y_0.shape != Y_t.shape:
                raise ValueError("Y_0 must have the same [N,d] shape as Y_t.")
            if Y_0.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise TypeError("Y_0 must use float32 or float64.")
