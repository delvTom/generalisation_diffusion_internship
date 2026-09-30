"""Orientation-coupled polynomial regression at fixed wavelet scale and location."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelet_score.band_features import band_monomial_exponents, evaluate_band_monomials
from wavelet_score.independent import IndependentPolynomialRegressor
from wavelet_score.indexing import SubbandDescriptor, WaveletLayout


def detail_descriptors_by_level(
    layout: WaveletLayout,
) -> dict[int, tuple[SubbandDescriptor, SubbandDescriptor, SubbandDescriptor]]:
    """Group detail descriptors by level in the fixed orientation order H,V,D."""
    grouped: dict[int, tuple[SubbandDescriptor, SubbandDescriptor, SubbandDescriptor]] = {}
    for pywt_level in range(layout.decomposition_level, 0, -1):
        descriptors = tuple(
            band
            for orientation in ("H", "V", "D")
            for band in layout.subbands
            if band.coefficient_type == "detail"
            and band.pywt_level == pywt_level
            and band.orientation == orientation
        )
        if len(descriptors) != 3:
            raise ValueError(f"Level {pywt_level} does not contain exactly H,V,D.")
        if len({descriptor.shape for descriptor in descriptors}) != 1:
            raise ValueError(f"Detail bands at level {pywt_level} have different shapes.")
        grouped[pywt_level] = descriptors  # type: ignore[assignment]
    return grouped


def extract_detail_triplets(
    Y_t: np.ndarray,
    descriptors: tuple[SubbandDescriptor, SubbandDescriptor, SubbandDescriptor],
) -> np.ndarray:
    """Extract same-location coefficients as [N,K,3] ordered H,V,D."""
    expected_orientations = ("H", "V", "D")
    if tuple(descriptor.orientation for descriptor in descriptors) != expected_orientations:
        raise ValueError("descriptors must follow the order H,V,D.")
    return np.stack([Y_t[:, descriptor.flat_slice] for descriptor in descriptors], axis=-1)


@dataclass(slots=True)
class BandCoupledPolynomialRegressor:
    """Fit scaling coefficients independently and couple H,V,D at each (j,k)."""

    degree: int
    ridge: float = 0.0
    scaling_model_: IndependentPolynomialRegressor | None = field(
        default=None, init=False, repr=False
    )
    detail_coefficients_: dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    layout_: WaveletLayout | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("degree must be at least one.")
        if self.ridge < 0.0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be finite and non-negative.")

    def fit(
        self,
        Y_t: np.ndarray,
        Y_0: np.ndarray,
        layout: WaveletLayout,
        batch_size: int = 1024,
    ) -> BandCoupledPolynomialRegressor:
        """Fit all coefficient-wise systems from paired [N,d] arrays."""
        self._validate_arrays(Y_t, Y_0, layout)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        num_samples = Y_t.shape[0]

        scaling = layout.subbands[0]
        self.scaling_model_ = IndependentPolynomialRegressor(
            degree=self.degree, ridge=self.ridge
        ).fit(
            Y_t[:, scaling.flat_slice],
            Y_0[:, scaling.flat_slice],
            batch_size=batch_size,
        )

        grouped = detail_descriptors_by_level(layout)
        orientations = ("H", "V", "D")
        self.detail_coefficients_ = {}
        for pywt_level, descriptors in grouped.items():
            num_positions = descriptors[0].size
            num_features = len(band_monomial_exponents(self.degree, "H"))
            gram = np.zeros((3, num_positions, num_features, num_features), dtype=np.float64)
            rhs = np.zeros((3, num_positions, num_features), dtype=np.float64)

            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                triplets = extract_detail_triplets(Y_t[start:stop], descriptors)
                targets = extract_detail_triplets(Y_0[start:stop], descriptors).astype(
                    np.float64, copy=False
                )
                for target_axis, orientation in enumerate(orientations):
                    exponents = band_monomial_exponents(self.degree, orientation)
                    features = evaluate_band_monomials(triplets, exponents).astype(
                        np.float64, copy=False
                    )
                    gram[target_axis] += np.einsum(
                        "nkp,nkq->kpq", features, features, optimize=True
                    )
                    rhs[target_axis] += np.einsum(
                        "nkp,nk->kp", features, targets[..., target_axis], optimize=True
                    )

            gram /= num_samples
            rhs /= num_samples
            if self.ridge > 0.0:
                gram += self.ridge * np.eye(num_features, dtype=np.float64)[None, None]
            try:
                coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError as error:
                raise np.linalg.LinAlgError(
                    f"Band-coupled systems are singular at pywt_level={pywt_level}; "
                    "use a positive ridge."
                ) from error
            self.detail_coefficients_[pywt_level] = coefficients

        self.layout_ = layout
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict clean wavelet coefficients with shape [N,d]."""
        if self.layout_ is None or self.scaling_model_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        self._validate_arrays(Y_t, None, self.layout_)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions = np.empty(Y_t.shape, dtype=np.float64)
        scaling = self.layout_.subbands[0]
        predictions[:, scaling.flat_slice] = self.scaling_model_.predict(
            Y_t[:, scaling.flat_slice], batch_size=batch_size
        )
        grouped = detail_descriptors_by_level(self.layout_)
        orientations = ("H", "V", "D")

        for start in range(0, Y_t.shape[0], batch_size):
            stop = min(start + batch_size, Y_t.shape[0])
            for pywt_level, descriptors in grouped.items():
                triplets = extract_detail_triplets(Y_t[start:stop], descriptors)
                coefficients = self.detail_coefficients_[pywt_level]
                for target_axis, (orientation, descriptor) in enumerate(
                    zip(orientations, descriptors, strict=True)
                ):
                    features = evaluate_band_monomials(
                        triplets, band_monomial_exponents(self.degree, orientation)
                    ).astype(np.float64, copy=False)
                    predictions[start:stop, descriptor.flat_slice] = np.einsum(
                        "nkp,kp->nk", features, coefficients[target_axis], optimize=True
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
