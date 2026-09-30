"""CUDA-capable implementations of the audited structured regressors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from wavelet_score.band_coupled import detail_descriptors_by_level
from wavelet_score.band_features import band_monomial_exponents
from wavelet_score.indexing import SubbandDescriptor, WaveletLayout
from wavelet_score.local_coupled import validate_local_radius
from wavelet_score.local_features import LocalMonomialTerm, local_monomial_terms


def torch_dtype(name: str) -> torch.dtype:
    """Map a CLI dtype name to its torch dtype."""
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64.")


def tensor_batch(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move one contiguous NumPy batch to the requested compute device and dtype."""
    return torch.as_tensor(array, device=device, dtype=dtype)


def independent_features(values: torch.Tensor, degree: int) -> torch.Tensor:
    """Build [1,y,...,y**D] from values [N,K], returning [N,K,D+1]."""
    features = torch.empty((*values.shape, degree + 1), device=values.device, dtype=values.dtype)
    features[..., 0] = 1.0
    for power in range(1, degree + 1):
        features[..., power] = features[..., power - 1] * values
    return features


def shared_polynomial_features(values: torch.Tensor, degree: int) -> torch.Tensor:
    """Build the shared dictionary [1,{y_i**p}_{i,p}] from values [N,d]."""
    if values.ndim != 2:
        raise ValueError("values must have shape [N,d].")
    num_samples, dimension = values.shape
    features = torch.empty(
        (num_samples, 1 + dimension * degree),
        device=values.device,
        dtype=values.dtype,
    )
    features[:, 0] = 1.0
    powers = features[:, 1:].view(num_samples, dimension, degree)
    powers[..., 0] = values
    for power_index in range(1, degree):
        powers[..., power_index] = powers[..., power_index - 1] * values
    return features


def band_features(
    triplets: torch.Tensor,
    exponents: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    """Evaluate audited H/V/D monomials on triplets [N,K,3]."""
    features = torch.ones(
        (*triplets.shape[:-1], len(exponents)),
        device=triplets.device,
        dtype=triplets.dtype,
    )
    for feature_index, exponent in enumerate(exponents):
        for axis, power in enumerate(exponent):
            if power:
                features[..., feature_index] *= triplets[..., axis].pow(power)
    return features


def local_features(
    maps: torch.Tensor,
    terms: tuple[LocalMonomialTerm, ...],
) -> torch.Tensor:
    """Evaluate periodic local features from maps [N,h,w], returning [N,K,P]."""
    num_samples, height, width = maps.shape
    features = torch.empty(
        (num_samples, height * width, len(terms)),
        device=maps.device,
        dtype=maps.dtype,
    )
    neighbors: dict[tuple[int, int], torch.Tensor] = {}
    for feature_index, term in enumerate(terms):
        values = maps.pow(term.center_power)
        if term.offset is not None:
            if term.offset not in neighbors:
                delta_1, delta_2 = term.offset
                neighbors[term.offset] = torch.roll(
                    maps,
                    shifts=(-delta_1, -delta_2),
                    dims=(-2, -1),
                )
            values = values * neighbors[term.offset].pow(term.neighbor_power)
        features[..., feature_index] = values.reshape(num_samples, height * width)
    return features


def solve_ridge(gram: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    """Normalize externally accumulated moments, regularize, and solve batched systems."""
    if ridge > 0.0:
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        gram = gram + ridge * identity
    return torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)


@dataclass(slots=True)
class TorchIndependentPolynomialRegressor:
    degree: int
    ridge: float
    device: torch.device
    dtype: torch.dtype
    coefficients_: torch.Tensor | None = field(default=None, init=False, repr=False)

    def fit(self, Y_t: np.ndarray, Y_0: np.ndarray, batch_size: int = 1024):
        """Fit independent systems on the selected torch device."""
        num_samples, dimension = Y_t.shape
        num_features = self.degree + 1
        gram = torch.zeros((dimension, num_features, num_features), device=self.device, dtype=self.dtype)
        rhs = torch.zeros((dimension, num_features), device=self.device, dtype=self.dtype)
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            values = tensor_batch(Y_t[start:stop], self.device, self.dtype)
            targets = tensor_batch(Y_0[start:stop], self.device, self.dtype)
            features = independent_features(values, self.degree)
            gram.add_(torch.einsum("ndp,ndq->dpq", features, features))
            rhs.add_(torch.einsum("ndp,nd->dp", features, targets))
        self.coefficients_ = solve_ridge(gram / num_samples, rhs / num_samples, self.ridge)
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict [N,d] and return a CPU NumPy array."""
        if self.coefficients_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        prediction = np.empty(Y_t.shape, dtype=np.float64 if self.dtype == torch.float64 else np.float32)
        for start in range(0, len(Y_t), batch_size):
            stop = min(start + batch_size, len(Y_t))
            values = tensor_batch(Y_t[start:stop], self.device, self.dtype)
            result = torch.einsum(
                "ndp,dp->nd", independent_features(values, self.degree), self.coefficients_
            )
            prediction[start:stop] = result.detach().cpu().numpy()
        return prediction


@dataclass(slots=True)
class TorchDenseSharedPolynomialRegressor:
    """Literal Equation (23): one shared polynomial map and a dense output W."""

    degree: int
    ridge: float
    device: torch.device
    dtype: torch.dtype
    coefficients_: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def num_features_(self) -> int:
        if self.coefficients_ is None:
            raise RuntimeError("The regressor must be fitted before reading num_features_.")
        return self.coefficients_.shape[0]

    def fit(self, Y_t: np.ndarray, Y_0: np.ndarray, batch_size: int = 1024):
        """Fit Y_0_hat = Phi(Y_t) W by accumulating its sufficient statistics."""
        if Y_t.ndim != 2 or Y_0.shape != Y_t.shape:
            raise ValueError("Y_t and Y_0 must have the same shape [N,d].")
        num_samples, dimension = Y_t.shape
        num_features = 1 + dimension * self.degree
        gram = torch.zeros(
            (num_features, num_features), device=self.device, dtype=self.dtype
        )
        rhs = torch.zeros(
            (num_features, dimension), device=self.device, dtype=self.dtype
        )
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            features = shared_polynomial_features(
                tensor_batch(Y_t[start:stop], self.device, self.dtype), self.degree
            )
            targets = tensor_batch(Y_0[start:stop], self.device, self.dtype)
            gram.add_(features.T @ features)
            rhs.add_(features.T @ targets)
        if self.ridge > 0.0:
            gram.add_(self.ridge * num_samples * torch.eye(
                num_features, device=self.device, dtype=self.dtype
            ))
        self.coefficients_ = torch.linalg.solve(gram, rhs)
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict all clean wavelet coefficients and return a CPU NumPy array."""
        if self.coefficients_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        output_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        prediction = np.empty(Y_t.shape, dtype=output_dtype)
        for start in range(0, len(Y_t), batch_size):
            stop = min(start + batch_size, len(Y_t))
            features = shared_polynomial_features(
                tensor_batch(Y_t[start:stop], self.device, self.dtype), self.degree
            )
            prediction[start:stop] = (
                features @ self.coefficients_
            ).detach().cpu().numpy()
        return prediction

    def off_independent_mask_weight_fraction(self) -> float:
        """Return the weight energy using another coefficient than the target one."""
        if self.coefficients_ is None:
            raise RuntimeError("The regressor must be fitted before this diagnostic.")
        weights = self.coefficients_
        dimension = weights.shape[1]
        coefficient_weights = weights[1:].view(dimension, self.degree, dimension)
        indices = torch.arange(dimension, device=weights.device)
        allowed_energy = weights[0].square().sum()
        allowed_energy = allowed_energy + coefficient_weights[
            indices, :, indices
        ].square().sum()
        total_energy = weights.square().sum()
        if total_energy.item() == 0.0:
            return 0.0
        return float(((total_energy - allowed_energy) / total_energy).item())


@dataclass(slots=True)
class TorchBandCoupledPolynomialRegressor:
    degree: int
    ridge: float
    device: torch.device
    dtype: torch.dtype
    scaling_model_: TorchIndependentPolynomialRegressor | None = field(default=None, init=False)
    detail_coefficients_: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    layout_: WaveletLayout | None = field(default=None, init=False)

    def fit(self, Y_t: np.ndarray, Y_0: np.ndarray, layout: WaveletLayout, batch_size: int = 1024):
        """Fit scaling-independent and orientation-coupled systems on torch."""
        num_samples = len(Y_t)
        scaling = layout.subbands[0]
        self.scaling_model_ = TorchIndependentPolynomialRegressor(
            self.degree, self.ridge, self.device, self.dtype
        ).fit(Y_t[:, scaling.flat_slice], Y_0[:, scaling.flat_slice], batch_size)
        self.detail_coefficients_ = {}
        for level, descriptors in detail_descriptors_by_level(layout).items():
            positions = descriptors[0].size
            feature_count = len(band_monomial_exponents(self.degree, "H"))
            gram = torch.zeros((3, positions, feature_count, feature_count), device=self.device, dtype=self.dtype)
            rhs = torch.zeros((3, positions, feature_count), device=self.device, dtype=self.dtype)
            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                triplets = torch.stack(
                    [tensor_batch(Y_t[start:stop, item.flat_slice], self.device, self.dtype) for item in descriptors],
                    dim=-1,
                )
                targets = torch.stack(
                    [tensor_batch(Y_0[start:stop, item.flat_slice], self.device, self.dtype) for item in descriptors],
                    dim=-1,
                )
                for axis, orientation in enumerate(("H", "V", "D")):
                    features = band_features(triplets, band_monomial_exponents(self.degree, orientation))
                    gram[axis].add_(torch.einsum("nkp,nkq->kpq", features, features))
                    rhs[axis].add_(torch.einsum("nkp,nk->kp", features, targets[..., axis]))
            self.detail_coefficients_[level] = solve_ridge(
                gram / num_samples, rhs / num_samples, self.ridge
            )
        self.layout_ = layout
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict clean coefficients and return them on CPU."""
        if self.layout_ is None or self.scaling_model_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        output_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        prediction = np.empty(Y_t.shape, dtype=output_dtype)
        scaling = self.layout_.subbands[0]
        prediction[:, scaling.flat_slice] = self.scaling_model_.predict(
            Y_t[:, scaling.flat_slice], batch_size
        )
        for start in range(0, len(Y_t), batch_size):
            stop = min(start + batch_size, len(Y_t))
            for level, descriptors in detail_descriptors_by_level(self.layout_).items():
                triplets = torch.stack(
                    [tensor_batch(Y_t[start:stop, item.flat_slice], self.device, self.dtype) for item in descriptors],
                    dim=-1,
                )
                for axis, (orientation, descriptor) in enumerate(zip(("H", "V", "D"), descriptors, strict=True)):
                    features = band_features(triplets, band_monomial_exponents(self.degree, orientation))
                    result = torch.einsum(
                        "nkp,kp->nk", features, self.detail_coefficients_[level][axis]
                    )
                    prediction[start:stop, descriptor.flat_slice] = result.detach().cpu().numpy()
        return prediction


@dataclass(slots=True)
class TorchLocalCoupledPolynomialRegressor:
    degree: int
    radius: int
    ridge: float
    device: torch.device
    dtype: torch.dtype
    coefficients_: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    layout_: WaveletLayout | None = field(default=None, init=False)

    @staticmethod
    def _maps(Y_t: np.ndarray, descriptor: SubbandDescriptor) -> np.ndarray:
        return Y_t[:, descriptor.flat_slice].reshape(len(Y_t), *descriptor.shape)

    def fit(self, Y_t: np.ndarray, Y_0: np.ndarray, layout: WaveletLayout, batch_size: int = 1024):
        """Fit same-subband local systems using CUDA-capable batched contractions."""
        validate_local_radius(layout, self.radius)
        terms = local_monomial_terms(self.degree, self.radius)
        num_samples = len(Y_t)
        self.coefficients_ = {}
        for descriptor in layout.subbands:
            gram = torch.zeros((descriptor.size, len(terms), len(terms)), device=self.device, dtype=self.dtype)
            rhs = torch.zeros((descriptor.size, len(terms)), device=self.device, dtype=self.dtype)
            noisy_maps = self._maps(Y_t, descriptor)
            targets = Y_0[:, descriptor.flat_slice]
            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                maps = tensor_batch(noisy_maps[start:stop], self.device, self.dtype)
                target = tensor_batch(targets[start:stop], self.device, self.dtype)
                features = local_features(maps, terms)
                gram.add_(torch.einsum("nkp,nkq->kpq", features, features))
                rhs.add_(torch.einsum("nkp,nk->kp", features, target))
            self.coefficients_[descriptor.subband_index] = solve_ridge(
                gram / num_samples, rhs / num_samples, self.ridge
            )
        self.layout_ = layout
        return self

    def predict(self, Y_t: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict clean coefficients with local models and return them on CPU."""
        if self.layout_ is None:
            raise RuntimeError("The regressor must be fitted before prediction.")
        output_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        prediction = np.empty(Y_t.shape, dtype=output_dtype)
        terms = local_monomial_terms(self.degree, self.radius)
        for descriptor in self.layout_.subbands:
            maps = self._maps(Y_t, descriptor)
            for start in range(0, len(Y_t), batch_size):
                stop = min(start + batch_size, len(Y_t))
                features = local_features(
                    tensor_batch(maps[start:stop], self.device, self.dtype), terms
                )
                result = torch.einsum(
                    "nkp,kp->nk", features, self.coefficients_[descriptor.subband_index]
                )
                prediction[start:stop, descriptor.flat_slice] = result.detach().cpu().numpy()
        return prediction
