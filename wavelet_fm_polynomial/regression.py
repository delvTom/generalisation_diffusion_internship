"""Unregularized dense readouts for nested wavelet feature dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from wavelet_fm_polynomial.features import (
    FeatureFamily,
    WaveletFeatureStructure,
    evaluate_features,
    feature_count,
)


@dataclass(slots=True)
class PolynomialVelocityRegressor:
    family: FeatureFamily
    degree: int
    structure: WaveletFeatureStructure
    feature_mean_: torch.Tensor
    feature_scale_: torch.Tensor
    target_mean_: torch.Tensor
    weights_: torch.Tensor
    ridge: float
    diagnostics: dict[str, float | str]

    @property
    def device(self) -> torch.device:
        return self.weights_.device

    @torch.no_grad()
    def predict_tensor(self, Y_t: torch.Tensor) -> torch.Tensor:
        Y_t = Y_t.to(device=self.device, dtype=self.feature_mean_.dtype)
        features = evaluate_features(Y_t, self.structure, self.family, self.degree)
        normalized = (features - self.feature_mean_) / self.feature_scale_
        return self.target_mean_ + normalized @ self.weights_

    @torch.no_grad()
    def mse(
        self,
        Y_t: np.ndarray,
        targets: np.ndarray,
        *,
        batch_size: int,
    ) -> float:
        if Y_t.shape != targets.shape or Y_t.ndim != 2:
            raise ValueError("Y_t and targets must have the same matrix shape.")
        squared_error = torch.zeros((), device=self.device, dtype=torch.float64)
        for start in range(0, len(Y_t), batch_size):
            stop = min(start + batch_size, len(Y_t))
            inputs = torch.as_tensor(Y_t[start:stop], device=self.device)
            expected = torch.as_tensor(
                targets[start:stop], device=self.device, dtype=self.target_mean_.dtype
            )
            prediction = self.predict_tensor(inputs)
            squared_error.add_((prediction - expected).square().sum(dtype=torch.float64))
        return float((squared_error / targets.size).item())


@torch.no_grad()
def fit_nested_degrees(
    Y_t: np.ndarray,
    targets: np.ndarray,
    *,
    structure: WaveletFeatureStructure,
    family: FeatureFamily,
    degrees: tuple[int, ...],
    device: torch.device,
    batch_size: int = 1024,
    moment_dtype: torch.dtype = torch.float32,
    solve_dtype: torch.dtype = torch.float64,
    ridge: float = 0.0,
    minimum_feature_scale: float = 1e-12,
) -> dict[int, PolynomialVelocityRegressor]:
    """Fit several nested degrees from one maximum-degree moment matrix.

    Raw feature products are formed in ``moment_dtype`` for GPU throughput and
    accumulated/solved in ``solve_dtype``. No regularization is added unless
    ``ridge`` is explicitly nonzero.
    """
    if Y_t.shape != targets.shape or Y_t.ndim != 2:
        raise ValueError("Y_t and targets must have the same matrix shape.")
    if not degrees or min(degrees) < 2 or len(set(degrees)) != len(degrees):
        raise ValueError("degrees must contain distinct values greater than or equal to two.")
    if batch_size < 1 or ridge < 0.0 or minimum_feature_scale <= 0.0:
        raise ValueError("Invalid batch size, ridge, or feature-scale threshold.")
    degrees = tuple(sorted(degrees))
    max_degree = max(degrees)
    num_samples, output_dimension = targets.shape
    max_features = feature_count(
        structure, family, max_degree, include_intercept=False
    )
    if num_samples <= max_features:
        raise ValueError(
            f"Unregularized {family} D={max_degree} has {max_features} nonconstant "
            f"features but only {num_samples} fit samples. Use more than "
            f"{max_features} samples to avoid an underdetermined interpolation problem."
        )

    start_time = perf_counter()
    sum_features = torch.zeros(max_features, device=device, dtype=solve_dtype)
    sum_targets = torch.zeros(output_dimension, device=device, dtype=solve_dtype)
    raw_gram = torch.zeros(
        (max_features, max_features), device=device, dtype=solve_dtype
    )
    raw_rhs = torch.zeros(
        (max_features, output_dimension), device=device, dtype=solve_dtype
    )

    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        inputs = torch.as_tensor(Y_t[start:stop], device=device, dtype=moment_dtype)
        expected = torch.as_tensor(targets[start:stop], device=device, dtype=moment_dtype)
        features = evaluate_features(inputs, structure, family, max_degree)
        sum_features.add_(features.sum(dim=0, dtype=solve_dtype))
        sum_targets.add_(expected.sum(dim=0, dtype=solve_dtype))
        raw_gram.add_((features.T @ features).to(solve_dtype))
        raw_rhs.add_((features.T @ expected).to(solve_dtype))

    feature_mean = sum_features / num_samples
    target_mean = sum_targets / num_samples
    covariance = raw_gram / num_samples - torch.outer(feature_mean, feature_mean)
    covariance = 0.5 * (covariance + covariance.T)
    cross_covariance = raw_rhs / num_samples - torch.outer(feature_mean, target_mean)
    feature_variance = covariance.diagonal().clone()
    if float(feature_variance.min().item()) < -minimum_feature_scale:
        raise RuntimeError(
            f"{family} feature covariance has negative diagonal entries; "
            "use higher moment precision."
        )
    feature_scale = feature_variance.clamp_min(0.0).sqrt()
    inactive = feature_scale <= minimum_feature_scale
    if bool(inactive.any()):
        indices = torch.nonzero(inactive, as_tuple=False).flatten()[:8].tolist()
        raise RuntimeError(
            f"{family} has {int(inactive.sum().item())} zero-variance features "
            f"(first indices: {indices}). Remove them explicitly before fitting."
        )
    covariance.div_(feature_scale[:, None])
    covariance.div_(feature_scale[None, :])
    cross_covariance.div_(feature_scale[:, None])

    models: dict[int, PolynomialVelocityRegressor] = {}
    for degree in degrees:
        count = feature_count(structure, family, degree, include_intercept=False)
        system = covariance[:count, :count].clone()
        rhs = cross_covariance[:count].clone()
        if ridge:
            system.diagonal().add_(ridge)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if int(info.max().item()) != 0:
            raise RuntimeError(
                f"Cholesky failed for {family} D={degree} with ridge={ridge:g}. "
                "The dictionary is rank-deficient or numerically ill-conditioned. "
                "Retry with --moment-dtype float64; use a ridge only as a separately "
                "reported diagnostic."
            )
        weights = torch.cholesky_solve(rhs, factor)
        rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(solve_dtype).tiny)
        relative_residual = torch.linalg.vector_norm(system @ weights - rhs) / rhs_norm
        if not torch.isfinite(weights).all() or not torch.isfinite(relative_residual):
            raise RuntimeError(f"Non-finite solution for {family} D={degree}.")
        if float(relative_residual.item()) > 1e-8:
            raise RuntimeError(
                f"Inaccurate solve for {family} D={degree}: "
                f"relative residual={relative_residual.item():.3e}."
            )
        models[degree] = PolynomialVelocityRegressor(
            family=family,
            degree=degree,
            structure=structure,
            feature_mean_=feature_mean[:count].clone(),
            feature_scale_=feature_scale[:count].clone(),
            target_mean_=target_mean.clone(),
            weights_=weights,
            ridge=ridge,
            diagnostics={
                "num_samples": float(num_samples),
                "num_nonconstant_features": float(count),
                "ridge": float(ridge),
                "relative_normal_equation_residual": float(relative_residual.item()),
                "moment_seconds": perf_counter() - start_time,
                "moment_dtype": str(moment_dtype),
                "solve_dtype": str(solve_dtype),
            },
        )
    return models
