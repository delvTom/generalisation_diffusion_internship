"""Numerical helpers for the spatial-radius mask experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class FlowMatchingQueries:
    clean_indices: np.ndarray
    clean_Y0: np.ndarray
    noise: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True, slots=True)
class CleanRisk:
    clean_mse: float
    clean_variance: float
    normalized_clean_risk: float
    identity_max_abs_error: float


def sample_flow_matching_queries(
    reference_Y0: np.ndarray,
    num_queries: int,
    seed: int,
) -> FlowMatchingQueries:
    """Draw paired empirical-clean/noise queries reproducibly."""
    if (
        reference_Y0.ndim != 2
        or len(reference_Y0) == 0
        or reference_Y0.dtype not in (np.dtype(np.float32), np.dtype(np.float64))
        or not np.isfinite(reference_Y0).all()
    ):
        raise ValueError(
            "reference_Y0 must be a non-empty finite float32 or float64 matrix."
        )
    if num_queries < 1:
        raise ValueError("num_queries must be positive.")
    rng = np.random.default_rng(seed)
    clean_indices = rng.integers(0, len(reference_Y0), size=num_queries)
    clean_Y0 = np.ascontiguousarray(reference_Y0[clean_indices])
    noise = rng.standard_normal(clean_Y0.shape).astype(reference_Y0.dtype)
    return FlowMatchingQueries(
        clean_indices=clean_indices,
        clean_Y0=clean_Y0,
        noise=noise,
        velocity=clean_Y0 - noise,
    )


def evaluate_clean_risk(
    predicted_velocity: np.ndarray,
    target_velocity: np.ndarray,
    noisy_coefficient: np.ndarray,
    clean_coefficient: np.ndarray,
    time: float,
    clean_variance: float,
) -> CleanRisk:
    """Evaluate clean reconstruction and audit its FM velocity identity."""
    arrays = (
        predicted_velocity,
        target_velocity,
        noisy_coefficient,
        clean_coefficient,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("All coefficient arrays must be one-dimensional.")
    if len({len(array) for array in arrays}) != 1 or len(predicted_velocity) == 0:
        raise ValueError("Coefficient arrays must have one common positive length.")
    if any(array.dtype != np.dtype(np.float64) for array in arrays):
        raise TypeError("Clean-risk reference calculations require float64 arrays.")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Coefficient arrays must be finite.")
    if not 0.0 <= time < 1.0:
        raise ValueError("time must lie in [0,1).")
    if not np.isfinite(clean_variance) or clean_variance <= 0.0:
        raise ValueError("clean_variance must be finite and strictly positive.")

    predicted_clean = noisy_coefficient + (1.0 - time) * predicted_velocity
    clean_error = predicted_clean - clean_coefficient
    velocity_error_scaled = (1.0 - time) * (
        predicted_velocity - target_velocity
    )
    identity_error = float(np.max(np.abs(clean_error - velocity_error_scaled)))
    clean_mse = float(np.mean(clean_error**2))
    return CleanRisk(
        clean_mse=clean_mse,
        clean_variance=clean_variance,
        normalized_clean_risk=clean_mse / clean_variance,
        identity_max_abs_error=identity_error,
    )


def select_smallest_near_optimal_radius(
    normalized_risks: Mapping[int, float],
) -> int:
    """Select the smallest radius within 1 percent of maximal-radius risk."""
    if not normalized_risks:
        raise ValueError("At least one radius risk is required.")
    radii = sorted(normalized_risks)
    if radii != list(range(radii[-1] + 1)):
        raise ValueError("Radius risks must be provided contiguously from zero.")
    risks = np.asarray([normalized_risks[radius] for radius in radii], dtype=np.float64)
    if np.any(~np.isfinite(risks)) or np.any(risks < 0.0):
        raise ValueError("Radius risks must be finite and non-negative.")

    maximal_radius_risk = float(risks[-1])
    additive_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, abs(maximal_radius_risk)
    )
    threshold = 1.01 * maximal_radius_risk + additive_tolerance
    return next(radius for radius in radii if normalized_risks[radius] <= threshold)


def approximate_physical_radius_pixels(radius: int, pywt_level: int) -> int:
    """Convert a coefficient-grid radius to its center-spacing pixel scale."""
    if radius < 0 or pywt_level < 1:
        raise ValueError("radius must be non-negative and pywt_level positive.")
    return radius * 2**pywt_level
