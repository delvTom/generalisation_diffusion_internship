"""Second- and fourth-order patch statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_gsm_local.patches import CENTER_INDEX


@dataclass(frozen=True, slots=True)
class EnergyStatistics:
    mean: np.ndarray
    covariance: np.ndarray
    signed_correlation: np.ndarray
    square_correlation: np.ndarray
    fourth_order_excess: np.ndarray


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - first.mean(axis=0, keepdims=True)
    second_centered = second - second.mean(axis=0, keepdims=True)
    covariance = np.mean(first_centered * second_centered, axis=0)
    denominator = np.sqrt(
        np.mean(first_centered**2, axis=0) * np.mean(second_centered**2, axis=0)
    )
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)


def compute_energy_statistics(patches: np.ndarray) -> EnergyStatistics:
    if patches.ndim != 2 or patches.shape[1] != 9:
        raise ValueError("patches must have shape [N,9].")
    patches = np.asarray(patches, dtype=np.float64)
    mean = patches.mean(axis=0)
    centered = patches - mean
    covariance = centered.T @ centered / len(centered)
    center = centered[:, CENTER_INDEX : CENTER_INDEX + 1]
    signed = _safe_correlation(center, centered)
    squared = _safe_correlation(center**2, centered**2)
    cross_second = np.mean(center * centered, axis=0)
    fourth = (
        np.mean(center**2 * centered**2, axis=0)
        - np.mean(center**2, axis=0) * np.mean(centered**2, axis=0)
        - 2.0 * cross_second**2
    )
    return EnergyStatistics(
        mean=mean,
        covariance=0.5 * (covariance + covariance.T),
        signed_correlation=signed,
        square_correlation=squared,
        fourth_order_excess=fourth,
    )


def gsm_fourth_order_excess(
    covariance: np.ndarray,
    mixing_weights: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Predicted center-to-offset fourth cumulants under E[S]=1."""
    variance_scale = float(np.dot(mixing_weights, scales**2) - 1.0)
    center_variance = covariance[CENTER_INDEX, CENTER_INDEX]
    return variance_scale * (
        center_variance * np.diag(covariance)
        + 2.0 * covariance[CENTER_INDEX] ** 2
    )
