"""Audited monomial features for orientation-coupled wavelet details."""

from __future__ import annotations

from typing import Final

import numpy as np

from wavelet_score.indexing import Orientation

ORIENTATION_TO_AXIS: Final[dict[Orientation, int]] = {"H": 0, "V": 1, "D": 2}


def band_monomial_exponents(
    degree: int,
    target_orientation: Orientation,
) -> tuple[tuple[int, int, int], ...]:
    """List [constant, target-containing monomials] in paper-loop order."""
    if degree < 1:
        raise ValueError("degree must be at least one.")
    if target_orientation not in ORIENTATION_TO_AXIS:
        raise ValueError("target_orientation must be H, V, or D.")

    target_axis = ORIENTATION_TO_AXIS[target_orientation]
    other_axes = tuple(axis for axis in range(3) if axis != target_axis)
    exponents: list[tuple[int, int, int]] = [(0, 0, 0)]

    for target_power in range(1, degree + 1):
        remaining_after_target = degree - target_power
        for first_power in range(remaining_after_target + 1):
            remaining_after_first = remaining_after_target - first_power
            for second_power in range(remaining_after_first + 1):
                exponent = [0, 0, 0]
                exponent[target_axis] = target_power
                exponent[other_axes[0]] = first_power
                exponent[other_axes[1]] = second_power
                exponents.append(tuple(exponent))
    return tuple(exponents)


def evaluate_band_monomials(
    triplets: np.ndarray,
    exponents: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Evaluate features on [...,3] H/V/D triplets, returning [...,P]."""
    if not isinstance(triplets, np.ndarray) or triplets.ndim < 2:
        raise ValueError("triplets must be a numpy array with shape [...,3].")
    if triplets.shape[-1] != 3:
        raise ValueError("The last triplet axis must follow the order [H,V,D].")
    if triplets.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("triplets must use float32 or float64.")
    if not exponents:
        raise ValueError("At least one exponent tuple is required.")
    if any(len(exponent) != 3 or min(exponent) < 0 for exponent in exponents):
        raise ValueError("Every exponent must contain three non-negative integers.")

    features = np.ones((*triplets.shape[:-1], len(exponents)), dtype=triplets.dtype)
    for feature_index, exponent in enumerate(exponents):
        for orientation_axis, power in enumerate(exponent):
            if power:
                features[..., feature_index] *= triplets[..., orientation_axis] ** power
    return features
