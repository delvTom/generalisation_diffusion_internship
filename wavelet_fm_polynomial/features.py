"""Nested polynomial dictionaries on a critical 2D wavelet transform."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
import torch

from wavelet_score.indexing import WaveletLayout, build_metadata

FeatureFamily = Literal["independent", "band", "local"]


@dataclass(frozen=True, slots=True)
class WaveletFeatureStructure:
    """Indices needed by the three global wavelet feature dictionaries.

    ``orientation_triplets`` groups H/V/D coefficients at the same scale and
    spatial position. ``local_pairs`` contains each unordered spatial edge
    exactly once and never connects two different subbands.
    """

    num_coefficients: int
    orientation_triplets: np.ndarray  # [G,3]
    local_pairs: np.ndarray  # [E,2], unique unordered pairs
    local_radius: int

    def __post_init__(self) -> None:
        triplets = np.asarray(self.orientation_triplets, dtype=np.int64)
        pairs = np.asarray(self.local_pairs, dtype=np.int64)
        if triplets.ndim != 2 or triplets.shape[1:] != (3,):
            raise ValueError("orientation_triplets must have shape [G,3].")
        if pairs.ndim != 2 or pairs.shape[1:] != (2,):
            raise ValueError("local_pairs must have shape [E,2].")
        if self.num_coefficients < 1 or self.local_radius < 1:
            raise ValueError("num_coefficients and local_radius must be positive.")
        if triplets.size and (triplets.min() < 0 or triplets.max() >= self.num_coefficients):
            raise ValueError("An orientation index lies outside the coefficient vector.")
        if pairs.size and (pairs.min() < 0 or pairs.max() >= self.num_coefficients):
            raise ValueError("A local-pair index lies outside the coefficient vector.")
        canonical_pairs = np.sort(pairs, axis=1)
        if len(canonical_pairs) and (
            np.any(canonical_pairs[:, 0] == canonical_pairs[:, 1])
            or len(np.unique(canonical_pairs, axis=0)) != len(canonical_pairs)
        ):
            raise ValueError("local_pairs must be distinct edges without self-pairs.")
        triplets.setflags(write=False)
        canonical_pairs.setflags(write=False)
        object.__setattr__(self, "orientation_triplets", triplets)
        object.__setattr__(self, "local_pairs", canonical_pairs)


def build_feature_structure(
    layout: WaveletLayout,
    *,
    local_radius: int = 1,
) -> WaveletFeatureStructure:
    """Build orientation groups and within-subband Chebyshev neighborhoods."""
    if local_radius < 1:
        raise ValueError("local_radius must be at least one.")
    metadata = build_metadata(layout)

    detail_lookup = {
        (item.pywt_level, item.k1, item.k2, item.orientation): item.flat_index
        for item in metadata
        if item.coefficient_type == "detail"
    }
    detail_positions = sorted(
        {(level, k1, k2) for level, k1, k2, _ in detail_lookup}
    )
    triplets = np.asarray(
        [
            [detail_lookup[(level, k1, k2, orientation)] for orientation in ("H", "V", "D")]
            for level, k1, k2 in detail_positions
        ],
        dtype=np.int64,
    )

    local_pairs: list[tuple[int, int]] = []
    for band in layout.subbands:
        height, width = band.shape
        for k1 in range(height):
            for k2 in range(width):
                first = band.flat_start + k1 * width + k2
                for delta_1 in range(0, local_radius + 1):
                    delta_2_start = 1 if delta_1 == 0 else -local_radius
                    for delta_2 in range(delta_2_start, local_radius + 1):
                        neighbor_1 = k1 + delta_1
                        neighbor_2 = k2 + delta_2
                        if not (0 <= neighbor_1 < height and 0 <= neighbor_2 < width):
                            continue
                        second = band.flat_start + neighbor_1 * width + neighbor_2
                        local_pairs.append((first, second))

    pairs_array = np.asarray(local_pairs, dtype=np.int64)
    if not len(pairs_array):
        pairs_array = np.empty((0, 2), dtype=np.int64)
    return WaveletFeatureStructure(
        num_coefficients=layout.num_coefficients,
        orientation_triplets=triplets,
        local_pairs=pairs_array,
        local_radius=local_radius,
    )


def _coupled_orientation_exponents(total_degree: int) -> tuple[tuple[int, int, int], ...]:
    """All degree-exact H/V/D monomials involving at least two variables."""
    return tuple(
        exponent
        for exponent in product(range(total_degree + 1), repeat=3)
        if sum(exponent) == total_degree and sum(power > 0 for power in exponent) >= 2
    )


def _local_exponents(total_degree: int) -> tuple[tuple[int, int], ...]:
    """Degree-exact bivariate monomials in which both neighbors occur."""
    return tuple(
        (first, total_degree - first)
        for first in range(1, total_degree)
    )


def feature_count(
    structure: WaveletFeatureStructure,
    family: FeatureFamily,
    degree: int,
    *,
    include_intercept: bool = True,
) -> int:
    """Return the cumulative number of unique features through ``degree``."""
    if degree < 1:
        raise ValueError("degree must be at least one.")
    count = structure.num_coefficients * degree
    if family == "band":
        count += len(structure.orientation_triplets) * sum(
            len(_coupled_orientation_exponents(total))
            for total in range(2, degree + 1)
        )
    elif family == "local":
        count += len(structure.local_pairs) * sum(
            len(_local_exponents(total)) for total in range(2, degree + 1)
        )
    elif family != "independent":
        raise ValueError(f"Unknown feature family: {family!r}.")
    return count + int(include_intercept)


def evaluate_features(
    Y_t: torch.Tensor,
    structure: WaveletFeatureStructure,
    family: FeatureFamily,
    degree: int,
) -> torch.Tensor:
    """Evaluate nonconstant features in a degree-nested deterministic order.

    Every family starts with all linear coefficients in their original flat
    order. Consequently the degree-one dictionary is exactly ``Y_t`` for all
    three families. The intercept is handled separately by the regressor.
    """
    if Y_t.ndim != 2 or Y_t.shape[1] != structure.num_coefficients:
        raise ValueError(
            f"Y_t must have shape [N,{structure.num_coefficients}], got {tuple(Y_t.shape)}."
        )
    if degree < 1:
        raise ValueError("degree must be at least one.")
    if family not in ("independent", "band", "local"):
        raise ValueError(f"Unknown feature family: {family!r}.")

    triplets = None
    pairs = None
    if family == "band" and degree >= 2:
        indices = torch.tensor(
            structure.orientation_triplets, device=Y_t.device, dtype=torch.long
        )
        triplets = Y_t[:, indices]
    if family == "local" and degree >= 2:
        indices = torch.tensor(
            structure.local_pairs, device=Y_t.device, dtype=torch.long
        )
        pairs = Y_t[:, indices]

    blocks: list[torch.Tensor] = []
    for total_degree in range(1, degree + 1):
        blocks.append(Y_t.pow(total_degree))
        if triplets is not None:
            for exponent in _coupled_orientation_exponents(total_degree):
                values = torch.ones(
                    triplets.shape[:2], device=Y_t.device, dtype=Y_t.dtype
                )
                for axis, power in enumerate(exponent):
                    if power:
                        values = values * triplets[..., axis].pow(power)
                blocks.append(values)
        if pairs is not None:
            first = pairs[..., 0]
            second = pairs[..., 1]
            blocks.extend(
                first.pow(first_power) * second.pow(second_power)
                for first_power, second_power in _local_exponents(total_degree)
            )
    features = torch.cat(blocks, dim=1)
    expected = feature_count(
        structure, family, degree, include_intercept=False
    )
    if features.shape != (Y_t.shape[0], expected):
        raise RuntimeError(
            f"Constructed features have shape {tuple(features.shape)}; expected "
            f"({Y_t.shape[0]}, {expected})."
        )
    return features
