"""Periodic same-band neighborhood features for local wavelet coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LocalMonomialTerm:
    """One center-only or center-neighbor monomial."""

    center_power: int
    neighbor_power: int
    offset: tuple[int, int] | None

    def __post_init__(self) -> None:
        if self.center_power < 0 or self.neighbor_power < 0:
            raise ValueError("Monomial powers must be non-negative.")
        if self.offset is None:
            if self.neighbor_power != 0:
                raise ValueError("A center-only term cannot have a neighbor power.")
        elif self.center_power < 1 or self.neighbor_power < 1:
            raise ValueError("A coupled term must contain both center and neighbor.")


def chebyshev_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """List nonzero offsets in a radius-r Chebyshev ball, in row-major order."""
    if radius < 1:
        raise ValueError("radius must be at least one.")
    return tuple(
        (delta_1, delta_2)
        for delta_1 in range(-radius, radius + 1)
        for delta_2 in range(-radius, radius + 1)
        if (delta_1, delta_2) != (0, 0)
    )


def local_monomial_terms(degree: int, radius: int) -> tuple[LocalMonomialTerm, ...]:
    """Build the paper's center powers and center-neighbor interaction terms."""
    if degree < 1:
        raise ValueError("degree must be at least one.")
    terms = [
        LocalMonomialTerm(center_power=power, neighbor_power=0, offset=None)
        for power in range(degree + 1)
    ]
    for offset in chebyshev_offsets(radius):
        for center_power in range(1, degree):
            for neighbor_power in range(1, degree - center_power + 1):
                terms.append(
                    LocalMonomialTerm(
                        center_power=center_power,
                        neighbor_power=neighbor_power,
                        offset=offset,
                    )
                )
    return tuple(terms)


def periodic_neighbor(
    coefficient_maps: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    """Return values at k+offset for every k using periodic boundaries."""
    if not isinstance(coefficient_maps, np.ndarray) or coefficient_maps.ndim != 3:
        raise ValueError("coefficient_maps must have shape [N,h,w].")
    if coefficient_maps.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("coefficient_maps must use float32 or float64.")
    delta_1, delta_2 = offset
    return np.roll(coefficient_maps, shift=(-delta_1, -delta_2), axis=(-2, -1))


def evaluate_local_monomials(
    coefficient_maps: np.ndarray,
    terms: tuple[LocalMonomialTerm, ...],
) -> np.ndarray:
    """Evaluate local features as [N,K,P], with K flattened row-major."""
    if not isinstance(coefficient_maps, np.ndarray) or coefficient_maps.ndim != 3:
        raise ValueError("coefficient_maps must have shape [N,h,w].")
    if coefficient_maps.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("coefficient_maps must use float32 or float64.")
    if coefficient_maps.shape[0] == 0 or min(coefficient_maps.shape[1:]) == 0:
        raise ValueError("coefficient_maps must be non-empty.")
    if not terms:
        raise ValueError("At least one local monomial term is required.")

    num_samples, height, width = coefficient_maps.shape
    features = np.empty(
        (num_samples, height * width, len(terms)), dtype=coefficient_maps.dtype
    )
    neighbor_cache: dict[tuple[int, int], np.ndarray] = {}
    for feature_index, term in enumerate(terms):
        values = coefficient_maps**term.center_power
        if term.offset is not None:
            if term.offset not in neighbor_cache:
                neighbor_cache[term.offset] = periodic_neighbor(coefficient_maps, term.offset)
            neighbor = neighbor_cache[term.offset]
            values = values * neighbor**term.neighbor_power
        features[..., feature_index] = values.reshape(num_samples, height * width)
    return features
