"""Coarse-to-fine wavelet masks for empirical Flow Matching."""

from wavelet_coarse_to_fine.masks import (
    CoarseToFineSchedule,
    build_coarse_to_fine_rgb_groups,
    build_rgb_region_indices,
)

__all__ = [
    "CoarseToFineSchedule",
    "build_coarse_to_fine_rgb_groups",
    "build_rgb_region_indices",
]
