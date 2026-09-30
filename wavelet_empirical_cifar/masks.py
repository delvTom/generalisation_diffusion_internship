"""RGB-site masks for Wiener sensitivity and coarse wavelet context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wavelet_coarse_to_fine.masks import _aligned_center
from wavelet_fm_structure.masks import build_layout_radius_masks
from wavelet_score.indexing import WaveletLayout, build_metadata
from wavelet_score.mask_groups import WaveletMaskGroup


CoarseAmount = Literal["none", "center", "local", "all"]
Family = Literal["same_band", "orientation_augmented", "scale_augmented", "full"]


def expand_sites(
    sites: np.ndarray | list[int], dimension: int, channels: int = 3
) -> np.ndarray:
    """Flatten complete RGB sites in site-major order."""
    values = np.asarray(sites, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or len(np.unique(values)) != len(values):
        raise ValueError("sites must be a nonempty vector without duplicates")
    if np.any(values < 0) or np.any(values >= dimension):
        raise ValueError("site outside the scalar wavelet layout")
    return np.asarray(
        [channel * dimension + int(site) for site in values for channel in range(channels)],
        dtype=np.int64,
    )


def site_sensitivity(denoiser: np.ndarray, dimension: int) -> np.ndarray:
    """Frobenius norm of every 3x3 input/output channel block, shape [d,d]."""
    if denoiser.shape != (3 * dimension, 3 * dimension):
        raise ValueError("denoiser must be a full RGB wavelet matrix")
    # Output and input are channel-major in the repository.
    blocks = denoiser.reshape(3, dimension, 3, dimension)
    return np.sqrt(np.sum(blocks * blocks, axis=(0, 2), dtype=np.float64))


def top_sensitivity_sites(scores: np.ndarray, target: int, budget: int) -> np.ndarray:
    if scores.ndim != 1 or not 1 <= budget <= len(scores):
        raise ValueError("invalid score vector or site budget")
    if not 0 <= target < len(scores):
        raise ValueError("target outside the score vector")
    # Stable sorting makes equal scores reproducible across devices.
    order = np.argsort(-scores, kind="stable")
    selected = np.concatenate(([target], order[order != target][: budget - 1]))
    return selected.astype(np.int64)


def groups_from_sites(
    layout: WaveletLayout,
    selections: list[np.ndarray],
    family: str,
    channels: int = 3,
) -> tuple[WaveletMaskGroup, ...]:
    metadata = build_metadata(layout)
    if len(selections) != layout.num_coefficients:
        raise ValueError("one site selection per output site is required")
    groups = []
    for target, sites in zip(metadata, selections, strict=True):
        if target.flat_index not in sites:
            raise ValueError("every selected mask must observe its target")
        groups.append(
            WaveletMaskGroup(
                input_indices=expand_sites(sites, layout.num_coefficients, channels),
                output_indices=np.asarray(
                    [channel * layout.num_coefficients + target.flat_index for channel in range(channels)],
                    dtype=np.int64,
                ),
                family=family,
                coefficient_type=target.coefficient_type,
                pywt_level=target.pywt_level,
                center=(target.k1, target.k2),
                orientations=() if target.orientation is None else (target.orientation,),
            )
        )
    return tuple(groups)


def geometric_sites(layout: WaveletLayout, budget: int) -> list[np.ndarray]:
    """Nearest periodic sites in the target's own subband, target first."""
    if budget < 1:
        raise ValueError("budget must be positive")
    result = []
    for target in build_metadata(layout):
        band = layout.subbands[target.subband_index]
        height, width = band.shape
        local_budget = min(budget, band.size)
        positions = []
        for row in range(height):
            for col in range(width):
                delta_row = min((row - target.k1) % height, (target.k1 - row) % height)
                delta_col = min((col - target.k2) % width, (target.k2 - col) % width)
                positions.append((max(delta_row, delta_col), delta_row + delta_col, row, col))
        positions.sort()
        sites = [target.flat_index]
        if local_budget == 1:
            result.append(np.asarray(sites, dtype=np.int64))
            continue
        for _, _, row, col in positions:
            site = band.flat_start + row * width + col
            if site != target.flat_index:
                sites.append(site)
            if len(sites) == local_budget:
                break
        result.append(np.asarray(sites, dtype=np.int64))
    return result


def sensitivity_groups(
    layout: WaveletLayout, denoiser: np.ndarray, budget: int
) -> tuple[WaveletMaskGroup, ...]:
    scores = site_sensitivity(denoiser, layout.num_coefficients)
    metadata = build_metadata(layout)
    sites = [top_sensitivity_sites(scores[target], target, min(budget, layout.subbands[item.subband_index].size))
             for target, item in enumerate(metadata)]
    return groups_from_sites(layout, sites, "wiener_sensitivity")


def artem_threshold_groups(
    layout: WaveletLayout, denoiser: np.ndarray, tau: float = 0.05,
) -> tuple[WaveletMaskGroup, ...]:
    """Apply Artem's scalar row/diagonal/global-max rule before RGB-site union."""
    dimension = layout.num_coefficients
    if denoiser.shape != (3 * dimension, 3 * dimension) or not 0.0 < tau < 1.0:
        raise ValueError("expected a full RGB denoiser and tau in (0,1)")
    diagonal = np.diag(denoiser).copy()
    diagonal[np.abs(diagonal) < 1e-6] = 1.0
    normalized = denoiser / diagonal[:, None]
    scalar_mask = np.abs(normalized) >= tau * np.max(np.abs(normalized))
    site_mask = scalar_mask.reshape(3, dimension, 3, dimension).any(axis=(0, 2))
    np.fill_diagonal(site_mask, True)
    selections = [np.concatenate(([q], np.flatnonzero(site_mask[q] & (np.arange(dimension) != q))))
                  for q in range(dimension)]
    return groups_from_sites(layout, selections, "artem_threshold")


def geometric_groups(layout: WaveletLayout, budget: int) -> tuple[WaveletMaskGroup, ...]:
    return groups_from_sites(layout, geometric_sites(layout, budget), "geometric_budget")


def coarse_context_sites(layout: WaveletLayout, target: int, amount: CoarseAmount) -> list[int]:
    if amount == "none":
        return []
    if amount not in ("center", "local", "all"):
        raise ValueError("unknown coarse context amount")
    coarse = layout.subbands[0]
    descriptor = build_metadata(layout)[target]
    center = _aligned_center(descriptor, coarse.shape)
    height, width = coarse.shape
    if amount == "all":
        return list(range(coarse.flat_start, coarse.flat_stop))
    radius = 0 if amount == "center" else 1
    return list(dict.fromkeys(
        coarse.flat_start + ((center[0] + dr) % height) * width + (center[1] + dc) % width
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
    ))


def add_coarse_context(
    layout: WaveletLayout,
    baseline: tuple[WaveletMaskGroup, ...],
    amount: CoarseAmount,
) -> tuple[WaveletMaskGroup, ...]:
    """Extend only level-1 detail outputs; all other outputs retain their masks."""
    if len(baseline) != layout.num_coefficients:
        raise ValueError("baseline must predict each RGB wavelet site separately")
    result = []
    metadata_all = build_metadata(layout)
    for target, group in enumerate(baseline):
        metadata = metadata_all[target]
        if metadata.coefficient_type != "detail" or metadata.pywt_level != 1:
            result.append(group)
            continue
        d = layout.num_coefficients
        sites = list(dict.fromkeys((int(index) % d for index in group.input_indices)))
        sites.extend(site for site in coarse_context_sites(layout, target, amount) if site not in sites)
        result.append(
            WaveletMaskGroup(
                expand_sites(sites, d),
                group.output_indices,
                f"{group.family}_coarse_{amount}",
                group.coefficient_type,
                group.pywt_level,
                group.center,
                group.orientations,
            )
        )
    return tuple(result)


def family_groups(
    layout: WaveletLayout, family: Family, radius: int = 1
) -> tuple[WaveletMaskGroup, ...]:
    scalar = build_layout_radius_masks(
        layout, {f"cA{layout.decomposition_level}": radius, **{
            f"c{orientation}{level}": radius
            for level in range(layout.decomposition_level, 0, -1)
            for orientation in ("H", "V", "D")
        }}
    )[family]
    sites = [np.asarray(group.flat_indices, dtype=np.int64) for group in scalar]
    return groups_from_sites(layout, sites, family)
