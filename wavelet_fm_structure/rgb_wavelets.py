"""Channelwise RGB wavelets and spatial masks shared across color channels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from wavelet_fm_structure.adaptive_radius import RADIUS_FAMILIES
from wavelet_fm_structure.masks import (
    RadiusMaskFamily,
    ReceptiveMask,
    build_central_targets,
    build_layout_radius_masks,
    max_nonaliasing_radius,
)
from wavelet_score.indexing import WaveletLayout
from wavelet_score.mask_groups import WaveletMaskGroup
from wavelet_score.transforms import WaveletTransform2D


def analyze_rgb(
    images: np.ndarray,
    transform: WaveletTransform2D,
) -> tuple[np.ndarray, WaveletLayout]:
    """Apply the same 2D DWT independently to images shaped [N,C,H,W]."""
    if not isinstance(images, np.ndarray) or images.ndim != 4:
        raise ValueError("images must be a numpy array with shape [N,C,H,W].")
    if images.shape[0] == 0 or images.shape[1] == 0:
        raise ValueError("images must contain at least one sample and one channel.")
    if images.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("images must use float32 or float64.")

    channel_coefficients: list[np.ndarray] = []
    shared_layout: WaveletLayout | None = None
    for channel in range(images.shape[1]):
        Y_t_channel, layout = transform.analyze(
            np.ascontiguousarray(images[:, channel])
        )
        if shared_layout is None:
            shared_layout = layout
        elif layout != shared_layout:
            raise RuntimeError("All RGB channels must produce the same wavelet layout.")
        channel_coefficients.append(Y_t_channel)
    if shared_layout is None:
        raise RuntimeError("Wavelet analysis produced no channels.")
    return np.stack(channel_coefficients, axis=1), shared_layout


def synthesize_rgb(
    Y_t: np.ndarray,
    layout: WaveletLayout,
    transform: WaveletTransform2D,
) -> np.ndarray:
    """Invert channelwise coefficients shaped [N,C,d] into [N,C,H,W]."""
    if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 3:
        raise ValueError("Y_t must be a numpy array with shape [N,C,d].")
    if Y_t.shape[0] == 0 or Y_t.shape[1] == 0:
        raise ValueError("Y_t must contain at least one sample and one channel.")
    if Y_t.shape[2] != layout.num_coefficients:
        raise ValueError("Y_t coefficient width does not match the scalar layout.")
    channels = [transform.synthesize(Y_t[:, channel], layout) for channel in range(Y_t.shape[1])]
    return np.stack(channels, axis=1)


def flatten_rgb(Y_t: np.ndarray) -> np.ndarray:
    """Flatten [N,C,d] coefficients in channel-major order into [N,C*d]."""
    if not isinstance(Y_t, np.ndarray) or Y_t.ndim != 3:
        raise ValueError("Y_t must be a numpy array with shape [N,C,d].")
    return np.ascontiguousarray(Y_t).reshape(Y_t.shape[0], -1)


def unflatten_rgb(
    Y_t_flat: np.ndarray,
    layout: WaveletLayout,
    channels: int = 3,
) -> np.ndarray:
    """Restore channel-major [N,C*d] coefficients to [N,C,d]."""
    if not isinstance(Y_t_flat, np.ndarray) or Y_t_flat.ndim != 2:
        raise ValueError("Y_t_flat must be a numpy array with shape [N,C*d].")
    if channels < 1 or Y_t_flat.shape[1] != channels * layout.num_coefficients:
        raise ValueError("Y_t_flat width must equal channels * layout size.")
    return np.ascontiguousarray(Y_t_flat).reshape(
        Y_t_flat.shape[0], channels, layout.num_coefficients
    )


def _expand_masks_rgb(
    masks: Sequence[ReceptiveMask],
    layout: WaveletLayout,
    channels: int,
) -> tuple[WaveletMaskGroup, ...]:
    if channels < 1:
        raise ValueError("channels must be positive.")
    dimension = layout.num_coefficients
    groups: list[WaveletMaskGroup] = []
    for mask in masks:
        inputs = np.asarray(
            [
                channel * dimension + site
                for site in mask.flat_indices
                for channel in range(channels)
            ],
            dtype=np.int64,
        )
        outputs = np.asarray(
            [channel * dimension + mask.target.flat_index for channel in range(channels)],
            dtype=np.int64,
        )
        orientation = mask.target.metadata.orientation
        groups.append(
            WaveletMaskGroup(
                input_indices=inputs,
                output_indices=outputs,
                family=mask.name,
                coefficient_type=mask.target.metadata.coefficient_type,
                pywt_level=mask.target.metadata.pywt_level,
                center=(mask.target.k1, mask.target.k2),
                orientations=() if orientation is None else (orientation,),
            )
        )
    return tuple(groups)


def build_rgb_radius_group_families(
    layout: WaveletLayout,
    radii_by_subband: Mapping[str, int],
    channels: int = 3,
) -> dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]]:
    """Build complete RGB groups while sharing every spatial mask across channels."""
    masks_by_family = build_layout_radius_masks(layout, radii_by_subband)
    return {
        family: _expand_masks_rgb(masks_by_family[family], layout, channels)
        for family in RADIUS_FAMILIES
    }


def build_rgb_subband_radius_groups(
    layout: WaveletLayout,
    subband: str,
    radius: int,
    family: RadiusMaskFamily = "full",
    channels: int = 3,
) -> tuple[WaveletMaskGroup, ...]:
    """Build RGB groups for every output position of one selected subband."""
    maxima = {
        target.subband: max_nonaliasing_radius(target)
        for target in build_central_targets(layout)
    }
    if subband not in maxima:
        raise ValueError(f"Unknown subband: {subband}.")
    if not 0 <= radius <= maxima[subband]:
        raise ValueError(
            f"radius={radius} lies outside [0,{maxima[subband]}] for {subband}."
        )
    radii = {name: (radius if name == subband else 0) for name in maxima}
    masks = tuple(
        mask
        for mask in build_layout_radius_masks(layout, radii)[family]
        if mask.target.subband == subband
    )
    return _expand_masks_rgb(masks, layout, channels)
