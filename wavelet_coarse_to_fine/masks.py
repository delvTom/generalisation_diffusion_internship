"""Hand-designed moving-window masks in wavelet scale space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wavelet_score.indexing import (
    SubbandDescriptor,
    WaveletCoefficientMetadata,
    WaveletLayout,
    build_metadata,
)
from wavelet_score.mask_groups import WaveletMaskGroup

ScaleRegion = Literal["coarse", "medium", "fine"]
TimeRegime = Literal["high_noise", "intermediate", "low_noise"]


@dataclass(frozen=True, slots=True)
class CoarseToFineSchedule:
    """Three piecewise-constant regimes for a noise-to-data FM path."""

    high_noise_end: float = 1.0 / 3.0
    low_noise_start: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.high_noise_end < self.low_noise_start < 1.0:
            raise ValueError(
                "Schedule boundaries must satisfy "
                "0 < high_noise_end < low_noise_start < 1."
            )

    def regime_at(self, time: float) -> TimeRegime:
        if not 0.0 <= time <= 1.0:
            raise ValueError("Flow time must lie in [0,1].")
        if time < self.high_noise_end:
            return "high_noise"
        if time < self.low_noise_start:
            return "intermediate"
        return "low_noise"


def descriptor_name(descriptor: SubbandDescriptor) -> str:
    if descriptor.coefficient_type == "scaling":
        return f"cA{descriptor.pywt_level}"
    return f"c{descriptor.orientation}{descriptor.pywt_level}"


def scale_region(descriptor: SubbandDescriptor, layout: WaveletLayout) -> ScaleRegion:
    """Map the level-3 DWT into C=(A3,D3), M=D2, and F=D1."""
    if layout.decomposition_level != 3:
        raise ValueError("The baseline requires a level-3 wavelet decomposition.")
    if descriptor.coefficient_type == "scaling" or descriptor.pywt_level == 3:
        return "coarse"
    if descriptor.pywt_level == 2:
        return "medium"
    if descriptor.pywt_level == 1:
        return "fine"
    raise ValueError(f"Unsupported wavelet descriptor: {descriptor}.")


def _descriptor_lookups(
    layout: WaveletLayout,
) -> tuple[dict[str, SubbandDescriptor], dict[int, SubbandDescriptor]]:
    by_name = {descriptor_name(item): item for item in layout.subbands}
    expected = {
        "cA3",
        "cH3",
        "cV3",
        "cD3",
        "cH2",
        "cV2",
        "cD2",
        "cH1",
        "cV1",
        "cD1",
    }
    if set(by_name) != expected:
        raise ValueError(
            "The baseline expects cA3 and H/V/D details at levels 3, 2, and 1."
        )
    return by_name, {
        item.subband_index: item for item in layout.subbands
    }


def _aligned_center(
    target: WaveletCoefficientMetadata,
    source_shape: tuple[int, int],
) -> tuple[int, int]:
    """Map a target site to the dyadically aligned site of a coarser band."""
    target_height, target_width = target.subband_shape
    source_height, source_width = source_shape
    if source_height > target_height or source_width > target_width:
        raise ValueError("This baseline only aligns a target to its own or coarser scale.")
    return (
        target.k1 * source_height // target_height,
        target.k2 * source_width // target_width,
    )


def _periodic_patch(
    descriptors: tuple[SubbandDescriptor, ...],
    center: tuple[int, int],
    radius: int,
) -> tuple[int, ...]:
    """Return all orientation/site indices in a periodic Chebyshev patch."""
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    shape = descriptors[0].shape
    if any(item.shape != shape for item in descriptors):
        raise ValueError("A patch cannot mix subband shapes.")
    height, width = shape
    if 2 * radius + 1 > min(height, width):
        raise ValueError("The periodic patch would alias onto duplicate sites.")

    indices: list[int] = []
    for delta1 in range(-radius, radius + 1):
        for delta2 in range(-radius, radius + 1):
            k1 = (center[0] + delta1) % height
            k2 = (center[1] + delta2) % width
            for descriptor in descriptors:
                indices.append(descriptor.flat_start + k1 * width + k2)
    if len(indices) != len(set(indices)):
        raise RuntimeError("A context patch unexpectedly contains duplicate sites.")
    return tuple(indices)


def _coarse_context(
    target: WaveletCoefficientMetadata,
    by_name: dict[str, SubbandDescriptor],
    radius: int,
) -> tuple[int, ...]:
    detail = tuple(by_name[f"c{orientation}3"] for orientation in ("H", "V", "D"))
    center = _aligned_center(target, detail[0].shape)
    values = list(_periodic_patch(detail, center, radius))
    scaling = by_name["cA3"]
    scaling_center = _aligned_center(target, scaling.shape)
    values.append(
        scaling.flat_start + scaling_center[0] * scaling.shape[1] + scaling_center[1]
    )
    return tuple(values)


def _detail_context(
    target: WaveletCoefficientMetadata,
    by_name: dict[str, SubbandDescriptor],
    level: int,
    radius: int,
) -> tuple[int, ...]:
    descriptors = tuple(
        by_name[f"c{orientation}{level}"] for orientation in ("H", "V", "D")
    )
    return _periodic_patch(
        descriptors,
        _aligned_center(target, descriptors[0].shape),
        radius,
    )


def _visible_regions(target_region: ScaleRegion, regime: TimeRegime) -> tuple[ScaleRegion, ...]:
    if target_region == "coarse":
        return ("coarse",)
    if target_region == "medium":
        if regime == "high_noise":
            return ("coarse",)
        if regime == "intermediate":
            return ("coarse", "medium")
        return ("medium",)
    if regime == "high_noise":
        return ("coarse",)
    if regime == "intermediate":
        return ("coarse", "medium")
    return ("medium", "fine")


def _scalar_visible_indices(
    target: WaveletCoefficientMetadata,
    target_descriptor: SubbandDescriptor,
    layout: WaveletLayout,
    regime: TimeRegime,
    radius: int,
    by_name: dict[str, SubbandDescriptor],
) -> tuple[int, ...]:
    values = [target.flat_index]
    seen = {target.flat_index}
    for region in _visible_regions(scale_region(target_descriptor, layout), regime):
        if region == "coarse":
            context = _coarse_context(target, by_name, radius)
        elif region == "medium":
            context = _detail_context(target, by_name, 2, radius)
        else:
            context = _detail_context(target, by_name, 1, radius)
        for index in context:
            if index not in seen:
                values.append(index)
                seen.add(index)
    return tuple(values)


def build_coarse_to_fine_rgb_groups(
    layout: WaveletLayout,
    regime: TimeRegime,
    *,
    channels: int = 3,
    radius: int = 1,
) -> tuple[WaveletMaskGroup, ...]:
    """Build every RGB output group for one coarse-to-fine time regime.

    A selected scalar wavelet site exposes all channels. Every group predicts
    the RGB vector at exactly one target site, and that target is always among
    its inputs.
    """
    if regime not in ("high_noise", "intermediate", "low_noise"):
        raise ValueError(f"Unknown time regime: {regime}.")
    if channels < 1:
        raise ValueError("channels must be positive.")
    if radius != 1:
        raise ValueError("The interpretable baseline fixes a 3x3 context (radius=1).")

    by_name, descriptor_by_index = _descriptor_lookups(layout)
    dimension = layout.num_coefficients
    groups: list[WaveletMaskGroup] = []
    for target in build_metadata(layout):
        descriptor = descriptor_by_index[target.subband_index]
        scalar_inputs = _scalar_visible_indices(
            target, descriptor, layout, regime, radius, by_name
        )
        input_indices = np.asarray(
            [
                channel * dimension + index
                for index in scalar_inputs
                for channel in range(channels)
            ],
            dtype=np.int64,
        )
        output_indices = np.asarray(
            [channel * dimension + target.flat_index for channel in range(channels)],
            dtype=np.int64,
        )
        groups.append(
            WaveletMaskGroup(
                input_indices=input_indices,
                output_indices=output_indices,
                family=f"coarse_to_fine_{regime}",
                coefficient_type=target.coefficient_type,
                pywt_level=target.pywt_level,
                center=(target.k1, target.k2),
                orientations=() if target.orientation is None else (target.orientation,),
            )
        )

    outputs = np.concatenate([group.output_indices for group in groups])
    if not np.array_equal(np.sort(outputs), np.arange(channels * dimension)):
        raise RuntimeError("Coarse-to-fine groups must cover every RGB output once.")
    return tuple(groups)


def build_rgb_region_indices(
    layout: WaveletLayout,
    *,
    channels: int = 3,
) -> dict[ScaleRegion, np.ndarray]:
    """Return channel-major flat indices for C, M, and F diagnostics."""
    if channels < 1:
        raise ValueError("channels must be positive.")
    dimension = layout.num_coefficients
    result: dict[ScaleRegion, list[int]] = {
        "coarse": [],
        "medium": [],
        "fine": [],
    }
    for channel in range(channels):
        offset = channel * dimension
        for descriptor in layout.subbands:
            region = scale_region(descriptor, layout)
            result[region].extend(
                range(offset + descriptor.flat_start, offset + descriptor.flat_stop)
            )
    arrays = {name: np.asarray(values, dtype=np.int64) for name, values in result.items()}
    merged = np.concatenate(tuple(arrays.values()))
    if not np.array_equal(np.sort(merged), np.arange(channels * dimension)):
        raise RuntimeError("C/M/F regions must partition the complete RGB layout.")
    return arrays
