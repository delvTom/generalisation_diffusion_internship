"""Immutable wavelet mask groups for empirical conditional FM estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from wavelet_score.indexing import WaveletLayout

MaskFamily = Literal[
    "independent", "band_tied", "local", "local_band_tied", "scale", "band_local_scale"
]


@dataclass(frozen=True, slots=True)
class WaveletMaskGroup:
    input_indices: np.ndarray
    output_indices: np.ndarray
    family: str
    coefficient_type: str
    pywt_level: int
    center: tuple[int, int]
    orientations: tuple[str, ...]

    def __post_init__(self) -> None:
        inputs = np.asarray(self.input_indices, dtype=np.int64)
        outputs = np.asarray(self.output_indices, dtype=np.int64)
        if inputs.ndim != 1 or outputs.ndim != 1 or not len(inputs) or not len(outputs):
            raise ValueError("Mask indices must be non-empty one-dimensional arrays.")
        if len(np.unique(inputs)) != len(inputs) or len(np.unique(outputs)) != len(outputs):
            raise ValueError("Mask indices cannot contain duplicates.")
        if not set(outputs).issubset(set(inputs)):
            raise ValueError("Every output coefficient must be observed by its group.")
        inputs.setflags(write=False)
        outputs.setflags(write=False)
        object.__setattr__(self, "input_indices", inputs)
        object.__setattr__(self, "output_indices", outputs)


def _periodic_positions(
    center: tuple[int, int], shape: tuple[int, int], radius: int
) -> tuple[tuple[int, int], ...]:
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    height, width = shape
    positions: list[tuple[int, int]] = []
    for delta_1 in range(-radius, radius + 1):
        for delta_2 in range(-radius, radius + 1):
            position = ((center[0] + delta_1) % height, (center[1] + delta_2) % width)
            if position not in positions:
                positions.append(position)
    return tuple(positions)


def build_mask_groups(
    layout: WaveletLayout,
    family: MaskFamily,
    radius: int = 1,
) -> tuple[WaveletMaskGroup, ...]:
    """Build groups solely from the centralized wavelet subband layout."""
    if family not in (
        "independent", "band_tied", "local", "local_band_tied", "scale", "band_local_scale"
    ):
        raise ValueError(f"Unknown mask family: {family}.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if family in ("scale", "band_local_scale"):
        return (
            build_scale_receptive_mask(layout)
            if family == "scale"
            else build_band_local_scale_receptive_mask(layout, radius)
        )
    groups: list[WaveletMaskGroup] = []

    def flat_index(descriptor, k1: int, k2: int) -> int:
        return descriptor.flat_start + k1 * descriptor.shape[1] + k2

    scaling = layout.subbands[0]
    detail_by_level: dict[int, tuple] = {}
    for descriptor in layout.subbands[1:]:
        detail_by_level.setdefault(descriptor.pywt_level, []).append(descriptor)
    detail_by_level = {
        level: tuple(sorted(items, key=lambda item: ("H", "V", "D").index(item.orientation)))
        for level, items in detail_by_level.items()
    }

    def append_single(descriptor, k1: int, k2: int, local: bool) -> None:
        positions = _periodic_positions((k1, k2), descriptor.shape, radius if local else 0)
        inputs = np.array([flat_index(descriptor, *position) for position in positions])
        output = np.array([flat_index(descriptor, k1, k2)])
        groups.append(
            WaveletMaskGroup(
                inputs,
                output,
                family,
                descriptor.coefficient_type,
                descriptor.pywt_level,
                (k1, k2),
                () if descriptor.orientation is None else (descriptor.orientation,),
            )
        )

    scaling_local = family in ("local", "local_band_tied")
    for k1 in range(scaling.shape[0]):
        for k2 in range(scaling.shape[1]):
            append_single(scaling, k1, k2, scaling_local)

    for level in sorted(detail_by_level, reverse=True):
        descriptors = detail_by_level[level]
        height, width = descriptors[0].shape
        for k1 in range(height):
            for k2 in range(width):
                if family in ("band_tied", "local_band_tied"):
                    positions = _periodic_positions(
                        (k1, k2), descriptors[0].shape,
                        radius if family == "local_band_tied" else 0,
                    )
                    inputs = np.array(
                        [
                            flat_index(descriptor, *position)
                            for position in positions
                            for descriptor in descriptors
                        ]
                    )
                    outputs = np.array([flat_index(descriptor, k1, k2) for descriptor in descriptors])
                    groups.append(
                        WaveletMaskGroup(
                            inputs,
                            outputs,
                            family,
                            "detail",
                            level,
                            (k1, k2),
                            ("H", "V", "D"),
                        )
                    )
                else:
                    for descriptor in descriptors:
                        append_single(descriptor, k1, k2, family == "local")

    outputs = np.concatenate([group.output_indices for group in groups])
    if sorted(outputs.tolist()) != list(range(layout.num_coefficients)):
        raise RuntimeError("Mask groups must cover every output exactly once.")
    return tuple(groups)


def _groups_to_visible_sets(
    layout: WaveletLayout, groups: tuple[WaveletMaskGroup, ...]
) -> list[set[int]]:
    visible = [{index} for index in range(layout.num_coefficients)]
    for group in groups:
        inputs = set(int(index) for index in group.input_indices)
        for output in group.output_indices:
            visible[int(output)].update(inputs)
    return visible


def _singleton_output_groups(
    layout: WaveletLayout, family: MaskFamily, visible: list[set[int]]
) -> tuple[WaveletMaskGroup, ...]:
    from wavelet_score.indexing import build_metadata

    metadata = build_metadata(layout)
    return tuple(
        WaveletMaskGroup(
            np.array(sorted(visible[index]), dtype=np.int64),
            np.array([index], dtype=np.int64),
            family,
            item.coefficient_type,
            item.pywt_level,
            (item.k1, item.k2),
            () if item.orientation is None else (item.orientation,),
        )
        for index, item in enumerate(metadata)
    )


def build_scale_receptive_mask(layout: WaveletLayout) -> tuple[WaveletMaskGroup, ...]:
    """Observe self, adjacent same-orientation scales, and scaling/detail links."""
    from wavelet_score.scale_relations import (
        build_layout_scale_interaction_terms,
        receptive_indices_from_scale_edges,
    )

    receptive = receptive_indices_from_scale_edges(
        layout.num_coefficients, build_layout_scale_interaction_terms(layout)
    )
    return _singleton_output_groups(layout, "scale", [set(indices) for indices in receptive])


def build_band_local_scale_receptive_mask(
    layout: WaveletLayout, radius: int = 1
) -> tuple[WaveletMaskGroup, ...]:
    """Union band, local, and adjacent-scale receptive coordinates per output."""
    band = _groups_to_visible_sets(layout, build_mask_groups(layout, "band_tied", radius))
    local = _groups_to_visible_sets(layout, build_mask_groups(layout, "local", radius))
    scale = _groups_to_visible_sets(layout, build_scale_receptive_mask(layout))
    visible = [band[index] | local[index] | scale[index] for index in range(layout.num_coefficients)]
    return _singleton_output_groups(layout, "band_local_scale", visible)
