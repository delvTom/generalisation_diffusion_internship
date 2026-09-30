"""Central wavelet targets and deterministic receptive masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from wavelet_score.indexing import WaveletCoefficientMetadata, WaveletLayout, build_metadata

MaskName = Literal["self", "orientation", "local", "scale", "all"]
RadiusMaskFamily = Literal[
    "same_band", "orientation_augmented", "scale_augmented", "full"
]
ReceptiveMaskName = MaskName | RadiusMaskFamily


@dataclass(frozen=True, slots=True)
class TargetCoefficient:
    subband: str
    flat_index: int
    k1: int
    k2: int
    metadata: WaveletCoefficientMetadata


@dataclass(frozen=True, slots=True)
class ReceptiveMask:
    name: ReceptiveMaskName
    target: TargetCoefficient
    coordinates: tuple[WaveletCoefficientMetadata, ...]

    def __post_init__(self) -> None:
        indices = self.flat_indices
        if not indices or indices[0] != self.target.flat_index:
            raise ValueError("The target must be the first visible coordinate.")
        if len(indices) != len(set(indices)):
            raise ValueError("A receptive mask cannot contain duplicate coordinates.")

    @property
    def flat_indices(self) -> tuple[int, ...]:
        return tuple(item.flat_index for item in self.coordinates)

    @property
    def descriptions(self) -> tuple[str, ...]:
        return tuple(describe_coordinate(item) for item in self.coordinates)


def subband_name(item: WaveletCoefficientMetadata) -> str:
    if item.coefficient_type == "scaling":
        return f"cA{item.pywt_level}"
    return f"c{item.orientation}{item.pywt_level}"


def describe_coordinate(item: WaveletCoefficientMetadata) -> str:
    return (
        f"{subband_name(item)}(k1={item.k1},k2={item.k2},"
        f"math_scale_j={item.math_scale_j},flat={item.flat_index})"
    )


def _metadata_lookups(layout: WaveletLayout):
    metadata = build_metadata(layout)
    by_identity = {
        (item.coefficient_type, item.pywt_level, item.orientation, item.k1, item.k2): item
        for item in metadata
    }
    by_subband: dict[str, list[WaveletCoefficientMetadata]] = {}
    for item in metadata:
        by_subband.setdefault(subband_name(item), []).append(item)
    return metadata, by_identity, by_subband


def build_central_targets(layout: WaveletLayout) -> tuple[TargetCoefficient, ...]:
    """Return cA_L then H/V/D details ordered from coarse to fine."""
    _, _, by_subband = _metadata_lookups(layout)
    names = _ordered_subband_names(layout)
    targets: list[TargetCoefficient] = []
    for name in names:
        items = by_subband[name]
        height, width = items[0].subband_shape
        k1, k2 = height // 2, width // 2
        item = next(entry for entry in items if (entry.k1, entry.k2) == (k1, k2))
        targets.append(_target_from_metadata(name, item))
    return tuple(targets)


def _ordered_subband_names(layout: WaveletLayout) -> list[str]:
    names = [f"cA{layout.decomposition_level}"]
    for level in range(layout.decomposition_level, 0, -1):
        names.extend(f"c{orientation}{level}" for orientation in ("H", "V", "D"))
    return names


def _target_from_metadata(
    name: str, item: WaveletCoefficientMetadata
) -> TargetCoefficient:
    return TargetCoefficient(name, item.flat_index, item.k1, item.k2, item)


def build_subband_target_groups(
    layout: WaveletLayout,
    exhaustive_max_size: int = 64,
    sampled_grid_size: int = 4,
) -> dict[str, tuple[TargetCoefficient, ...]]:
    """Select exhaustive coarse targets and a spatial grid of fine targets.

    Every subband containing at most ``exhaustive_max_size`` coefficients is
    evaluated exhaustively. Larger bands use cell-centered positions from a
    regular ``sampled_grid_size`` by ``sampled_grid_size`` grid.
    """
    if exhaustive_max_size < 1 or sampled_grid_size < 1:
        raise ValueError("Target-selection sizes must be positive.")
    _, _, by_subband = _metadata_lookups(layout)
    groups: dict[str, tuple[TargetCoefficient, ...]] = {}
    for name in _ordered_subband_names(layout):
        items = by_subband[name]
        height, width = items[0].subband_shape
        if len(items) <= exhaustive_max_size:
            selected = items
        else:
            if sampled_grid_size > min(height, width):
                raise ValueError(
                    "sampled_grid_size cannot exceed a sampled subband dimension."
                )
            rows = tuple(
                ((2 * index + 1) * height) // (2 * sampled_grid_size)
                for index in range(sampled_grid_size)
            )
            columns = tuple(
                ((2 * index + 1) * width) // (2 * sampled_grid_size)
                for index in range(sampled_grid_size)
            )
            by_position = {(item.k1, item.k2): item for item in items}
            selected = [by_position[(k1, k2)] for k1 in rows for k2 in columns]
        groups[name] = tuple(_target_from_metadata(name, item) for item in selected)
    return groups


def max_nonaliasing_radius(target: TargetCoefficient) -> int:
    """Largest periodic Chebyshev radius containing no duplicate positions."""
    height, width = target.metadata.subband_shape
    return (min(height, width) - 1) // 2


def _same_band_patch(
    target: TargetCoefficient,
    radius: int,
    by_identity: dict[tuple, WaveletCoefficientMetadata],
) -> tuple[WaveletCoefficientMetadata, ...]:
    if not isinstance(radius, int) or isinstance(radius, bool):
        raise TypeError("radius must be an integer.")
    maximum = max_nonaliasing_radius(target)
    if not 0 <= radius <= maximum:
        raise ValueError(
            f"radius={radius} is invalid for {target.metadata.subband_shape}; "
            f"the non-aliasing maximum is {maximum}."
        )

    item = target.metadata
    height, width = item.subband_shape
    coordinates = [item]
    for shell in range(1, radius + 1):
        for delta1 in range(-shell, shell + 1):
            for delta2 in range(-shell, shell + 1):
                if max(abs(delta1), abs(delta2)) != shell:
                    continue
                key = (
                    item.coefficient_type,
                    item.pywt_level,
                    item.orientation,
                    (item.k1 + delta1) % height,
                    (item.k2 + delta2) % width,
                )
                coordinates.append(by_identity[key])
    indices = [coordinate.flat_index for coordinate in coordinates]
    if len(indices) != (2 * radius + 1) ** 2 or len(indices) != len(set(indices)):
        raise RuntimeError("The periodic radius unexpectedly produced duplicate positions.")
    return tuple(coordinates)


def _orientation_group(
    target: TargetCoefficient,
    by_identity: dict[tuple, WaveletCoefficientMetadata],
) -> tuple[WaveletCoefficientMetadata, ...]:
    item = target.metadata
    if item.coefficient_type == "scaling":
        return ()
    return tuple(
        by_identity[("detail", item.pywt_level, orientation, item.k1, item.k2)]
        for orientation in ("H", "V", "D")
        if orientation != item.orientation
    )


def _local_group(
    target: TargetCoefficient,
    by_identity: dict[tuple, WaveletCoefficientMetadata],
) -> tuple[WaveletCoefficientMetadata, ...]:
    item = target.metadata
    height, width = item.subband_shape
    values: list[WaveletCoefficientMetadata] = []
    for delta1 in (-1, 0, 1):
        for delta2 in (-1, 0, 1):
            if (delta1, delta2) == (0, 0):
                continue
            key = (
                item.coefficient_type,
                item.pywt_level,
                item.orientation,
                (item.k1 + delta1) % height,
                (item.k2 + delta2) % width,
            )
            candidate = by_identity[key]
            if candidate.flat_index not in {value.flat_index for value in values}:
                values.append(candidate)
    return tuple(values)


def _scale_group(
    target: TargetCoefficient,
    layout: WaveletLayout,
    by_identity: dict[tuple, WaveletCoefficientMetadata],
) -> tuple[WaveletCoefficientMetadata, ...]:
    item = target.metadata
    if item.coefficient_type == "scaling":
        return tuple(
            by_identity[("detail", layout.decomposition_level, orientation, item.k1, item.k2)]
            for orientation in ("H", "V", "D")
        )

    values: list[WaveletCoefficientMetadata] = []
    if item.pywt_level == layout.decomposition_level:
        values.append(
            by_identity[("scaling", layout.decomposition_level, None, item.k1, item.k2)]
        )
    else:
        values.append(
            by_identity[
                (
                    "detail",
                    item.pywt_level + 1,
                    item.orientation,
                    item.k1 // 2,
                    item.k2 // 2,
                )
            ]
        )
    if item.pywt_level > 1:
        child_level = item.pywt_level - 1
        for delta1 in (0, 1):
            for delta2 in (0, 1):
                values.append(
                    by_identity[
                        (
                            "detail",
                            child_level,
                            item.orientation,
                            2 * item.k1 + delta1,
                            2 * item.k2 + delta2,
                        )
                    ]
                )
    return tuple(values)


def _build_radius_masks_from_lookup(
    layout: WaveletLayout,
    target: TargetCoefficient,
    radius: int,
    by_identity: dict[tuple, WaveletCoefficientMetadata],
) -> dict[RadiusMaskFamily, ReceptiveMask]:
    same_coordinates = _same_band_patch(target, radius, by_identity)

    orientation_additions: list[WaveletCoefficientMetadata] = []
    scale_additions: list[WaveletCoefficientMetadata] = []
    for coordinate in same_coordinates:
        patch_target = _target_from_metadata(target.subband, coordinate)
        orientation_additions.extend(_orientation_group(patch_target, by_identity))
        scale_additions.extend(_scale_group(patch_target, layout, by_identity))

    def make(
        name: RadiusMaskFamily,
        additions: tuple[WaveletCoefficientMetadata, ...],
    ) -> ReceptiveMask:
        coordinates = [target.metadata]
        seen = {target.flat_index}
        for coordinate in additions:
            if coordinate.flat_index not in seen:
                coordinates.append(coordinate)
                seen.add(coordinate.flat_index)
        return ReceptiveMask(name, target, tuple(coordinates))

    same = make("same_band", same_coordinates)
    orientation = make(
        "orientation_augmented",
        same_coordinates + tuple(orientation_additions),
    )
    scale = make(
        "scale_augmented",
        same_coordinates + tuple(scale_additions),
    )
    full = make("full", orientation.coordinates + scale.coordinates)
    return {
        "same_band": same,
        "orientation_augmented": orientation,
        "scale_augmented": scale,
        "full": full,
    }


def build_radius_masks(
    layout: WaveletLayout,
    target: TargetCoefficient,
    radius: int,
) -> dict[RadiusMaskFamily, ReceptiveMask]:
    """Build nested spatial-radius masks using the shared wavelet topology."""
    _, by_identity, _ = _metadata_lookups(layout)
    return _build_radius_masks_from_lookup(layout, target, radius, by_identity)


def build_layout_radius_masks(
    layout: WaveletLayout,
    radii_by_subband: Mapping[str, int],
) -> dict[RadiusMaskFamily, tuple[ReceptiveMask, ...]]:
    """Build one radius mask per output coefficient in flat-index order."""
    metadata, by_identity, by_subband = _metadata_lookups(layout)
    expected_subbands = set(by_subband)
    provided_subbands = set(radii_by_subband)
    if provided_subbands != expected_subbands:
        missing = sorted(expected_subbands - provided_subbands)
        extra = sorted(provided_subbands - expected_subbands)
        raise ValueError(
            "radii_by_subband must cover the layout exactly; "
            f"missing={missing}, extra={extra}."
        )

    masks_by_family: dict[RadiusMaskFamily, list[ReceptiveMask]] = {
        family: []
        for family in (
            "same_band",
            "orientation_augmented",
            "scale_augmented",
            "full",
        )
    }
    for item in metadata:
        name = subband_name(item)
        target = _target_from_metadata(name, item)
        masks = _build_radius_masks_from_lookup(
            layout,
            target,
            radii_by_subband[name],
            by_identity,
        )
        for family, mask in masks.items():
            masks_by_family[family].append(mask)

    result = {
        family: tuple(family_masks)
        for family, family_masks in masks_by_family.items()
    }
    expected_indices = tuple(range(layout.num_coefficients))
    for family_masks in result.values():
        if tuple(mask.target.flat_index for mask in family_masks) != expected_indices:
            raise RuntimeError("Layout radius masks do not cover every output exactly once.")
    return result


def build_target_masks(
    layout: WaveletLayout, target: TargetCoefficient
) -> dict[MaskName, ReceptiveMask]:
    """Build cumulative masks while preserving target-first deterministic order."""
    _, by_identity, _ = _metadata_lookups(layout)
    groups = {
        "orientation": _orientation_group(target, by_identity),
        "local": _local_group(target, by_identity),
        "scale": _scale_group(target, layout, by_identity),
    }

    def make(name: MaskName, additions: tuple[WaveletCoefficientMetadata, ...]) -> ReceptiveMask:
        coordinates = [target.metadata]
        seen = {target.flat_index}
        for item in additions:
            if item.flat_index not in seen:
                coordinates.append(item)
                seen.add(item.flat_index)
        return ReceptiveMask(name, target, tuple(coordinates))

    masks: dict[MaskName, ReceptiveMask] = {
        "self": make("self", ()),
        "orientation": make("orientation", groups["orientation"]),
        "local": make("local", groups["local"]),
        "scale": make("scale", groups["scale"]),
    }
    masks["all"] = make(
        "all", groups["orientation"] + groups["local"] + groups["scale"]
    )
    return masks
