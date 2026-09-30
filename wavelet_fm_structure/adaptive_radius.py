"""Calibrated radius schedules and full-layout wavelet mask groups."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from wavelet_fm_structure.masks import (
    RadiusMaskFamily,
    build_central_targets,
    build_layout_radius_masks,
    max_nonaliasing_radius,
)
from wavelet_score.indexing import WaveletLayout
from wavelet_score.mask_groups import WaveletMaskGroup

RADIUS_FAMILIES: tuple[RadiusMaskFamily, ...] = (
    "same_band",
    "orientation_augmented",
    "scale_augmented",
    "full",
)


def select_smallest_radius_within_best(
    risks_by_radius: Mapping[int, float],
    relative_tolerance: float = 0.01,
) -> int:
    """Select the smallest radius within a relative tolerance of the best risk."""
    if not risks_by_radius:
        raise ValueError("At least one radius risk is required.")
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative.")
    radii = sorted(risks_by_radius)
    if radii[0] < 0 or any(not np.isfinite(risks_by_radius[r]) for r in radii):
        raise ValueError("Radii must be non-negative and risks must be finite.")
    best = min(float(risks_by_radius[radius]) for radius in radii)
    if best < 0.0:
        raise ValueError("Risks must be non-negative.")
    threshold = (1.0 + relative_tolerance) * best
    roundoff = np.finfo(np.float64).eps * max(1.0, threshold) * 32.0
    return min(
        radius
        for radius in radii
        if float(risks_by_radius[radius]) <= threshold + roundoff
    )


@dataclass(frozen=True, slots=True)
class FullRadiusSchedule:
    """Piecewise-nearest schedule calibrated from full-mask clean risks."""

    calibration_times: tuple[float, ...]
    subbands: tuple[str, ...]
    radii: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.calibration_times
            or tuple(sorted(self.calibration_times)) != self.calibration_times
            or len(set(self.calibration_times)) != len(self.calibration_times)
        ):
            raise ValueError("Calibration times must be non-empty, unique, and sorted.")
        if not self.subbands or len(set(self.subbands)) != len(self.subbands):
            raise ValueError("Subband names must be non-empty and unique.")
        if len(self.radii) != len(self.calibration_times):
            raise ValueError("One radius row is required per calibration time.")
        if any(len(row) != len(self.subbands) for row in self.radii):
            raise ValueError("Each radius row must cover every subband.")
        if any(radius < 0 for row in self.radii for radius in row):
            raise ValueError("Calibrated radii must be non-negative.")

    @classmethod
    def from_metrics_csv(
        cls,
        metrics_path: Path,
        layout: WaveletLayout,
    ) -> FullRadiusSchedule:
        """Load the shared full-family selected radius for every subband and time."""
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Radius metrics are empty: {metrics_path}")

        subbands = tuple(target.subband for target in build_central_targets(layout))
        values: dict[tuple[str, float], set[int]] = {}
        for row in rows:
            if row.get("family") != "full":
                continue
            key = (str(row["subband"]), float(row["t"]))
            values.setdefault(key, set()).add(int(row["selected_radius"]))
        times = tuple(sorted({time for _, time in values}))
        if not times:
            raise ValueError("Radius metrics contain no full-family rows.")

        radius_rows: list[tuple[int, ...]] = []
        maximum_by_subband = {
            target.subband: max_nonaliasing_radius(target)
            for target in build_central_targets(layout)
        }
        for time in times:
            time_radii: list[int] = []
            for subband in subbands:
                candidates = values.get((subband, time))
                if candidates is None or len(candidates) != 1:
                    raise ValueError(
                        "Each (subband, time) must have one consistent selected radius; "
                        f"got {candidates} for ({subband}, {time})."
                    )
                radius = next(iter(candidates))
                if radius > maximum_by_subband[subband]:
                    raise ValueError(
                        f"Selected radius {radius} exceeds the maximum for {subband}."
                    )
                time_radii.append(radius)
            radius_rows.append(tuple(time_radii))
        return cls(times, subbands, tuple(radius_rows))

    def nearest_calibration_time(self, time: float) -> float:
        """Return the nearest measured time, choosing the earlier one on a tie."""
        if not 0.0 <= time < 1.0:
            raise ValueError("Flow time must lie in [0,1).")
        return min(self.calibration_times, key=lambda value: (abs(value - time), value))

    def radii_at(self, time: float) -> dict[str, int]:
        """Return the full-selected radii at the nearest calibrated time."""
        nearest = self.nearest_calibration_time(time)
        row = self.radii[self.calibration_times.index(nearest)]
        return dict(zip(self.subbands, row, strict=True))


def fixed_radii(layout: WaveletLayout, radius: int) -> dict[str, int]:
    """Clamp one requested radius to each subband's non-aliasing maximum."""
    if radius < 0:
        raise ValueError("A fixed radius must be non-negative.")
    return {
        target.subband: min(radius, max_nonaliasing_radius(target))
        for target in build_central_targets(layout)
    }


def build_radius_groups(
    layout: WaveletLayout,
    family: RadiusMaskFamily,
    radii_by_subband: Mapping[str, int],
) -> tuple[WaveletMaskGroup, ...]:
    """Convert tested receptive masks into groups for vector-valued prediction."""
    return build_radius_group_families(layout, radii_by_subband)[family]


def build_radius_group_families(
    layout: WaveletLayout,
    radii_by_subband: Mapping[str, int],
) -> dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]]:
    """Build all four group families while traversing the layout only once."""
    masks_by_family = build_layout_radius_masks(layout, radii_by_subband)
    result: dict[RadiusMaskFamily, tuple[WaveletMaskGroup, ...]] = {}
    for family in RADIUS_FAMILIES:
        groups = tuple(
            WaveletMaskGroup(
                input_indices=np.asarray(mask.flat_indices, dtype=np.int64),
                output_indices=np.asarray([mask.target.flat_index], dtype=np.int64),
                family=family,
                coefficient_type=mask.target.metadata.coefficient_type,
                pywt_level=mask.target.metadata.pywt_level,
                center=(mask.target.k1, mask.target.k2),
                orientations=(
                    ()
                    if mask.target.metadata.orientation is None
                    else (mask.target.metadata.orientation,)
                ),
            )
            for mask in masks_by_family[family]
        )
        outputs = np.concatenate([group.output_indices for group in groups])
        if outputs.tolist() != list(range(layout.num_coefficients)):
            raise RuntimeError("Radius groups must follow exact flat-index order.")
        result[family] = groups
    return result
