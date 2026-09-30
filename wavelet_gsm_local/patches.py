"""Balanced extraction of valid 3x3 coefficient patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATCH_OFFSETS = np.asarray(
    [(delta_1, delta_2) for delta_1 in (-1, 0, 1) for delta_2 in (-1, 0, 1)],
    dtype=np.int64,
)
CENTER_INDEX = 4


@dataclass(frozen=True, slots=True)
class PatchSelection:
    image_indices: np.ndarray  # [P]
    center_rows: np.ndarray  # [P]
    center_columns: np.ndarray  # [P]
    band_shape: tuple[int, int]

    def __post_init__(self) -> None:
        size = len(self.image_indices)
        if self.center_rows.shape != (size,) or self.center_columns.shape != (size,):
            raise ValueError("Patch-selection arrays must have identical lengths.")
        height, width = self.band_shape
        if size and (
            self.center_rows.min() < 1
            or self.center_rows.max() >= height - 1
            or self.center_columns.min() < 1
            or self.center_columns.max() >= width - 1
        ):
            raise ValueError("A selected patch would cross a subband boundary.")


def sample_valid_centers(
    num_images: int,
    band_shape: tuple[int, int],
    patches_per_image: int,
    *,
    seed: int,
) -> PatchSelection:
    """Choose the same number of distinct valid centers from every image."""
    height, width = band_shape
    valid_rows = height - 2
    valid_columns = width - 2
    num_valid = valid_rows * valid_columns
    if num_images < 1 or not 1 <= patches_per_image <= num_valid:
        raise ValueError("Invalid number of images or patches per image.")
    generator = np.random.default_rng(seed)
    local_indices = np.empty((num_images, patches_per_image), dtype=np.int64)
    for image_index in range(num_images):
        local_indices[image_index] = generator.choice(
            num_valid, size=patches_per_image, replace=False
        )
    return PatchSelection(
        image_indices=np.repeat(np.arange(num_images, dtype=np.int64), patches_per_image),
        center_rows=(local_indices // valid_columns + 1).reshape(-1),
        center_columns=(local_indices % valid_columns + 1).reshape(-1),
        band_shape=band_shape,
    )


def extract_patches(band: np.ndarray, selection: PatchSelection) -> np.ndarray:
    """Extract row-major 3x3 patches without wrapping or padding."""
    if band.ndim != 3 or band.shape[1:] != selection.band_shape:
        raise ValueError("band and selection shapes are incompatible.")
    rows = selection.center_rows[:, None] + PATCH_OFFSETS[None, :, 0]
    columns = selection.center_columns[:, None] + PATCH_OFFSETS[None, :, 1]
    patches = band[selection.image_indices[:, None], rows, columns]
    return np.ascontiguousarray(patches)


def per_image_mean(values: np.ndarray, selection: PatchSelection, num_images: int) -> np.ndarray:
    """Average one scalar patch statistic within each source image."""
    if values.shape != (len(selection.image_indices),):
        raise ValueError("values must contain one scalar per selected patch.")
    sums = np.bincount(selection.image_indices, weights=values, minlength=num_images)
    counts = np.bincount(selection.image_indices, minlength=num_images)
    if np.any(counts == 0):
        raise ValueError("Every image must contribute at least one patch.")
    return sums / counts
