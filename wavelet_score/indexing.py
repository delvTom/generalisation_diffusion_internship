"""Explicit indexing for flattened, critically sampled 2D wavelet coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CoefficientType = Literal["scaling", "detail"]
Orientation = Literal["H", "V", "D"]


@dataclass(frozen=True, slots=True)
class SubbandDescriptor:
    """One contiguous subband in a flattened coefficient vector."""

    subband_index: int
    coefficient_type: CoefficientType
    pywt_level: int
    math_scale_j: int
    orientation: Orientation | None
    shape: tuple[int, int]
    flat_start: int
    flat_stop: int

    def __post_init__(self) -> None:
        height, width = self.shape
        if self.subband_index < 0 or self.pywt_level < 1 or self.math_scale_j < 0:
            raise ValueError("Subband indices and scales are invalid.")
        if height <= 0 or width <= 0:
            raise ValueError(f"Subband shape must be positive, got {self.shape}.")
        if self.coefficient_type == "scaling" and self.orientation is not None:
            raise ValueError("A scaling subband cannot have an orientation.")
        if self.coefficient_type == "detail" and self.orientation not in ("H", "V", "D"):
            raise ValueError("A detail subband requires orientation H, V, or D.")
        if self.flat_start < 0 or self.flat_stop <= self.flat_start:
            raise ValueError("A subband must occupy a non-empty positive interval.")
        if self.flat_stop - self.flat_start != height * width:
            raise ValueError("The flat interval size must equal the subband area.")

    @property
    def size(self) -> int:
        return self.flat_stop - self.flat_start

    @property
    def flat_slice(self) -> slice:
        return slice(self.flat_start, self.flat_stop)


@dataclass(frozen=True, slots=True)
class WaveletLayout:
    """Complete indexing and transform configuration of one DWT."""

    image_shape: tuple[int, int]
    wavelet_name: str
    filter_length: int
    mode: str
    decomposition_level: int
    max_safe_level: int
    subbands: tuple[SubbandDescriptor, ...]

    def __post_init__(self) -> None:
        height, width = self.image_shape
        if height <= 0 or width <= 0 or not self.wavelet_name or self.filter_length <= 0:
            raise ValueError("Invalid image or wavelet configuration.")
        if self.mode != "periodization":
            raise ValueError("Only periodization is supported.")
        if not 1 <= self.decomposition_level <= self.max_safe_level:
            raise ValueError("decomposition_level lies outside the safe range.")
        if not self.subbands:
            raise ValueError("A layout requires subbands.")
        if self.subbands[0].coefficient_type != "scaling":
            raise ValueError("The first subband must be the scaling approximation.")

        expected_start = 0
        for expected_index, descriptor in enumerate(self.subbands):
            if descriptor.subband_index != expected_index:
                raise ValueError("Subband indices must be contiguous.")
            if descriptor.flat_start != expected_start:
                raise ValueError("Subband intervals must be contiguous.")
            expected_start = descriptor.flat_stop
        if expected_start != height * width:
            raise ValueError("A critical DWT layout must contain exactly H*W coefficients.")

    @property
    def num_coefficients(self) -> int:
        return self.image_shape[0] * self.image_shape[1]


@dataclass(frozen=True, slots=True)
class WaveletCoefficientMetadata:
    """Identity of one wavelet coefficient in the flattened vector."""

    flat_index: int
    subband_index: int
    local_flat_index: int
    coefficient_type: CoefficientType
    pywt_level: int
    math_scale_j: int
    orientation: Orientation | None
    k1: int
    k2: int
    subband_shape: tuple[int, int]

    def __post_init__(self) -> None:
        height, width = self.subband_shape
        if min(self.flat_index, self.subband_index, self.local_flat_index) < 0:
            raise ValueError("Coefficient indices must be non-negative.")
        if not 0 <= self.k1 < height or not 0 <= self.k2 < width:
            raise ValueError("Coefficient position lies outside its subband.")
        if self.local_flat_index != self.k1 * width + self.k2:
            raise ValueError("local_flat_index must follow row-major ordering.")


def _power_of_two_exponent(value: int, name: str) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}.")
    return value.bit_length() - 1


def build_wavelet_layout(
    image_shape: tuple[int, int],
    wavelet_name: str,
    filter_length: int,
    mode: str,
    decomposition_level: int,
    max_safe_level: int,
) -> WaveletLayout:
    """Build the deterministic layout of a periodized critical 2D DWT."""
    height, width = image_shape
    exponent_h = _power_of_two_exponent(height, "image height")
    exponent_w = _power_of_two_exponent(width, "image width")
    if height != width or exponent_h != exponent_w:
        raise ValueError(f"Square power-of-two images are required, got {image_shape}.")
    if height % (2**decomposition_level) != 0:
        raise ValueError("H and W must be divisible by 2**decomposition_level.")
    if not 1 <= decomposition_level <= max_safe_level:
        raise ValueError("decomposition_level exceeds the safe range.")

    subbands: list[SubbandDescriptor] = []
    flat_start = 0

    def append(
        coefficient_type: CoefficientType,
        pywt_level: int,
        orientation: Orientation | None,
    ) -> None:
        nonlocal flat_start
        shape = (height // 2**pywt_level, width // 2**pywt_level)
        size = shape[0] * shape[1]
        subbands.append(
            SubbandDescriptor(
                subband_index=len(subbands),
                coefficient_type=coefficient_type,
                pywt_level=pywt_level,
                math_scale_j=exponent_h - pywt_level,
                orientation=orientation,
                shape=shape,
                flat_start=flat_start,
                flat_stop=flat_start + size,
            )
        )
        flat_start += size

    append("scaling", decomposition_level, None)
    for pywt_level in range(decomposition_level, 0, -1):
        for orientation in ("H", "V", "D"):
            append("detail", pywt_level, orientation)

    return WaveletLayout(
        image_shape=image_shape,
        wavelet_name=wavelet_name,
        filter_length=filter_length,
        mode=mode,
        decomposition_level=decomposition_level,
        max_safe_level=max_safe_level,
        subbands=tuple(subbands),
    )


def build_metadata(layout: WaveletLayout) -> tuple[WaveletCoefficientMetadata, ...]:
    """Build the bijection from flat indices to wavelet atom identities."""
    metadata: list[WaveletCoefficientMetadata] = []
    for descriptor in layout.subbands:
        height, width = descriptor.shape
        for k1 in range(height):
            for k2 in range(width):
                local_index = k1 * width + k2
                metadata.append(
                    WaveletCoefficientMetadata(
                        flat_index=descriptor.flat_start + local_index,
                        subband_index=descriptor.subband_index,
                        local_flat_index=local_index,
                        coefficient_type=descriptor.coefficient_type,
                        pywt_level=descriptor.pywt_level,
                        math_scale_j=descriptor.math_scale_j,
                        orientation=descriptor.orientation,
                        k1=k1,
                        k2=k2,
                        subband_shape=descriptor.shape,
                    )
                )
    if [item.flat_index for item in metadata] != list(range(layout.num_coefficients)):
        raise RuntimeError("Metadata do not form a bijection with flat indices.")
    return tuple(metadata)
