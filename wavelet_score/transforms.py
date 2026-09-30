"""Audited critically sampled 2D wavelet analysis and synthesis."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pywt

from wavelet_score.indexing import WaveletLayout, build_wavelet_layout

DetailCoefficients: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
PyWaveletCoefficients: TypeAlias = list[np.ndarray | DetailCoefficients]


class WaveletTransform2D:
    """Apply a fixed wavelet family and depth to batches [N,H,W]."""

    def __init__(
        self,
        wavelet_name: str,
        decomposition_level: int,
        mode: str = "periodization",
    ) -> None:
        if decomposition_level < 1:
            raise ValueError("decomposition_level must be at least one.")
        if mode != "periodization":
            raise ValueError("Only periodization is supported.")
        try:
            self._wavelet = pywt.Wavelet(wavelet_name)
        except ValueError as error:
            raise ValueError(f"Unknown wavelet: {wavelet_name!r}.") from error
        self.wavelet_name = self._wavelet.name
        self.decomposition_level = decomposition_level
        self.mode = mode

    @staticmethod
    def _validate_float_array(array: np.ndarray, name: str, ndim: int) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray.")
        if array.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions, got {array.shape}.")
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{name} must use float32 or float64, got {array.dtype}.")
        if array.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample.")

    def _validate_image_shape(self, image_shape: tuple[int, int]) -> int:
        height, width = image_shape
        if height <= 0 or width <= 0 or height != width:
            raise ValueError(f"Positive square images are required, got {image_shape}.")
        divisor = 2**self.decomposition_level
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(f"H and W must be divisible by 2**L={divisor}.")
        max_safe_level = min(
            pywt.dwt_max_level(height, self._wavelet.dec_len),
            pywt.dwt_max_level(width, self._wavelet.dec_len),
        )
        if self.decomposition_level > max_safe_level:
            raise ValueError(
                f"level={self.decomposition_level} exceeds pywt.dwt_max_level="
                f"{max_safe_level} for {self.wavelet_name} on {height}x{width}."
            )
        return max_safe_level

    def analyze(self, images: np.ndarray) -> tuple[np.ndarray, WaveletLayout]:
        """Transform images [N,H,W] into flattened coefficients Y_t [N,H*W]."""
        self._validate_float_array(images, "images", ndim=3)
        image_shape = images.shape[1], images.shape[2]
        self._validate_image_shape(image_shape)
        coeffs = pywt.wavedec2(
            images,
            wavelet=self._wavelet,
            mode=self.mode,
            level=self.decomposition_level,
            axes=(-2, -1),
        )
        Y_t, layout = self.flatten_coefficients(coeffs)
        expected_shape = images.shape[0], images.shape[1] * images.shape[2]
        if layout.image_shape != image_shape or Y_t.shape != expected_shape:
            raise RuntimeError("Wavelet analysis returned an inconsistent layout or shape.")
        if Y_t.dtype != images.dtype:
            raise RuntimeError("Wavelet analysis unexpectedly changed dtype.")
        return Y_t, layout

    def flatten_coefficients(
        self,
        coeffs: PyWaveletCoefficients,
    ) -> tuple[np.ndarray, WaveletLayout]:
        """Flatten a batched PyWavelets tree into Y_t and an explicit layout."""
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError("coeffs must be returned by pywt.wavedec2.")
        if len(coeffs) != self.decomposition_level + 1:
            raise ValueError("The coefficient tree depth does not match the transform.")

        approximation = coeffs[0]
        self._validate_float_array(approximation, "cA_L", ndim=3)
        batch_size, dtype = approximation.shape[0], approximation.dtype
        ordered_arrays: list[np.ndarray] = [approximation]

        for tuple_index, details in enumerate(coeffs[1:], start=1):
            pywt_level = self.decomposition_level - tuple_index + 1
            if not isinstance(details, (tuple, list)) or len(details) != 3:
                raise ValueError(f"Level {pywt_level} must contain (cH,cV,cD).")
            for orientation, array in zip(("H", "V", "D"), details, strict=True):
                self._validate_float_array(array, f"c{orientation}{pywt_level}", ndim=3)
                if array.shape[0] != batch_size or array.dtype != dtype:
                    raise ValueError("All coefficient arrays must share batch and dtype.")
                ordered_arrays.append(array)

        finest_details = coeffs[-1]
        if not isinstance(finest_details, (tuple, list)):
            raise RuntimeError("Malformed finest detail tuple.")
        image_shape = finest_details[0].shape[1] * 2, finest_details[0].shape[2] * 2
        max_safe_level = self._validate_image_shape(image_shape)
        layout = build_wavelet_layout(
            image_shape=image_shape,
            wavelet_name=self.wavelet_name,
            filter_length=self._wavelet.dec_len,
            mode=self.mode,
            decomposition_level=self.decomposition_level,
            max_safe_level=max_safe_level,
        )
        if len(ordered_arrays) != len(layout.subbands):
            raise RuntimeError("Coefficient tree and layout have different band counts.")

        blocks: list[np.ndarray] = []
        for array, descriptor in zip(ordered_arrays, layout.subbands, strict=True):
            if array.shape[1:] != descriptor.shape:
                raise ValueError(
                    f"Subband {descriptor.subband_index} has shape {array.shape[1:]}; "
                    f"expected {descriptor.shape}."
                )
            blocks.append(array.reshape(batch_size, descriptor.size))
        Y_t = np.concatenate(blocks, axis=1)
        if Y_t.shape != (batch_size, layout.num_coefficients):
            raise RuntimeError("Unexpected flattened coefficient shape.")
        return Y_t, layout

    def unflatten_coefficients(
        self,
        Y_t: np.ndarray,
        layout: WaveletLayout,
    ) -> PyWaveletCoefficients:
        """Rebuild the batched PyWavelets tree from Y_t."""
        self._validate_float_array(Y_t, "Y_t", ndim=2)
        self._validate_layout(layout)
        if Y_t.shape[1] != layout.num_coefficients:
            raise ValueError(f"Y_t must have shape [N,{layout.num_coefficients}].")
        batch_size = Y_t.shape[0]
        arrays = [
            Y_t[:, band.flat_slice].reshape(batch_size, *band.shape)
            for band in layout.subbands
        ]
        coeffs: PyWaveletCoefficients = [arrays[0]]
        index = 1
        for pywt_level in range(self.decomposition_level, 0, -1):
            descriptors = layout.subbands[index : index + 3]
            if tuple(item.pywt_level for item in descriptors) != (pywt_level,) * 3:
                raise RuntimeError("Detail levels are not ordered correctly.")
            if tuple(item.orientation for item in descriptors) != ("H", "V", "D"):
                raise RuntimeError("Detail orientations are not ordered H, V, D.")
            coeffs.append(tuple(arrays[index : index + 3]))
            index += 3
        return coeffs

    def synthesize(self, Y_t: np.ndarray, layout: WaveletLayout) -> np.ndarray:
        """Reconstruct images [N,H,W] from Y_t [N,H*W]."""
        coeffs = self.unflatten_coefficients(Y_t, layout)
        images = pywt.waverec2(
            coeffs,
            wavelet=self._wavelet,
            mode=self.mode,
            axes=(-2, -1),
        )
        expected_shape = Y_t.shape[0], *layout.image_shape
        if images.shape != expected_shape or images.dtype != Y_t.dtype:
            raise RuntimeError("Wavelet synthesis changed shape or dtype.")
        return images

    def _validate_layout(self, layout: WaveletLayout) -> None:
        if not isinstance(layout, WaveletLayout):
            raise TypeError("layout must be a WaveletLayout.")
        expected = (
            self.wavelet_name,
            self._wavelet.dec_len,
            self.mode,
            self.decomposition_level,
        )
        observed = (
            layout.wavelet_name,
            layout.filter_length,
            layout.mode,
            layout.decomposition_level,
        )
        if observed != expected:
            raise ValueError(f"Layout {observed} is incompatible with transform {expected}.")
