"""Standard Flow Matching paths in pixel and critical-wavelet coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_score.indexing import WaveletLayout
from wavelet_score.transforms import WaveletTransform2D


@dataclass(frozen=True, slots=True)
class FlowMatchingBatch:
    """One paired FM path represented consistently in pixels and wavelets."""

    images_0: np.ndarray  # [N,H,W]
    noise: np.ndarray  # [N,H,W]
    images_t: np.ndarray  # [N,H,W]
    velocity_target_pixels: np.ndarray  # [N,H,W]
    Y_0: np.ndarray  # [N,d]
    xi: np.ndarray  # [N,d]
    Y_t: np.ndarray  # [N,d]
    velocity_target_wavelets: np.ndarray  # [N,d]
    t: float
    layout: WaveletLayout

    def __post_init__(self) -> None:
        pixel_arrays = (
            self.images_0,
            self.noise,
            self.images_t,
            self.velocity_target_pixels,
        )
        if any(array.shape != self.images_0.shape for array in pixel_arrays):
            raise ValueError("All pixel arrays must have the same [N,H,W] shape.")
        if self.images_0.ndim != 3:
            raise ValueError("Pixel arrays must have shape [N,H,W].")
        wavelet_arrays = (self.Y_0, self.xi, self.Y_t, self.velocity_target_wavelets)
        expected = (self.images_0.shape[0], self.layout.num_coefficients)
        if any(array.shape != expected for array in wavelet_arrays):
            raise ValueError(f"All wavelet arrays must have shape {expected}.")
        dtypes = {array.dtype for array in (*pixel_arrays, *wavelet_arrays)}
        if len(dtypes) != 1 or next(iter(dtypes)) not in (
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError("All FM arrays must share a float32 or float64 dtype.")
        if not 0.0 <= self.t <= 1.0 or not np.isfinite(self.t):
            raise ValueError("t must be finite and lie in [0,1].")


def build_flow_matching_batch(
    images_0: np.ndarray,
    t: float,
    transform: WaveletTransform2D,
    *,
    noise: np.ndarray | None = None,
    generator: np.random.Generator | None = None,
) -> FlowMatchingBatch:
    """Construct X_t, V_X, Y_t and V_Y for one scalar FM time."""
    if not isinstance(images_0, np.ndarray) or images_0.ndim != 3:
        raise ValueError("images_0 must be a NumPy array with shape [N,H,W].")
    if images_0.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("images_0 must use float32 or float64.")
    if not np.isscalar(t):
        raise TypeError("t must be a scalar.")
    t_float = float(t)
    if not np.isfinite(t_float) or not 0.0 <= t_float <= 1.0:
        raise ValueError("t must be finite and lie in [0,1].")
    if noise is not None and generator is not None:
        raise ValueError("Provide either noise or generator, not both.")
    if noise is None:
        if not isinstance(generator, np.random.Generator):
            raise TypeError("A NumPy generator is required when noise is not provided.")
        noise = generator.standard_normal(images_0.shape, dtype=images_0.dtype)
    elif not isinstance(noise, np.ndarray) or noise.shape != images_0.shape:
        raise ValueError("noise must be a NumPy array with the same shape as images_0.")
    elif noise.dtype != images_0.dtype:
        raise TypeError("noise and images_0 must have the same dtype.")

    images_t = (1.0 - t_float) * noise + t_float * images_0
    velocity_target_pixels = images_0 - noise
    Y_0, layout = transform.analyze(images_0)
    xi, noise_layout = transform.analyze(noise)
    Y_t, path_layout = transform.analyze(images_t)
    if noise_layout != layout or path_layout != layout:
        raise RuntimeError("FM arrays produced incompatible wavelet layouts.")
    velocity_target_wavelets = Y_0 - xi

    return FlowMatchingBatch(
        images_0=images_0,
        noise=noise,
        images_t=images_t,
        velocity_target_pixels=velocity_target_pixels,
        Y_0=Y_0,
        xi=xi,
        Y_t=Y_t,
        velocity_target_wavelets=velocity_target_wavelets,
        t=t_float,
        layout=layout,
    )
