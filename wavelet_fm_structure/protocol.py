"""Immutable protocol choices for the first controlled MNIST study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    image_shape: tuple[int, int] = (32, 32)
    wavelet_name: str = "db2"
    decomposition_level: int = 3
    mode: str = "periodization"
    times: tuple[float, ...] = (0.05, 0.12, 0.27, 0.50, 0.73, 0.88, 0.95)
    dtype: str = "float64"
    num_fit_queries: int = 1024
    num_validation_queries: int = 1024
    num_evaluation_queries: int = 4096
    kernel_width_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    ridge_values: tuple[float, ...] = (1e-5, 1e-3, 1e-1)
    local_radius: int = 1

    def __post_init__(self) -> None:
        if self.dtype != "float64":
            raise ValueError("The reference protocol requires float64.")
        if self.mode != "periodization" or self.local_radius != 1:
            raise ValueError("The first protocol fixes periodization and radius one.")
        if any(not 0.0 < time < 1.0 for time in self.times):
            raise ValueError("All evaluation times must lie strictly between zero and one.")
        if min(
            self.num_fit_queries,
            self.num_validation_queries,
            self.num_evaluation_queries,
        ) < 1:
            raise ValueError("Monte-Carlo sample counts must be positive.")


DEFAULT_PROTOCOL = StudyProtocol()
