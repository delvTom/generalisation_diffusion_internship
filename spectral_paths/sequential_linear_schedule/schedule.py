"""Vectorized sequential linear schedule in PCA coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch


TimeLike = Union[float, torch.Tensor]


@dataclass(frozen=True)
class SequentialLinearSchedule:
    """Reveal one PCA component per temporal interval.

    Parameters
    ----------
    dimension:
        Number of spectral components. CIFAR-10 uses ``3 * 32 * 32``.
    epsilon_1:
        Residual level after the first component's main ramp. It must lie
        strictly between zero and one.

    Notes
    -----
    Component ``k`` in the implementation is zero-indexed and corresponds to
    component ``k + 1`` in the mathematical notes. At a knot ``t < 1``,
    :meth:`dot_beta` returns the right derivative. At ``t = 1`` it returns
    the left limit.
    """

    dimension: int
    epsilon_1: float
    _geometry_cache: dict[
        tuple[str, torch.dtype],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least two.")
        if not 0.0 < self.epsilon_1 < 1.0:
            raise ValueError("epsilon_1 must lie strictly between zero and one.")

    @property
    def rho(self) -> float:
        """Return the common residual slope after a main ramp."""
        return self.dimension * self.epsilon_1 / (self.dimension - 1)

    def epsilon(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return residual levels ``epsilon_k`` with shape ``[D]``."""
        dtype = dtype or torch.get_default_dtype()
        _, _, _, epsilon = self._component_geometry(device=device, dtype=dtype)
        return epsilon

    def beta(
        self,
        t: TimeLike,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Evaluate ``beta_k(t)`` for all components without a Python loop.

        Parameters
        ----------
        t:
            Scalar time or one-dimensional tensor of times in ``[0, 1]``.
        device, dtype:
            Used when ``t`` is a Python scalar. Tensor inputs retain their
            device unless an explicit device is supplied.

        Returns
        -------
        torch.Tensor
            Shape ``[D]`` for a scalar time and ``[T, D]`` for ``T`` times.
        """
        time = self._as_time_tensor(t, device=device, dtype=dtype)
        time_column, indices, start, end, epsilon = self._geometry(time)

        main_value = (1.0 - epsilon) * (
            self.dimension * time_column - indices
        )
        residual_value = (
            1.0
            - epsilon
            + torch.as_tensor(self.rho, device=time.device, dtype=time.dtype)
            * (time_column - end)
        )
        beta = torch.where(
            time_column <= start,
            torch.zeros_like(main_value),
            torch.where(time_column < end, main_value, residual_value),
        )
        return beta.clamp(0.0, 1.0)

    def dot_beta(
        self,
        t: TimeLike,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Evaluate the documented right derivative of ``beta``.

        The value at ``t = 1`` is defined by the left limit because no right
        derivative exists at the terminal point.
        """
        time = self._as_time_tensor(t, device=device, dtype=dtype)
        time_column, indices, start, end, epsilon = self._geometry(time)

        main_slope = self.dimension * (1.0 - epsilon)
        residual_slope = torch.full_like(main_slope, self.rho)
        before = time_column < start
        on_main_ramp = (time_column >= start) & (time_column < end)

        derivative = torch.where(
            before,
            torch.zeros_like(main_slope),
            torch.where(on_main_ramp, main_slope, residual_slope),
        )

        # At t=1, component D is still on the left side of its main ramp.
        terminal_last = (time_column >= 1.0) & (
            indices == self.dimension - 1
        )
        return torch.where(terminal_last, main_slope, derivative)

    def knots(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return all temporal knots ``j / D``, including zero and one."""
        dtype = dtype or torch.get_default_dtype()
        return torch.linspace(
            0.0,
            1.0,
            self.dimension + 1,
            device=device,
            dtype=dtype,
        )

    def _as_time_tensor(
        self,
        t: TimeLike,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if isinstance(t, (float, int)) and not 0.0 <= float(t) <= 1.0:
            raise ValueError("time must lie in [0, 1].")

        if isinstance(t, torch.Tensor):
            target_device = device if device is not None else t.device
            target_dtype = dtype if dtype is not None else t.dtype
            time = t.to(device=target_device, dtype=target_dtype)
        else:
            time = torch.as_tensor(
                t,
                device=device,
                dtype=dtype or torch.get_default_dtype(),
            )

        if not time.is_floating_point():
            time = time.to(torch.get_default_dtype())
        if time.ndim > 1:
            raise ValueError("time must be a scalar or a one-dimensional tensor.")
        return time

    def _geometry(
        self,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, start, end, epsilon = self._component_geometry(
            device=time.device,
            dtype=time.dtype,
        )
        return time.unsqueeze(-1), indices, start, end, epsilon

    def _component_geometry(
        self,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        resolved_device = torch.device(device or "cpu")
        key = (str(resolved_device), dtype)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        indices = torch.arange(
            self.dimension,
            device=resolved_device,
            dtype=dtype,
        )
        start = indices / self.dimension
        end = (indices + 1.0) / self.dimension
        epsilon = self.epsilon_1 * (self.dimension - indices - 1.0) / (
            self.dimension - 1
        )
        geometry = (indices, start, end, epsilon)
        self._geometry_cache[key] = geometry
        return geometry
