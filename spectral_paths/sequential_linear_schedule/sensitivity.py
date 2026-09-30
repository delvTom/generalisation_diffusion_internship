"""Selected rows and spatial locality metrics for affine Jacobians."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .affine import AffineSpectralVelocity


@dataclass
class LocalSensitivity:
    """Spatial energy profile of one Jacobian row."""

    radii: torch.Tensor
    cumulative_mass: torch.Tensor
    effective_radius: float
    total_energy: float


@torch.no_grad()
def sensitivity_rows(
    model: AffineSpectralVelocity,
    U: torch.Tensor,
    t: float,
    q_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute selected rows of ``U diag(s(t)) U^T``.

    Parameters
    ----------
    model:
        Affine velocity model whose slope is diagonal in PCA coordinates.
    U:
        Full orthogonal PCA basis, shape ``[D, D]``.
    t:
        Evaluation time.
    q_indices:
        Flat output-coordinate indices, shape ``[Q]``.

    Returns
    -------
    torch.Tensor
        Selected Jacobian rows with shape ``[Q, D]``. The full Jacobian is
        never materialized.
    """
    if U.shape != (model.path.schedule.dimension, model.path.schedule.dimension):
        raise ValueError("U must have shape [D, D].")
    q_indices = q_indices.to(device=U.device, dtype=torch.long)
    slope = model.coefficients(t).slope
    selected_basis_rows = U.index_select(0, q_indices)
    return (selected_basis_rows * slope.unsqueeze(0)) @ U.T


def local_sensitivity_profile(
    row: torch.Tensor,
    *,
    target_y: int,
    target_x: int,
    channels: int,
    height: int,
    width: int,
    tau: float,
    zero_energy_tolerance: float,
) -> LocalSensitivity:
    """Aggregate squared sensitivity across channels by spatial radius."""
    if row.shape != (channels * height * width,):
        raise ValueError("The Jacobian row has an invalid image dimension.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between zero and one.")

    energy_map = row.view(channels, height, width).square().sum(dim=0)
    total_energy_tensor = energy_map.sum()
    total_energy = float(total_energy_tensor.detach().cpu())
    max_radius = math.ceil(
        max(
            math.hypot(target_y, target_x),
            math.hypot(target_y, width - 1 - target_x),
            math.hypot(height - 1 - target_y, target_x),
            math.hypot(height - 1 - target_y, width - 1 - target_x),
        )
    )
    radii = torch.arange(
        max_radius + 1,
        device=row.device,
        dtype=row.dtype,
    )

    if total_energy <= zero_energy_tolerance:
        return LocalSensitivity(
            radii=radii,
            cumulative_mass=torch.full_like(radii, torch.nan),
            effective_radius=float("nan"),
            total_energy=total_energy,
        )

    y_grid = torch.arange(height, device=row.device, dtype=row.dtype).unsqueeze(1)
    x_grid = torch.arange(width, device=row.device, dtype=row.dtype).unsqueeze(0)
    distances = torch.sqrt((y_grid - target_y).square() + (x_grid - target_x).square())
    cumulative = torch.stack(
        [energy_map[distances <= radius].sum() for radius in radii]
    ) / total_energy_tensor
    reached = torch.nonzero(cumulative >= tau, as_tuple=False)
    effective_radius = (
        float(radii[reached[0, 0]].detach().cpu())
        if reached.numel() > 0
        else float("nan")
    )
    return LocalSensitivity(
        radii=radii,
        cumulative_mass=cumulative,
        effective_radius=effective_radius,
        total_energy=total_energy,
    )
