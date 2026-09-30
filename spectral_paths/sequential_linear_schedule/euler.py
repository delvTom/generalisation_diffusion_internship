"""Euler integration utilities for the affine spectral field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .affine import AffineSpectralVelocity


@dataclass
class EulerResult:
    """Numerical trajectory and its errors against the exact solution."""

    final_state: torch.Tensor
    boundary_times: torch.Tensor
    relative_boundary_error: torch.Tensor
    final_component_mse: torch.Tensor


@dataclass(frozen=True)
class DiagonalAffineMap:
    """Diagonal affine map ``z -> scale * z + bias`` in PCA coordinates."""

    scale: torch.Tensor
    bias: torch.Tensor

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the map to a tensor whose last dimension is the PCA dimension."""
        if z.shape[-1] != self.scale.numel():
            raise ValueError("z and the affine map have different dimensions.")
        return self.scale * z + self.bias


@dataclass(frozen=True)
class AffineEulerMapResult:
    """Euler endpoint map and optional MSE observations along its trajectory."""

    affine_map: DiagonalAffineMap
    observation_steps: torch.Tensor
    observation_times: torch.Tensor
    observation_mse: torch.Tensor | None


@torch.no_grad()
def exact_affine_map(
    model: AffineSpectralVelocity,
    t: float | torch.Tensor,
) -> DiagonalAffineMap:
    """Return the exact affine ODE map from time zero to ``t``."""
    coefficients = model.coefficients(t)
    if coefficients.beta.ndim != 1:
        raise ValueError("exact_affine_map expects one scalar time.")
    return DiagonalAffineMap(
        scale=torch.sqrt(
            coefficients.variance.clamp_min(model.variance_floor)
        ),
        bias=coefficients.beta * model.mu_z,
    )


@torch.no_grad()
def integrate_classic_fm_wiener_legacy_map(
    lambda_sq: torch.Tensor,
    mu_z: torch.Tensor,
    num_steps: int,
    *,
    variance_floor: float = 1e-12,
) -> DiagonalAffineMap:
    """Reproduce the repository's legacy classic-FM Wiener sampler.

    The historical generation script evolves pixel-space images with

    ``x_{n+1} = x_n + h f_W(x_n, (n + 1/2) h)``.

    This is not the left-endpoint Euler convention used for the sequential
    paths in this module. In the PCA basis, however, the dense pixel-space
    Wiener operator is diagonal, so the same sampler can be composed cheaply
    as ``z_N = scale * z_0 + bias``. For a matched spectral noise ``z_0``, the
    corresponding legacy pixel noise is ``U z_0``.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if lambda_sq.ndim != 1 or mu_z.shape != lambda_sq.shape:
        raise ValueError("lambda_sq and mu_z must have the same shape [D].")
    if lambda_sq.device != mu_z.device or lambda_sq.dtype != mu_z.dtype:
        raise ValueError("lambda_sq and mu_z must share a device and dtype.")

    scale = torch.ones_like(lambda_sq)
    bias = torch.zeros_like(mu_z)
    step_size = 1.0 / num_steps

    for step in range(num_steps):
        # This midpoint field-evaluation convention intentionally matches
        # fm_comparisons/generate_fm_unet_closed_form_samples.py.
        t = (step + 0.5) * step_size
        variance = (1.0 - t) ** 2 + lambda_sq * t**2
        slope = (lambda_sq * t - (1.0 - t)) / variance.clamp_min(
            variance_floor
        )
        intercept = (1.0 - slope * t) * mu_z
        gain = 1.0 + step_size * slope

        bias = gain * bias + step_size * intercept
        scale = gain * scale

    return DiagonalAffineMap(scale=scale, bias=bias)


@torch.no_grad()
def integrate_uniform_affine_map(
    model: AffineSpectralVelocity,
    num_steps: int,
    *,
    observation_steps: Sequence[int] | torch.Tensor | None = None,
    noise_mean: torch.Tensor | None = None,
    noise_second_moment: torch.Tensor | None = None,
    time_chunk_size: int = 256,
) -> AffineEulerMapResult:
    """Integrate uniform Euler as one diagonal affine map.

    The affine field has the form ``f(z,t) = s(t) * z + c(t)``. One Euler
    step is therefore itself affine:

    ``z_{n+1} = (1 + h s_n) * z_n + h c_n``.

    Composing those maps yields ``z_N = a_N * z_0 + b_N``. This function
    composes short time blocks tensorially, so its cost is independent of the
    number of sampled initial noises. It is mathematically the same explicit
    Euler scheme as :func:`integrate_uniform_euler`.

    When observation steps and empirical noise moments are supplied, the
    returned MSE is

    ``mean_{b,k}[(z_Euler[b,k] - z_exact[b,k])**2]``

    evaluated without materializing the complete ``[B, D]`` state at every
    observation time.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be positive.")

    dimension = model.path.schedule.dimension
    device = model.lambda_sq.device
    dtype = model.lambda_sq.dtype
    requested_steps = _validated_observation_steps(observation_steps, num_steps)
    should_measure_mse = noise_mean is not None or noise_second_moment is not None
    if should_measure_mse:
        if noise_mean is None or noise_second_moment is None:
            raise ValueError(
                "noise_mean and noise_second_moment must be supplied together."
            )
        if noise_mean.shape != (dimension,) or noise_second_moment.shape != (
            dimension,
        ):
            raise ValueError("Noise moments must both have shape [D].")
        noise_mean = noise_mean.to(device=device, dtype=dtype)
        noise_second_moment = noise_second_moment.to(device=device, dtype=dtype)

    scale = torch.ones(dimension, device=device, dtype=dtype)
    bias = torch.zeros(dimension, device=device, dtype=dtype)
    step_size = 1.0 / num_steps
    mse_values: list[torch.Tensor] = []

    current_step = 0
    for target_step in requested_steps:
        while current_step < target_step:
            block_end = min(current_step + time_chunk_size, target_step)
            block_scale, block_bias = _euler_block_map(
                model,
                start_step=current_step,
                end_step=block_end,
                num_steps=num_steps,
            )
            bias = block_scale * bias + block_bias
            scale = block_scale * scale
            current_step = block_end

        if should_measure_mse:
            time_value = target_step / num_steps
            exact_map = exact_affine_map(model, time_value)
            mse_values.append(
                affine_map_mse_from_noise_moments(
                    numerical=DiagonalAffineMap(scale=scale, bias=bias),
                    exact=exact_map,
                    noise_mean=noise_mean,
                    noise_second_moment=noise_second_moment,
                )
            )

    observation_steps_tensor = torch.as_tensor(
        requested_steps,
        device=device,
        dtype=torch.long,
    )
    observation_times = observation_steps_tensor.to(dtype=dtype) / num_steps
    return AffineEulerMapResult(
        affine_map=DiagonalAffineMap(scale=scale, bias=bias),
        observation_steps=observation_steps_tensor,
        observation_times=observation_times,
        observation_mse=(torch.stack(mse_values) if mse_values else None),
    )


@torch.no_grad()
def affine_map_mse_from_noise_moments(
    *,
    numerical: DiagonalAffineMap,
    exact: DiagonalAffineMap,
    noise_mean: torch.Tensor,
    noise_second_moment: torch.Tensor,
) -> torch.Tensor:
    """Compute empirical map MSE from first and second moments of ``z_0``."""
    dimension = numerical.scale.numel()
    tensors = (
        numerical.bias,
        exact.scale,
        exact.bias,
        noise_mean,
        noise_second_moment,
    )
    if numerical.scale.shape != (dimension,) or any(
        tensor.shape != (dimension,) for tensor in tensors
    ):
        raise ValueError("All maps and moments must contain vectors of shape [D].")

    delta_scale = numerical.scale - exact.scale
    delta_bias = numerical.bias - exact.bias
    component_mse = (
        delta_scale.square() * noise_second_moment
        + 2.0 * delta_scale * delta_bias * noise_mean
        + delta_bias.square()
    )
    return component_mse.mean().clamp_min(0.0)


def _validated_observation_steps(
    observation_steps: Sequence[int] | torch.Tensor | None,
    num_steps: int,
) -> list[int]:
    """Return sorted observation steps including zero and the endpoint."""
    if observation_steps is None:
        steps = [0, num_steps]
    elif isinstance(observation_steps, torch.Tensor):
        steps = [int(value) for value in observation_steps.detach().cpu().tolist()]
    else:
        steps = [int(value) for value in observation_steps]
    steps.extend([0, num_steps])
    steps = sorted(set(steps))
    if any(step < 0 or step > num_steps for step in steps):
        raise ValueError("Observation steps must lie in [0, num_steps].")
    return steps


@torch.no_grad()
def _euler_block_map(
    model: AffineSpectralVelocity,
    *,
    start_step: int,
    end_step: int,
    num_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose consecutive Euler steps into one diagonal affine map."""
    if not 0 <= start_step < end_step <= num_steps:
        raise ValueError("Expected 0 <= start_step < end_step <= num_steps.")

    device = model.lambda_sq.device
    dtype = model.lambda_sq.dtype
    step_size = 1.0 / num_steps
    step_indices = torch.arange(start_step, end_step, device=device, dtype=dtype)
    times = step_indices / num_steps
    coefficients = model.coefficients(times)
    gains = 1.0 + step_size * coefficients.slope
    increments = step_size * coefficients.bias

    # For maps T_i(z) = g_i z + d_i, their chronological composition is
    # prod_i(g_i) z + sum_i d_i prod_{j>i}(g_j). No division is used, so the
    # formula remains valid when a gain is exactly zero.
    reverse_inclusive_products = torch.flip(
        torch.cumprod(torch.flip(gains, dims=(0,)), dim=0),
        dims=(0,),
    )
    suffix_after = torch.cat(
        [
            reverse_inclusive_products[1:],
            torch.ones(
                1,
                gains.shape[1],
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )
    block_scale = reverse_inclusive_products[0]
    block_bias = (increments * suffix_after).sum(dim=0)
    return block_scale, block_bias


@torch.no_grad()
def integrate_knot_aligned_euler(
    model: AffineSpectralVelocity,
    z0: torch.Tensor,
    steps_per_interval: int,
    *,
    eta: float = 1e-12,
) -> EulerResult:
    """Integrate with forward Euler without crossing schedule knots.

    Each interval ``[j / D, (j + 1) / D]`` is split into an equal number of
    substeps. The field is evaluated at the left endpoint, which uses the
    schedule's documented right-derivative convention. The final field
    evaluation occurs strictly before ``t = 1``.
    """
    if steps_per_interval <= 0:
        raise ValueError("steps_per_interval must be positive.")
    if z0.ndim != 2:
        raise ValueError("z0 must have shape [B, D].")

    dimension = model.path.schedule.dimension
    if z0.shape[-1] != dimension:
        raise ValueError("z0 and the affine model have different dimensions.")

    state = z0.clone()
    step_size = 1.0 / (dimension * steps_per_interval)
    errors = [torch.zeros((), device=z0.device, dtype=z0.dtype)]

    for interval in range(dimension):
        for substep in range(steps_per_interval):
            t = (interval * steps_per_interval + substep) * step_size
            state = state + step_size * model.field(state, t)

        boundary_time = (interval + 1) / dimension
        exact = model.exact_state(z0, boundary_time)
        numerator = torch.linalg.vector_norm(state - exact, dim=-1)
        denominator = torch.linalg.vector_norm(exact, dim=-1) + eta
        errors.append((numerator / denominator).mean())

    exact_final = model.exact_state(z0, 1.0)
    final_component_mse = (state - exact_final).square().mean(dim=0)
    return EulerResult(
        final_state=state,
        boundary_times=model.path.schedule.knots(
            device=z0.device,
            dtype=z0.dtype,
        ),
        relative_boundary_error=torch.stack(errors),
        final_component_mse=final_component_mse,
    )


@torch.no_grad()
def integrate_uniform_euler(
    model: AffineSpectralVelocity,
    z0: torch.Tensor,
    num_steps: int,
    *,
    eta: float = 1e-12,
) -> EulerResult:
    """Integrate with uniform forward-Euler steps on ``[0, 1]``.

    This deliberately allows steps to cross spectral knots. It is therefore a
    negative-control solver for the sequential schedule, not the reference
    numerical method.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if z0.ndim != 2 or z0.shape[-1] != model.path.schedule.dimension:
        raise ValueError("z0 must have shape [B, D].")

    state = z0.clone()
    step_size = 1.0 / num_steps
    times = torch.linspace(
        0.0,
        1.0,
        num_steps + 1,
        device=z0.device,
        dtype=z0.dtype,
    )
    errors = [torch.zeros((), device=z0.device, dtype=z0.dtype)]
    for step in range(num_steps):
        time = step / num_steps
        state = state + step_size * model.field(state, time)
        exact = model.exact_state(z0, (step + 1) / num_steps)
        numerator = torch.linalg.vector_norm(state - exact, dim=-1)
        denominator = torch.linalg.vector_norm(exact, dim=-1) + eta
        errors.append((numerator / denominator).mean())

    exact_final = model.exact_state(z0, 1.0)
    return EulerResult(
        final_state=state,
        boundary_times=times,
        relative_boundary_error=torch.stack(errors),
        final_component_mse=(state - exact_final).square().mean(dim=0),
    )


@torch.no_grad()
def integrate_exact_knot_transitions(
    model: AffineSpectralVelocity,
    z0: torch.Tensor,
    *,
    eta: float = 1e-12,
) -> EulerResult:
    """Compose the exact affine transition over all spectral intervals."""
    if z0.ndim != 2 or z0.shape[-1] != model.path.schedule.dimension:
        raise ValueError("z0 must have shape [B, D].")

    knots = model.path.schedule.knots(device=z0.device, dtype=z0.dtype)
    state = z0.clone()
    errors = [torch.zeros((), device=z0.device, dtype=z0.dtype)]
    for interval in range(model.path.schedule.dimension):
        t_from = interval / model.path.schedule.dimension
        t_to = (interval + 1) / model.path.schedule.dimension
        state = model.exact_transition(state, t_from, t_to)
        exact = model.exact_state(z0, t_to)
        numerator = torch.linalg.vector_norm(state - exact, dim=-1)
        denominator = torch.linalg.vector_norm(exact, dim=-1) + eta
        errors.append((numerator / denominator).mean())

    exact_final = model.exact_state(z0, 1.0)
    return EulerResult(
        final_state=state,
        boundary_times=knots,
        relative_boundary_error=torch.stack(errors),
        final_component_mse=(state - exact_final).square().mean(dim=0),
    )
