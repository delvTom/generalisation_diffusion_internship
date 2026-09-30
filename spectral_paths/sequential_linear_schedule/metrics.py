"""Moment diagnostics for exact affine spectral trajectories."""

from __future__ import annotations

from typing import Sequence

import torch

from .affine import AffineSpectralVelocity


@torch.no_grad()
def exact_moment_metrics(
    model: AffineSpectralVelocity,
    z0: torch.Tensor,
    times: Sequence[float],
    U: torch.Tensor,
    *,
    spectral_indices: Sequence[int],
    pixel_indices: Sequence[int],
    eta: float = 1e-12,
) -> dict[str, dict[str, float | list[list[float]]]]:
    """Compare empirical exact-flow moments to their theoretical values.

    Full spectral means and variances are checked. Pixel covariance is checked
    on the explicitly selected coordinates, avoiding an unnecessary dense
    ``D x D`` empirical covariance matrix.
    """
    if z0.ndim != 2:
        raise ValueError("z0 must have shape [B, D].")
    spectral_index_tensor = torch.as_tensor(
        spectral_indices,
        device=z0.device,
        dtype=torch.long,
    )
    pixel_index_tensor = torch.as_tensor(
        pixel_indices,
        device=z0.device,
        dtype=torch.long,
    )
    selected_U_rows = U.index_select(0, pixel_index_tensor)
    metrics: dict[str, dict[str, float | list[list[float]]]] = {}

    for time in times:
        state = model.exact_state(z0, float(time))
        coefficients = model.coefficients(float(time))
        expected_mean_z = coefficients.beta * model.mu_z
        expected_variance_z = coefficients.variance

        empirical_mean_z = state.mean(dim=0)
        empirical_variance_z = state.var(dim=0, unbiased=False)
        mean_residual_z = empirical_mean_z - expected_mean_z
        mean_error_over_total_std = torch.linalg.vector_norm(
            mean_residual_z
        ) / (torch.sqrt(expected_variance_z.sum()) + eta)
        mean_monte_carlo_z_rms = torch.sqrt(
            (
                mean_residual_z.square()
                / (expected_variance_z / z0.shape[0] + eta)
            ).mean()
        )
        variance_error = torch.linalg.vector_norm(
            empirical_variance_z - expected_variance_z
        ) / (torch.linalg.vector_norm(expected_variance_z) + eta)

        selected_z = state.index_select(1, spectral_index_tensor)
        selected_z_centered = selected_z - selected_z.mean(dim=0)
        selected_covariance = selected_z_centered.T @ selected_z_centered / z0.shape[0]
        selected_diagonal = expected_variance_z.index_select(
            0, spectral_index_tensor
        )
        expected_selected_covariance = torch.diag(selected_diagonal)
        selected_covariance_error = torch.linalg.matrix_norm(
            selected_covariance - expected_selected_covariance
        ) / (torch.linalg.matrix_norm(expected_selected_covariance) + eta)
        off_diagonal = selected_covariance - torch.diag(
            torch.diagonal(selected_covariance)
        )
        selected_std = torch.sqrt(
            torch.diagonal(selected_covariance).clamp_min(eta)
        )
        selected_correlation = selected_covariance / (
            selected_std.unsqueeze(0) * selected_std.unsqueeze(1)
        )
        off_diagonal_correlation = selected_correlation - torch.diag(
            torch.diagonal(selected_correlation)
        )

        empirical_mean_x = empirical_mean_z @ U.T
        expected_mean_x = expected_mean_z @ U.T
        pixel_mean_error_over_total_std = torch.linalg.vector_norm(
            empirical_mean_x - expected_mean_x
        ) / (torch.sqrt(expected_variance_z.sum()) + eta)

        selected_x = state @ selected_U_rows.T
        selected_x_centered = selected_x - selected_x.mean(dim=0)
        empirical_pixel_covariance = (
            selected_x_centered.T @ selected_x_centered / z0.shape[0]
        )
        expected_pixel_covariance = (
            selected_U_rows * expected_variance_z.unsqueeze(0)
        ) @ selected_U_rows.T
        expected_selected_pixel_mean = expected_mean_z @ selected_U_rows.T
        empirical_selected_pixel_mean = selected_x.mean(dim=0)
        selected_pixel_mean_mc_z_rms = torch.sqrt(
            (
                (empirical_selected_pixel_mean - expected_selected_pixel_mean).square()
                / (
                    torch.diagonal(expected_pixel_covariance) / z0.shape[0]
                    + eta
                )
            ).mean()
        )
        pixel_covariance_error = torch.linalg.matrix_norm(
            empirical_pixel_covariance - expected_pixel_covariance
        ) / (torch.linalg.matrix_norm(expected_pixel_covariance) + eta)

        metrics[f"{float(time):.8f}"] = {
            "spectral_mean_error_over_total_std": float(
                mean_error_over_total_std.cpu()
            ),
            "spectral_mean_monte_carlo_z_rms": float(mean_monte_carlo_z_rms.cpu()),
            "spectral_variance_relative_l2": float(variance_error.cpu()),
            "selected_spectral_covariance_relative_frobenius": float(
                selected_covariance_error.cpu()
            ),
            "selected_spectral_off_diagonal_max_abs": float(
                off_diagonal.abs().max().cpu()
            ),
            "selected_spectral_off_diagonal_correlation_max_abs": float(
                off_diagonal_correlation.abs().max().cpu()
            ),
            "pixel_mean_error_over_total_std": float(
                pixel_mean_error_over_total_std.cpu()
            ),
            "selected_pixel_mean_monte_carlo_z_rms": float(
                selected_pixel_mean_mc_z_rms.cpu()
            ),
            "selected_pixel_covariance_relative_frobenius": float(
                pixel_covariance_error.cpu()
            ),
            "empirical_selected_pixel_covariance": empirical_pixel_covariance.cpu().tolist(),
            "expected_selected_pixel_covariance": expected_pixel_covariance.cpu().tolist(),
        }
    return metrics
