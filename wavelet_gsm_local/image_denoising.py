"""Full-band posterior means for the image-level wavelet denoising study."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from wavelet_gsm_local.estimators import (
    gaussian_posterior_center_tensor,
    gaussian_self_posterior_center_tensor,
    gsm_local_posterior_center_tensor,
    gsm_self_posterior_center_tensor,
)
from wavelet_gsm_local.gsm import GSMParameters
from wavelet_gsm_local.patches import CENTER_INDEX


IMAGE_METHODS = (
    "wiener_matched_self",
    "wiener_matched_local",
    "wiener_empirical_local",
    "gsm_self",
    "gsm_local",
)


@dataclass(frozen=True, slots=True)
class VectorPosteriorResult:
    mean: torch.Tensor
    jitter: float


@torch.no_grad()
def gaussian_vector_posterior_tensor(
    noisy: torch.Tensor,
    time: float,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    *,
    max_jitter_attempts: int = 6,
) -> VectorPosteriorResult:
    """Posterior mean for a Gaussian vector under y_t=t*y_0+(1-t)*z.

    The small covariance system is solved in float64. No jitter is introduced
    unless Cholesky reports that the theoretically positive system is not
    numerically positive definite.
    """
    if noisy.ndim != 2 or mean.ndim != 1:
        raise ValueError("noisy and mean must have shapes [N,d] and [d].")
    dimension = noisy.shape[1]
    if mean.shape != (dimension,) or covariance.shape != (dimension, dimension):
        raise ValueError("Mean and covariance dimensions are inconsistent.")
    if noisy.dtype != torch.float64 or mean.dtype != torch.float64 or covariance.dtype != torch.float64:
        raise TypeError("The global vector Wiener calculation must use float64.")
    if time == 0.0:
        return VectorPosteriorResult(mean.expand_as(noisy).clone(), 0.0)
    if time == 1.0:
        return VectorPosteriorResult(noisy.clone(), 0.0)

    identity = torch.eye(dimension, device=noisy.device, dtype=torch.float64)
    system = time**2 * covariance + (1.0 - time) ** 2 * identity
    right_hand_side = time * covariance
    jitter = 0.0
    base_jitter = 1e-12 * max(
        float(torch.trace(system).item()) / dimension,
        1.0,
    )
    for attempt in range(max_jitter_attempts + 1):
        candidate = system if jitter == 0.0 else system + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        if int(info.max().item()) == 0:
            gain_transpose = torch.cholesky_solve(right_hand_side, factor).T
            posterior = mean + (noisy - time * mean) @ gain_transpose
            if not torch.isfinite(posterior).all():
                raise FloatingPointError("The global Wiener posterior is non-finite.")
            return VectorPosteriorResult(posterior, jitter)
        if attempt == max_jitter_attempts:
            break
        jitter = base_jitter if jitter == 0.0 else 10.0 * jitter
    raise RuntimeError(
        f"Cholesky failed for the {dimension}D approximation-band posterior."
    )


@torch.no_grad()
def predict_detail_band_tensor(
    noisy_band: torch.Tensor,
    time: float,
    empirical_mean: torch.Tensor,
    empirical_covariance: torch.Tensor,
    parameters: GSMParameters,
) -> dict[str, torch.Tensor]:
    """Denoise one complete detail band, using self estimators at its border."""
    if noisy_band.ndim != 3:
        raise ValueError("noisy_band must have shape [N,H,W].")
    num_images, height, width = noisy_band.shape
    if height < 3 or width < 3:
        raise ValueError("A detail band must be at least 3x3.")
    if empirical_mean.shape != (9,) or empirical_covariance.shape != (9, 9):
        raise ValueError("Empirical patch moments must have dimensions 9 and 9x9.")

    model_mean = torch.as_tensor(
        parameters.mean, device=noisy_band.device, dtype=noisy_band.dtype
    )
    model_covariance = torch.as_tensor(
        parameters.covariance, device=noisy_band.device, dtype=noisy_band.dtype
    )
    if time == 0.0:
        matched_prior = torch.ones_like(noisy_band) * model_mean[CENTER_INDEX]
        empirical_prior = torch.ones_like(noisy_band) * empirical_mean[CENTER_INDEX]
        return {
            "wiener_matched_self": matched_prior,
            "wiener_matched_local": matched_prior.clone(),
            "wiener_empirical_local": empirical_prior,
            "gsm_self": matched_prior.clone(),
            "gsm_local": matched_prior.clone(),
        }
    if time == 1.0:
        return {method: noisy_band.clone() for method in IMAGE_METHODS}

    noisy_flat = noisy_band.reshape(-1)
    matched_self = gaussian_self_posterior_center_tensor(
        noisy_flat,
        time,
        model_mean[CENTER_INDEX],
        model_covariance[CENTER_INDEX, CENTER_INDEX],
    ).reshape_as(noisy_band)
    empirical_self = gaussian_self_posterior_center_tensor(
        noisy_flat,
        time,
        empirical_mean[CENTER_INDEX],
        empirical_covariance[CENTER_INDEX, CENTER_INDEX],
    ).reshape_as(noisy_band)
    gsm_self = gsm_self_posterior_center_tensor(
        noisy_flat, time, parameters
    ).reshape_as(noisy_band)

    predictions = {
        "wiener_matched_self": matched_self,
        "wiener_matched_local": matched_self.clone(),
        "wiener_empirical_local": empirical_self.clone(),
        "gsm_self": gsm_self,
        "gsm_local": gsm_self.clone(),
    }

    # unfold returns one column per valid center in row-major spatial order.
    patches = functional.unfold(noisy_band[:, None], kernel_size=3)
    patches = patches.transpose(1, 2).reshape(-1, 9)
    expected_centers = num_images * (height - 2) * (width - 2)
    if patches.shape != (expected_centers, 9):
        raise RuntimeError("Unexpected number of valid 3x3 patches.")

    local_predictions = {
        "wiener_matched_local": gaussian_posterior_center_tensor(
            patches, time, model_mean, model_covariance
        ),
        "wiener_empirical_local": gaussian_posterior_center_tensor(
            patches, time, empirical_mean, empirical_covariance
        ),
        "gsm_local": gsm_local_posterior_center_tensor(patches, time, parameters),
    }
    interior_shape = (num_images, height - 2, width - 2)
    for method, values in local_predictions.items():
        predictions[method][:, 1:-1, 1:-1] = values.reshape(interior_shape)

    if tuple(predictions) != IMAGE_METHODS:
        raise RuntimeError("Image-level methods are incomplete or misordered.")
    return predictions
