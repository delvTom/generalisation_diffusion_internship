"""Posterior means for Gaussian and GSM wavelet-patch priors."""

from __future__ import annotations

import numpy as np
import torch

from wavelet_gsm_local.gsm import GSMParameters, _logsumexp
from wavelet_gsm_local.patches import CENTER_INDEX


METHODS = (
    "wiener_empirical_local",
    "wiener_matched_local",
    "wiener_matched_self",
    "gsm_self",
    "gsm_local",
)


@torch.no_grad()
def _tensor_parameters(
    parameters: GSMParameters,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        for value in (
            parameters.mean,
            parameters.covariance,
            parameters.mixing_weights,
            parameters.scales,
        )
    )  # type: ignore[return-value]


@torch.no_grad()
def gaussian_posterior_center_tensor(
    noisy_patches: torch.Tensor,
    time: float,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    if time == 0.0:
        return torch.ones_like(noisy_patches[:, CENTER_INDEX]) * mean[CENTER_INDEX]
    if time == 1.0:
        return noisy_patches[:, CENTER_INDEX].clone()
    identity = torch.eye(9, device=noisy_patches.device, dtype=noisy_patches.dtype)
    noisy_covariance = time**2 * covariance + (1.0 - time) ** 2 * identity
    center_gain = torch.linalg.solve(
        noisy_covariance, time * covariance[CENTER_INDEX]
    )
    return mean[CENTER_INDEX] + (noisy_patches - time * mean) @ center_gain


@torch.no_grad()
def gaussian_self_posterior_center_tensor(
    noisy_center: torch.Tensor,
    time: float,
    mean_center: torch.Tensor,
    variance_center: torch.Tensor,
) -> torch.Tensor:
    if time == 0.0:
        return torch.ones_like(noisy_center) * mean_center
    if time == 1.0:
        return noisy_center.clone()
    denominator = time**2 * variance_center + (1.0 - time) ** 2
    gain = time * variance_center / denominator
    return mean_center + gain * (noisy_center - time * mean_center)


@torch.no_grad()
def gsm_self_posterior_center_tensor(
    noisy_center: torch.Tensor,
    time: float,
    parameters: GSMParameters,
    *,
    return_responsibilities: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    mean, covariance, mixing, scales = _tensor_parameters(parameters, noisy_center)
    mean_center = mean[CENTER_INDEX]
    variance = covariance[CENTER_INDEX, CENTER_INDEX]
    component_variances = time**2 * scales * variance + (1.0 - time) ** 2
    residual = noisy_center - time * mean_center
    log_probabilities = (
        torch.log(mixing)[None, :]
        - 0.5 * torch.log(2.0 * torch.pi * component_variances)[None, :]
        - 0.5 * residual[:, None].square() / component_variances[None, :]
    )
    gamma = torch.softmax(log_probabilities, dim=1)
    gains = time * scales * variance / component_variances
    component_means = mean_center + residual[:, None] * gains[None, :]
    prediction = torch.sum(gamma * component_means, dim=1)
    if return_responsibilities:
        return prediction, gamma
    return prediction


@torch.no_grad()
def gsm_local_posterior_center_tensor(
    noisy_patches: torch.Tensor,
    time: float,
    parameters: GSMParameters,
    *,
    return_responsibilities: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    mean, covariance, mixing, scales = _tensor_parameters(parameters, noisy_patches)
    residual = noisy_patches - time * mean
    identity = torch.eye(9, device=noisy_patches.device, dtype=noisy_patches.dtype)
    log_probabilities: list[torch.Tensor] = []
    component_means: list[torch.Tensor] = []
    for state in range(2):
        noisy_covariance = (
            time**2 * scales[state] * covariance + (1.0 - time) ** 2 * identity
        )
        sign, log_determinant = torch.linalg.slogdet(noisy_covariance)
        if float(sign.item()) <= 0.0:
            raise RuntimeError("Noisy GSM covariance is not positive definite.")
        solved = torch.linalg.solve(noisy_covariance, residual.T).T
        mahalanobis = torch.sum(residual * solved, dim=1)
        log_probabilities.append(
            torch.log(mixing[state])
            - 0.5 * (9 * np.log(2.0 * np.pi) + log_determinant + mahalanobis)
        )
        center_gain = torch.linalg.solve(
            noisy_covariance, time * scales[state] * covariance[CENTER_INDEX]
        )
        component_means.append(mean[CENTER_INDEX] + residual @ center_gain)
    gamma = torch.softmax(torch.stack(log_probabilities, dim=1), dim=1)
    prediction = torch.sum(gamma * torch.stack(component_means, dim=1), dim=1)
    if return_responsibilities:
        return prediction, gamma
    return prediction


@torch.no_grad()
def predict_all_tensor(
    noisy_patches: torch.Tensor,
    time: float,
    empirical_mean: torch.Tensor,
    empirical_covariance: torch.Tensor,
    parameters: GSMParameters,
) -> dict[str, torch.Tensor]:
    mean, covariance, _, _ = _tensor_parameters(parameters, noisy_patches)
    return {
        "wiener_empirical_local": gaussian_posterior_center_tensor(
            noisy_patches, time, empirical_mean, empirical_covariance
        ),
        "wiener_matched_local": gaussian_posterior_center_tensor(
            noisy_patches, time, mean, covariance
        ),
        "wiener_matched_self": gaussian_self_posterior_center_tensor(
            noisy_patches[:, CENTER_INDEX],
            time,
            mean[CENTER_INDEX],
            covariance[CENTER_INDEX, CENTER_INDEX],
        ),
        "gsm_self": gsm_self_posterior_center_tensor(
            noisy_patches[:, CENTER_INDEX], time, parameters
        ),
        "gsm_local": gsm_local_posterior_center_tensor(
            noisy_patches, time, parameters
        ),
    }


def gaussian_posterior_center(
    noisy_patches: np.ndarray,
    time: float,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    if time == 0.0:
        return np.full(len(noisy_patches), mean[CENTER_INDEX])
    if time == 1.0:
        return noisy_patches[:, CENTER_INDEX].copy()
    noise_variance = (1.0 - time) ** 2
    noisy_covariance = time**2 * covariance + noise_variance * np.eye(9)
    center_gain = np.linalg.solve(
        noisy_covariance, (time * covariance[CENTER_INDEX]).T
    )
    return mean[CENTER_INDEX] + (noisy_patches - time * mean) @ center_gain


def gaussian_self_posterior_center(
    noisy_center: np.ndarray,
    time: float,
    mean_center: float,
    variance_center: float,
) -> np.ndarray:
    if time == 0.0:
        return np.full_like(noisy_center, mean_center)
    if time == 1.0:
        return noisy_center.copy()
    denominator = time**2 * variance_center + (1.0 - time) ** 2
    gain = time * variance_center / denominator
    return mean_center + gain * (noisy_center - time * mean_center)


def gsm_self_posterior_center(
    noisy_center: np.ndarray,
    time: float,
    parameters: GSMParameters,
    *,
    return_responsibilities: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    mean_center = parameters.mean[CENTER_INDEX]
    variance = parameters.covariance[CENTER_INDEX, CENTER_INDEX]
    component_variances = time**2 * parameters.scales * variance + (1.0 - time) ** 2
    residual = noisy_center - time * mean_center
    log_probabilities = (
        np.log(parameters.mixing_weights)[None, :]
        - 0.5 * np.log(2.0 * np.pi * component_variances)[None, :]
        - 0.5 * residual[:, None] ** 2 / component_variances[None, :]
    )
    log_normalizer = _logsumexp(log_probabilities, axis=1)
    gamma = np.exp(log_probabilities - log_normalizer[:, None])
    gains = time * parameters.scales * variance / component_variances
    component_means = mean_center + residual[:, None] * gains[None, :]
    prediction = np.sum(gamma * component_means, axis=1)
    if return_responsibilities:
        return prediction, gamma
    return prediction


def gsm_local_posterior_center(
    noisy_patches: np.ndarray,
    time: float,
    parameters: GSMParameters,
    *,
    return_responsibilities: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    residual = noisy_patches - time * parameters.mean
    log_probabilities: list[np.ndarray] = []
    component_means: list[np.ndarray] = []
    for weight, scale in zip(
        parameters.mixing_weights, parameters.scales, strict=True
    ):
        covariance = time**2 * scale * parameters.covariance + (1.0 - time) ** 2 * np.eye(9)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("Noisy GSM covariance is not positive definite.")
        solved = np.linalg.solve(covariance, residual.T).T
        mahalanobis = np.sum(residual * solved, axis=1)
        log_probabilities.append(
            np.log(weight) - 0.5 * (9 * np.log(2.0 * np.pi) + log_determinant + mahalanobis)
        )
        center_gain = np.linalg.solve(
            covariance, time * scale * parameters.covariance[CENTER_INDEX]
        )
        component_means.append(
            parameters.mean[CENTER_INDEX] + residual @ center_gain
        )
    log_probabilities_array = np.stack(log_probabilities, axis=1)
    log_normalizer = _logsumexp(log_probabilities_array, axis=1)
    gamma = np.exp(log_probabilities_array - log_normalizer[:, None])
    prediction = np.sum(gamma * np.stack(component_means, axis=1), axis=1)
    if return_responsibilities:
        return prediction, gamma
    return prediction


def predict_all(
    noisy_patches: np.ndarray,
    time: float,
    empirical_mean: np.ndarray,
    empirical_covariance: np.ndarray,
    parameters: GSMParameters,
) -> dict[str, np.ndarray]:
    return {
        "wiener_empirical_local": gaussian_posterior_center(
            noisy_patches, time, empirical_mean, empirical_covariance
        ),
        "wiener_matched_local": gaussian_posterior_center(
            noisy_patches, time, parameters.mean, parameters.covariance
        ),
        "wiener_matched_self": gaussian_self_posterior_center(
            noisy_patches[:, CENTER_INDEX],
            time,
            parameters.mean[CENTER_INDEX],
            parameters.covariance[CENTER_INDEX, CENTER_INDEX],
        ),
        "gsm_self": gsm_self_posterior_center(
            noisy_patches[:, CENTER_INDEX], time, parameters
        ),
        "gsm_local": gsm_local_posterior_center(noisy_patches, time, parameters),
    }
