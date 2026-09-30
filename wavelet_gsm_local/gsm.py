"""Explicit two-state proportional-covariance Gaussian scale mixture."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class GSMParameters:
    mean: np.ndarray  # [d], fixed empirical mean
    covariance: np.ndarray  # [d,d], marginal covariance under E[S]=1
    mixing_weights: np.ndarray  # [2]
    scales: np.ndarray  # [2], low then high
    log_likelihood: float
    iterations: int
    converged: bool
    scale_floor_hits: int
    likelihood_decreases: int
    condition_number: float
    restart: int

    def __post_init__(self) -> None:
        dimension = len(self.mean)
        if self.covariance.shape != (dimension, dimension):
            raise ValueError("GSM covariance has an incompatible shape.")
        if self.mixing_weights.shape != (2,) or self.scales.shape != (2,):
            raise ValueError("This experiment requires exactly two states.")
        if not np.all(self.mixing_weights > 0.0) or not np.all(self.scales > 0.0):
            raise ValueError("GSM weights and scales must be positive.")
        if not np.isclose(self.mixing_weights.sum(), 1.0, atol=1e-9):
            raise ValueError("GSM weights must sum to one.")
        if not np.isclose(np.dot(self.mixing_weights, self.scales), 1.0, atol=1e-8):
            raise ValueError("The GSM must satisfy E[S]=1.")
        if self.scales[0] > self.scales[1]:
            raise ValueError("GSM states must be ordered from low to high scale.")
        try:
            np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("GSM covariance must be positive definite.") from error

    @property
    def scale_variance(self) -> float:
        return float(np.dot(self.mixing_weights, self.scales**2) - 1.0)

    @property
    def state_collapse(self) -> bool:
        relative_gap = (self.scales[1] - self.scales[0]) / max(self.scales[1], 1e-15)
        return bool(relative_gap <= 1e-3)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def component_log_probabilities(
    observations: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    mixing_weights: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Return log pi_k N(y; mean, s_k covariance) with shape [N,K]."""
    centered = observations - mean
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("GSM covariance is not positive definite.")
    solved = np.linalg.solve(covariance, centered.T).T
    mahalanobis = np.sum(centered * solved, axis=1)
    dimension = observations.shape[1]
    normalizer = dimension * log(2.0 * pi) + log_determinant
    return np.stack(
        [
            np.log(mixing_weights[state])
            - 0.5
            * (
                normalizer
                + dimension * np.log(scales[state])
                + mahalanobis / scales[state]
            )
            for state in range(len(scales))
        ],
        axis=1,
    )


def responsibilities(
    observations: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    mixing_weights: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_probabilities = component_log_probabilities(
        observations, mean, covariance, mixing_weights, scales
    )
    log_normalizer = _logsumexp(log_probabilities, axis=1)
    gamma = np.exp(log_probabilities - log_normalizer[:, None])
    return gamma, float(log_normalizer.sum())


def _enforce_scale_constraint(
    mixing_weights: np.ndarray,
    scales: np.ndarray,
    covariance: np.ndarray,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Impose E[S]=1, then enforce the low-scale floor exactly."""
    normalization = float(np.dot(mixing_weights, scales))
    scales = scales / normalization
    covariance = covariance * normalization
    floor_hit = bool(np.any(scales < scale_floor))
    if floor_hit:
        low = int(np.argmin(scales))
        high = 1 - low
        scales[low] = scale_floor
        remaining = 1.0 - mixing_weights[low] * scale_floor
        scales[high] = remaining / mixing_weights[high]
        if scales[high] <= 0.0:
            raise FloatingPointError("The scale floor is incompatible with E[S]=1.")
    return scales, covariance, floor_hit


def _initial_parameters(
    centered: np.ndarray,
    restart: int,
    seed: int,
    covariance_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed + restart)
    covariance = centered.T @ centered / len(centered)
    covariance = 0.5 * (covariance + covariance.T)
    jitter = covariance_epsilon * max(float(np.trace(covariance) / centered.shape[1]), 1e-12)
    covariance = covariance + jitter * np.eye(centered.shape[1])
    high_weight = float(generator.uniform(0.15, 0.45))
    mixing = np.asarray([1.0 - high_weight, high_weight])
    low_scale = float(generator.uniform(0.15, 0.75))
    high_scale = (1.0 - mixing[0] * low_scale) / mixing[1]
    return covariance, mixing, np.asarray([low_scale, high_scale])


def fit_two_state_gsm(
    patches: np.ndarray,
    *,
    num_restarts: int = 5,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
    covariance_epsilon: float = 1e-6,
    scale_floor: float = 1e-4,
    seed: int = 0,
) -> tuple[GSMParameters, list[dict[str, float | int | bool]]]:
    """Fit a two-state GSM by ECM while holding mu at the empirical mean."""
    if patches.ndim != 2 or patches.shape[1] != 9 or len(patches) < 10:
        raise ValueError("patches must have shape [N,9] with N at least ten.")
    if min(num_restarts, max_iterations) < 1 or min(tolerance, covariance_epsilon, scale_floor) <= 0:
        raise ValueError("Invalid GSM optimization settings.")
    observations = np.asarray(patches, dtype=np.float64)
    mean = observations.mean(axis=0)
    centered = observations - mean
    best: GSMParameters | None = None
    histories: list[dict[str, float | int | bool]] = []

    for restart in range(num_restarts):
        covariance, mixing, scales = _initial_parameters(
            centered, restart, seed, covariance_epsilon
        )
        previous_likelihood = -np.inf
        floor_hits = 0
        decreases = 0
        converged = False
        likelihood_before_regularization = -np.inf
        gamma, previous_likelihood = responsibilities(
            observations, mean, covariance, mixing, scales
        )

        for iteration in range(1, max_iterations + 1):
            counts = gamma.sum(axis=0)
            mixing_new = np.maximum(counts / len(observations), 1e-12)
            mixing_new /= mixing_new.sum()

            solved = np.linalg.solve(covariance, centered.T).T
            mahalanobis = np.sum(centered * solved, axis=1)
            scales_new = (gamma * mahalanobis[:, None]).sum(axis=0)
            scales_new /= observations.shape[1] * counts
            raw_floor_hit = bool(np.any(scales_new < scale_floor))
            scales_new = np.maximum(scales_new, scale_floor)

            sample_weights = np.sum(gamma / scales_new[None, :], axis=1)
            covariance_raw = centered.T @ (centered * sample_weights[:, None])
            covariance_raw /= len(observations)
            covariance_raw = 0.5 * (covariance_raw + covariance_raw.T)
            try:
                _, likelihood_before_regularization = responsibilities(
                    observations, mean, covariance_raw, mixing_new, scales_new
                )
            except np.linalg.LinAlgError:
                likelihood_before_regularization = -np.inf

            jitter = covariance_epsilon * max(
                float(np.trace(covariance_raw) / observations.shape[1]), 1e-12
            )
            covariance_new = covariance_raw + jitter * np.eye(observations.shape[1])
            scales_new, covariance_new, normalized_floor_hit = _enforce_scale_constraint(
                mixing_new, scales_new, covariance_new, scale_floor
            )
            floor_hits += int(raw_floor_hit or normalized_floor_hit)
            gamma_new, new_likelihood = responsibilities(
                observations, mean, covariance_new, mixing_new, scales_new
            )
            if np.isfinite(previous_likelihood) and new_likelihood < previous_likelihood - 1e-8 * len(observations):
                decreases += 1

            covariance, mixing, scales = covariance_new, mixing_new, scales_new
            improvement = new_likelihood - previous_likelihood
            if np.isfinite(previous_likelihood) and abs(improvement) <= tolerance * (
                1.0 + abs(previous_likelihood)
            ):
                converged = True
                previous_likelihood = new_likelihood
                break
            previous_likelihood = new_likelihood
            gamma = gamma_new

        order = np.argsort(scales)
        scales = scales[order]
        mixing = mixing[order]
        condition_number = float(np.linalg.cond(covariance))
        candidate = GSMParameters(
            mean=mean.copy(),
            covariance=covariance.copy(),
            mixing_weights=mixing.copy(),
            scales=scales.copy(),
            log_likelihood=float(previous_likelihood),
            iterations=iteration,
            converged=converged,
            scale_floor_hits=floor_hits,
            likelihood_decreases=decreases,
            condition_number=condition_number,
            restart=restart,
        )
        histories.append(
            {
                "restart": restart,
                "log_likelihood": candidate.log_likelihood,
                "iterations": iteration,
                "converged": converged,
                "scale_floor_hits": floor_hits,
                "likelihood_decreases": decreases,
                "last_log_likelihood_before_regularization": float(
                    likelihood_before_regularization
                ),
                "condition_number": condition_number,
            }
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate

    if best is None:
        raise RuntimeError("No GSM initialization completed.")
    return best, histories


@torch.no_grad()
def _torch_responsibilities(
    centered: torch.Tensor,
    covariance: torch.Tensor,
    mixing_weights: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """GPU-friendly E-step with float64 parameters and bulk compute dtype data."""
    compute_dtype = centered.dtype
    sign, log_determinant = torch.linalg.slogdet(covariance)
    if float(sign.item()) <= 0.0:
        raise RuntimeError("GSM covariance is not positive definite.")
    # The matrix is only 9x9: invert it in float64, then apply the precision
    # to the large patch batch in the selected bulk-compute dtype.
    precision = torch.linalg.inv(covariance).to(dtype=compute_dtype)
    mahalanobis = torch.sum((centered @ precision) * centered, dim=1)
    dimension = centered.shape[1]
    normalizer = dimension * log(2.0 * pi) + float(log_determinant.item())
    mixing_compute = mixing_weights.to(dtype=compute_dtype)
    scales_compute = scales.to(dtype=compute_dtype)
    log_probabilities = torch.stack(
        [
            torch.log(mixing_compute[state])
            - 0.5
            * (
                normalizer
                + dimension * torch.log(scales_compute[state])
                + mahalanobis / scales_compute[state]
            )
            for state in range(2)
        ],
        dim=1,
    )
    log_normalizer = torch.logsumexp(log_probabilities, dim=1)
    gamma = torch.exp(log_probabilities - log_normalizer[:, None])
    likelihood = float(log_normalizer.sum(dtype=torch.float64).item())
    return gamma, likelihood, mahalanobis


@torch.no_grad()
def fit_two_state_gsm_torch(
    patches: np.ndarray,
    *,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float32,
    num_restarts: int = 5,
    max_iterations: int = 200,
    tolerance: float = 1e-7,
    covariance_epsilon: float = 1e-6,
    scale_floor: float = 1e-4,
    seed: int = 0,
) -> tuple[GSMParameters, list[dict[str, float | int | bool]]]:
    """Fit the GSM with CUDA bulk operations and float64 parameter updates."""
    if patches.ndim != 2 or patches.shape[1] != 9 or len(patches) < 10:
        raise ValueError("patches must have shape [N,9] with N at least ten.")
    if compute_dtype not in (torch.float32, torch.float64):
        raise TypeError("compute_dtype must be torch.float32 or torch.float64.")
    if min(num_restarts, max_iterations) < 1 or min(tolerance, covariance_epsilon, scale_floor) <= 0:
        raise ValueError("Invalid GSM optimization settings.")

    observations = torch.as_tensor(
        np.ascontiguousarray(patches), device=device, dtype=compute_dtype
    )
    mean = np.asarray(patches, dtype=np.float64).mean(axis=0)
    mean_tensor = torch.as_tensor(mean, device=device, dtype=compute_dtype)
    centered = observations - mean_tensor
    empirical_covariance = (centered.T @ centered).to(torch.float64) / len(centered)
    empirical_covariance = 0.5 * (empirical_covariance + empirical_covariance.T)
    average_variance = max(
        float(torch.trace(empirical_covariance).item() / centered.shape[1]), 1e-12
    )
    empirical_covariance.diagonal().add_(covariance_epsilon * average_variance)

    best: GSMParameters | None = None
    histories: list[dict[str, float | int | bool]] = []
    for restart in range(num_restarts):
        generator = np.random.default_rng(seed + restart)
        high_weight = float(generator.uniform(0.15, 0.45))
        mixing = np.asarray([1.0 - high_weight, high_weight], dtype=np.float64)
        low_scale = float(generator.uniform(0.15, 0.75))
        scales = np.asarray(
            [low_scale, (1.0 - mixing[0] * low_scale) / mixing[1]],
            dtype=np.float64,
        )
        covariance_tensor = empirical_covariance.clone()
        mixing_tensor = torch.as_tensor(mixing, device=device, dtype=torch.float64)
        scales_tensor = torch.as_tensor(scales, device=device, dtype=torch.float64)
        gamma, previous_likelihood, mahalanobis = _torch_responsibilities(
            centered, covariance_tensor, mixing_tensor, scales_tensor
        )
        floor_hits = 0
        decreases = 0
        converged = False
        likelihood_before_regularization = -np.inf

        for iteration in range(1, max_iterations + 1):
            counts_tensor = gamma.sum(dim=0, dtype=torch.float64)
            mixing_new_tensor = torch.clamp(counts_tensor / len(centered), min=1e-12)
            mixing_new_tensor /= mixing_new_tensor.sum()
            scales_new_tensor = torch.sum(
                gamma * mahalanobis[:, None], dim=0, dtype=torch.float64
            )
            scales_new_tensor /= centered.shape[1] * counts_tensor
            raw_floor_hit = bool(torch.any(scales_new_tensor < scale_floor).item())
            scales_new_tensor.clamp_(min=scale_floor)

            sample_weights = torch.sum(
                gamma / scales_new_tensor.to(compute_dtype)[None, :], dim=1
            )
            covariance_raw = (
                centered.T @ (centered * sample_weights[:, None])
            ).to(torch.float64) / len(centered)
            covariance_raw = 0.5 * (covariance_raw + covariance_raw.T)
            try:
                _, likelihood_before_regularization, _ = _torch_responsibilities(
                    centered,
                    covariance_raw,
                    mixing_new_tensor,
                    scales_new_tensor,
                )
            except RuntimeError:
                likelihood_before_regularization = -np.inf

            jitter = covariance_epsilon * max(
                float(torch.trace(covariance_raw).item() / centered.shape[1]), 1e-12
            )
            covariance_new = covariance_raw.clone()
            covariance_new.diagonal().add_(jitter)
            scales_numpy, covariance_numpy, normalized_floor_hit = _enforce_scale_constraint(
                mixing_new_tensor.cpu().numpy(),
                scales_new_tensor.cpu().numpy(),
                covariance_new.cpu().numpy(),
                scale_floor,
            )
            floor_hits += int(raw_floor_hit or normalized_floor_hit)
            covariance_new = torch.as_tensor(
                covariance_numpy, device=device, dtype=torch.float64
            )
            scales_new_tensor = torch.as_tensor(
                scales_numpy, device=device, dtype=torch.float64
            )
            gamma_new, new_likelihood, mahalanobis_new = _torch_responsibilities(
                centered,
                covariance_new,
                mixing_new_tensor,
                scales_new_tensor,
            )
            if new_likelihood < previous_likelihood - 1e-8 * len(centered):
                decreases += 1
            improvement = new_likelihood - previous_likelihood
            covariance_tensor = covariance_new
            mixing_tensor = mixing_new_tensor
            scales_tensor = scales_new_tensor
            if abs(improvement) <= tolerance * (1.0 + abs(previous_likelihood)):
                converged = True
                previous_likelihood = new_likelihood
                break
            previous_likelihood = new_likelihood
            gamma = gamma_new
            mahalanobis = mahalanobis_new

        covariance = covariance_tensor.cpu().numpy()
        mixing = mixing_tensor.cpu().numpy()
        scales = scales_tensor.cpu().numpy()
        order = np.argsort(scales)
        scales = scales[order]
        mixing = mixing[order]
        candidate = GSMParameters(
            mean=mean.copy(),
            covariance=covariance.copy(),
            mixing_weights=mixing.copy(),
            scales=scales.copy(),
            log_likelihood=float(previous_likelihood),
            iterations=iteration,
            converged=converged,
            scale_floor_hits=floor_hits,
            likelihood_decreases=decreases,
            condition_number=float(np.linalg.cond(covariance)),
            restart=restart,
        )
        histories.append(
            {
                "restart": restart,
                "log_likelihood": candidate.log_likelihood,
                "iterations": iteration,
                "converged": converged,
                "scale_floor_hits": floor_hits,
                "likelihood_decreases": decreases,
                "last_log_likelihood_before_regularization": float(
                    likelihood_before_regularization
                ),
                "condition_number": candidate.condition_number,
            }
        )
        if best is None or candidate.log_likelihood > best.log_likelihood:
            best = candidate

    del observations, centered, empirical_covariance
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("No GSM initialization completed.")
    return best, histories


def save_gsm(path: str, parameters: GSMParameters) -> None:
    np.savez_compressed(
        path,
        mean=parameters.mean,
        covariance=parameters.covariance,
        mixing_weights=parameters.mixing_weights,
        scales=parameters.scales,
        log_likelihood=parameters.log_likelihood,
        iterations=parameters.iterations,
        converged=parameters.converged,
        scale_floor_hits=parameters.scale_floor_hits,
        likelihood_decreases=parameters.likelihood_decreases,
        condition_number=parameters.condition_number,
        restart=parameters.restart,
    )


def load_gsm(path: str) -> GSMParameters:
    with np.load(path) as values:
        return GSMParameters(
            mean=values["mean"],
            covariance=values["covariance"],
            mixing_weights=values["mixing_weights"],
            scales=values["scales"],
            log_likelihood=float(values["log_likelihood"]),
            iterations=int(values["iterations"]),
            converged=bool(values["converged"]),
            scale_floor_hits=int(values["scale_floor_hits"]),
            likelihood_decreases=int(values["likelihood_decreases"]),
            condition_number=float(values["condition_number"]),
            restart=int(values["restart"]),
        )
