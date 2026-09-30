"""Generate samples with the unconstrained kappa spectral closed-form.

The probability path is defined in the PCA basis by

    z_t^i = (1 - beta_i(t)) eps_i + beta_i(t) z_1^i,

where beta_i depends on the PCA eigenvalue lambda_i and on a hierarchy
parameter kappa. For kappa=1, the path is the standard Flow Matching path.

For each kappa, this script integrates the empirical closed-form velocity

    f*(x,t) = E[u_cond(x_t, x_1, t) | x_t = x]

using uniform-step Euler sampling from the same initial Gaussian noise. The
output grid is intended to reveal whether the unconstrained closed-form
memorizes training samples for different spectral hierarchies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
EXPERIMENTS_ROOT = PROJECT_ROOT / "fm_comparisons"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (  # noqa: E402
    dataset_output_slug,
    load_dataset,
    load_experiment_config,
    load_or_compute_wiener_svd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument(
        "--trajectory-frames",
        type=int,
        default=6,
        help="Number of uniformly spaced trajectory states to display, including t=0 and t=1.",
    )
    parser.add_argument(
        "--trajectory-sample-index",
        type=int,
        default=0,
        help="Generated sample whose complete trajectory is displayed.",
    )
    parser.add_argument("--num-xi", type=int, default=None)
    parser.add_argument("--kappas", default="-0.5,0,0.5,1,1.5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--neighbor-space",
        choices=("pixel", "inception", "posterior"),
        default="pixel",
        help="Space used to retrieve nearest dataset neighbors.",
    )
    parser.add_argument(
        "--neighbor-t",
        type=float,
        default=0.98,
        help="Late time used for posterior-neighbor retrieval.",
    )
    parser.add_argument(
        "--inception-cache",
        type=Path,
        default=None,
        help="Optional cache path for candidate Inception features.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def parse_kappas(value: str) -> list[float]:
    """Parse comma-separated kappa values."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def euler_boundary_step(time: float, num_steps: int) -> int:
    """Return the Euler boundary index for time, rejecting off-grid probes."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not (0.0 < time < 1.0):
        raise ValueError("The probe time must be strictly between 0 and 1.")
    position = time * num_steps
    step = int(round(position))
    if abs(position - step) > 1e-8:
        raise ValueError(
            f"probe_t={time:g} is not an Euler boundary for "
            f"num_steps={num_steps}; choose k/num_steps exactly."
        )
    return step


def spectral_stds(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return lambda_i from covariance eigenvalues lambda_i^2."""
    return torch.sqrt(torch.clamp(eigenvalues, min=eps))


def geometric_mean(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return exp(mean(log(values))) in a numerically stable way."""
    return torch.exp(torch.log(torch.clamp(values, min=eps)).mean())


@dataclass(frozen=True)
class SpectralCandidateBank:
    """Training images and their PCA coordinates, cached once on the device."""

    images_flat: torch.Tensor  # [K, N]
    coefficients: torch.Tensor  # [K, N]
    chunk_size: int

    def chunks(self):
        for start in range(0, self.images_flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.images_flat.shape[0])
            yield self.images_flat[start:end], self.coefficients[start:end]

    @property
    def storage_megabytes(self) -> float:
        num_bytes = (
            self.images_flat.numel() * self.images_flat.element_size()
            + self.coefficients.numel() * self.coefficients.element_size()
        )
        return num_bytes / (1024.0**2)


@torch.no_grad()
def build_spectral_candidate_bank(
    candidate_dataset,
    U: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> SpectralCandidateBank:
    """Project every candidate once instead of repeating the PCA at every step."""
    image_chunks: list[torch.Tensor] = []
    coefficient_chunks: list[torch.Tensor] = []
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images_flat = images.to(device=device, dtype=dtype).flatten(start_dim=1)
        image_chunks.append(images_flat)
        coefficient_chunks.append(images_flat @ U)

    if not image_chunks:
        raise RuntimeError("The candidate dataset is empty.")
    return SpectralCandidateBank(
        images_flat=torch.cat(image_chunks, dim=0),
        coefficients=torch.cat(coefficient_chunks, dim=0),
        chunk_size=int(candidate_dataset.dataloader.batch_size),
    )


def kappa_schedule(
    t: float,
    kappa: float,
    lambdas: torch.Tensor,
    lambda_star: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return beta_i(t), alpha_i(t), and beta_dot_i(t) for one kappa.

    Shapes:
        lambdas: [N]
        beta: [N]
        alpha: [N]
        beta_dot: [N]
    """
    t_tensor = torch.tensor(t, device=lambdas.device, dtype=lambdas.dtype)
    ratio = torch.clamp(lambdas / lambda_star, min=1e-12)
    r = ratio ** (kappa - 1.0)  # [N]
    denom = 1.0 - t_tensor + t_tensor * r  # [N]
    beta = t_tensor * r / denom
    alpha = (1.0 - t_tensor) / denom
    beta_dot = r / (denom * denom)
    return beta, alpha, beta_dot


class GlobalStreamingSoftmax:
    """Streaming softmax average for logits [B, K] and values [K, N]."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.max_logits: torch.Tensor | None = None  # [B]
        self.sum_weights: torch.Tensor | None = None  # [B]
        self.sum_weighted: torch.Tensor | None = None  # [B, N]

    def add(self, values: torch.Tensor, logits: torch.Tensor) -> None:
        """Add a candidate chunk.

        Args:
            values: candidate targets, shape [K, N].
            logits: candidate log weights, shape [B, K].
        """
        values = values.to(device=self.device, dtype=self.dtype)
        logits = logits.to(device=self.device, dtype=self.dtype)

        batch_max = logits.max(dim=1).values  # [B]
        if self.max_logits is None:
            new_max = batch_max
            old_scale = None
        else:
            new_max = torch.maximum(self.max_logits, batch_max)
            old_scale = torch.exp(self.max_logits - new_max)  # [B]

        weights = torch.exp(logits - new_max.unsqueeze(1))  # [B, K]
        chunk_sum_weights = weights.sum(dim=1)  # [B]
        chunk_sum_weighted = weights @ values  # [B, N]

        if self.sum_weighted is None or self.sum_weights is None:
            self.sum_weights = chunk_sum_weights
            self.sum_weighted = chunk_sum_weighted
        else:
            assert old_scale is not None
            self.sum_weights = self.sum_weights * old_scale + chunk_sum_weights
            self.sum_weighted = (
                self.sum_weighted * old_scale.unsqueeze(1) + chunk_sum_weighted
            )

        self.max_logits = new_max

    def get_average(self) -> torch.Tensor:
        if self.sum_weighted is None or self.sum_weights is None:
            raise RuntimeError("No candidates were added to the streaming softmax.")
        return self.sum_weighted / self.sum_weights.clamp_min(1e-12).unsqueeze(1)


@torch.no_grad()
def predict_unconstrained_kappa_velocity(
    xt: torch.Tensor,
    t: float,
    kappa: float,
    candidate_bank: SpectralCandidateBank,
    U: torch.Tensor,
    lambdas: torch.Tensor,
    lambda_star: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Predict the unconstrained empirical closed-form velocity.

    The computation is done in PCA coordinates:

        z_x = F x_t
        z_i = F x_i

    The posterior weights use the diagonal Gaussian covariance alpha_i(t)^2,
    and the values are the conditional velocity targets

        beta_dot_i(t) / alpha_i(t) * (z_i - z_x).
    """
    xt_shape = xt.shape
    xt_flat = xt.flatten(start_dim=1)  # [B, N]
    z_xt = xt_flat @ U  # [B, N]

    beta, alpha, beta_dot = kappa_schedule(t, kappa, lambdas, lambda_star)
    precision = 1.0 / alpha.square().clamp_min(1e-12)  # [N]
    velocity_coeff = beta_dot / alpha.clamp_min(1e-12)  # [N]

    softmax = GlobalStreamingSoftmax(device=xt.device, dtype=xt.dtype)
    for _candidates, z_candidates in candidate_bank.chunks():
        residual = z_xt.unsqueeze(1) - beta.unsqueeze(0).unsqueeze(0) * z_candidates.unsqueeze(0)
        distances = (residual.square() * precision.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        logits = -0.5 * distances / temperature  # [B, K]

        values = z_candidates * velocity_coeff.unsqueeze(0)  # [K, N]
        softmax.add(values, logits)

    expected_candidate_part = softmax.get_average()  # [B, N]
    velocity_spectral = expected_candidate_part - z_xt * velocity_coeff.unsqueeze(0)
    velocity_flat = velocity_spectral @ U.T
    return velocity_flat.view(xt_shape)


@torch.no_grad()
def euler_generate(
    initial_noise: torch.Tensor,
    kappas: list[float],
    candidate_bank: SpectralCandidateBank,
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    num_steps: int,
    temperature: float,
    trajectory_frames: int,
    trajectory_sample_index: int,
    probe_t: float | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    list[float],
    dict[str, torch.Tensor] | None,
]:
    """Generate samples and retain one trajectory per kappa.

    The velocity is evaluated at Euler interval midpoints. Trajectory states
    are recorded at uniformly spaced interval boundaries, including t=0 and
    t=1, so the labels correspond to the actual state times.

    Returns:
        samples: final sample batch for every kappa.
        trajectories: one selected sample per kappa, shape [F, C, H, W].
        trajectory_times: boundary times associated with the F saved states.
        probe_states: complete sample batches at probe_t, when requested.
    """
    if num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if not (2 <= trajectory_frames <= num_steps + 1):
        raise ValueError(
            "--trajectory-frames must be between 2 and num_steps + 1."
        )
    if not (0 <= trajectory_sample_index < initial_noise.shape[0]):
        raise ValueError(
            "--trajectory-sample-index must refer to one of the generated samples."
        )
    probe_step: int | None = None
    if probe_t is not None:
        probe_step = euler_boundary_step(probe_t, num_steps)

    lambdas = spectral_stds(eigenvalues).to(device=initial_noise.device, dtype=initial_noise.dtype)
    lambda_star = geometric_mean(lambdas)
    samples = {f"kappa={kappa:g}": initial_noise.clone() for kappa in kappas}
    step_size = 1.0 / num_steps
    snapshot_steps = [
        round(frame * num_steps / (trajectory_frames - 1))
        for frame in range(trajectory_frames)
    ]
    snapshot_step_set = set(snapshot_steps)
    trajectory_times = [step / num_steps for step in snapshot_steps]
    trajectory_lists = {
        key: [sample[trajectory_sample_index].detach().cpu().clone()]
        for key, sample in samples.items()
    }
    probe_states: dict[str, torch.Tensor] | None = None

    print(f"lambda_star: {lambda_star.item():.6g}")
    print(f"trajectory_times: {trajectory_times}")
    for step in range(num_steps):
        t = (step + 0.5) / num_steps
        print(f"Euler step {step + 1:03d}/{num_steps} | t={t:.4f}")
        for kappa in kappas:
            key = f"kappa={kappa:g}"
            velocity = predict_unconstrained_kappa_velocity(
                xt=samples[key],
                t=t,
                kappa=kappa,
                candidate_bank=candidate_bank,
                U=U,
                lambdas=lambdas,
                lambda_star=lambda_star,
                temperature=temperature,
            )
            samples[key] = samples[key] + step_size * velocity

        completed_steps = step + 1
        if probe_step is not None and completed_steps == probe_step:
            probe_states = {
                key: sample.detach().clone() for key, sample in samples.items()
            }
        if completed_steps in snapshot_step_set:
            for key in samples:
                trajectory_lists[key].append(
                    samples[key][trajectory_sample_index].detach().cpu().clone()
                )

    trajectories = {
        key: torch.stack(states, dim=0)
        for key, states in trajectory_lists.items()
    }
    if probe_step is not None and probe_states is None:
        raise RuntimeError("The requested posterior probe state was not recorded.")
    return samples, trajectories, trajectory_times, probe_states


def save_trajectory_grid(
    trajectories: dict[str, torch.Tensor],
    trajectory_times: list[float],
    postprocess,
    out_path: Path,
) -> None:
    """Plot one generated trajectory per kappa.

    Rows correspond to kappa estimators and columns to common state times.
    """
    keys = list(trajectories.keys())
    fig, axes = plt.subplots(
        len(keys),
        len(trajectory_times),
        figsize=(2.1 * len(trajectory_times), 2.1 * len(keys)),
        squeeze=False,
    )

    for row, key in enumerate(keys):
        images = postprocess(trajectories[key]).permute(0, 2, 3, 1).numpy()
        for col, time in enumerate(trajectory_times):
            ax = axes[row, col]
            ax.imshow(images[col], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={time:.2f}", fontsize=13)
            if col == 0:
                ax.set_ylabel(key, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_final_grid(samples: dict[str, torch.Tensor], postprocess, out_path: Path) -> None:
    """Save a grid with rows=samples and columns=kappa values."""
    num_samples = next(iter(samples.values())).shape[0]
    keys = list(samples.keys())
    fig, axes = plt.subplots(
        num_samples,
        len(keys),
        figsize=(2.1 * len(keys), 2.1 * num_samples),
        squeeze=False,
    )

    for col, key in enumerate(keys):
        axes[0, col].set_title(key, fontsize=15)
        images = postprocess(samples[key].detach().cpu())
        images = images.permute(0, 2, 3, 1).numpy()
        for row in range(num_samples):
            ax = axes[row, col]
            ax.imshow(images[row], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


@torch.no_grad()
def find_nearest_dataset_neighbors(
    samples: dict[str, torch.Tensor],
    candidate_dataset,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Find the nearest dataset image for each generated sample.

    Distance is squared Euclidean distance in the normalized tensor space used by
    the model/dataset, not in postprocessed RGB space.

    Returns:
        neighbors: generated key -> nearest images, shape [B, C, H, W]
        distances: generated key -> minimum squared distances, shape [B]
        indices: generated key -> dataset indices within the candidate dataloader order, shape [B]
    """
    keys = list(samples.keys())
    first = samples[keys[0]]
    device = first.device
    dtype = first.dtype
    image_shape = first.shape[1:]
    flat_samples = torch.cat(
        [samples[key].flatten(start_dim=1) for key in keys],
        dim=0,
    )  # [num_keys * B, N]
    num_queries = flat_samples.shape[0]

    best_distances = torch.full((num_queries,), float("inf"), device=device, dtype=dtype)
    best_indices = torch.full((num_queries,), -1, device=device, dtype=torch.long)
    best_neighbors = torch.empty_like(flat_samples)

    dataset_offset = 0
    query_norms = flat_samples.square().sum(dim=1, keepdim=True)  # [Q, 1]
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        candidates = images.to(device=device, dtype=dtype).flatten(start_dim=1)  # [K, N]
        candidate_norms = candidates.square().sum(dim=1).unsqueeze(0)  # [1, K]
        distances = query_norms + candidate_norms - 2.0 * (flat_samples @ candidates.T)
        chunk_distances, chunk_indices = distances.min(dim=1)  # [Q]

        improved = chunk_distances < best_distances
        if improved.any():
            best_distances[improved] = chunk_distances[improved]
            absolute_indices = chunk_indices[improved] + dataset_offset
            best_indices[improved] = absolute_indices
            best_neighbors[improved] = candidates[chunk_indices[improved]]

        dataset_offset += candidates.shape[0]

    neighbors: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    indices: dict[str, torch.Tensor] = {}
    batch_size = first.shape[0]
    for key_index, key in enumerate(keys):
        start = key_index * batch_size
        end = start + batch_size
        neighbors[key] = best_neighbors[start:end].view(batch_size, *image_shape)
        distances[key] = best_distances[start:end]
        indices[key] = best_indices[start:end]

    return neighbors, distances, indices


@torch.no_grad()
def find_posterior_dataset_neighbors(
    samples: dict[str, torch.Tensor],
    kappas: list[float],
    candidate_bank: SpectralCandidateBank,
    U: torch.Tensor,
    lambdas: torch.Tensor,
    lambda_star: torch.Tensor,
    t_probe: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Find nearest neighbors in the kappa posterior metric at a late time.

    For each kappa and generated image x, this computes

        argmin_i sum_j (z_x^j - beta_j(t) z_i^j)^2 / alpha_j(t)^2,

    which is exactly the distance used by the closed-form posterior weights.
    For t close to 1, this approaches the anisotropic d_kappa distance from the
    notes, up to the common diverging factor (1-t)^(-2).
    """
    if not (0.0 < t_probe < 1.0):
        raise ValueError("--neighbor-t must be strictly between 0 and 1.")

    first = next(iter(samples.values()))
    device = first.device
    dtype = first.dtype
    image_shape = first.shape[1:]
    neighbors: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    indices: dict[str, torch.Tensor] = {}

    for kappa in kappas:
        key = f"kappa={kappa:g}"
        sample_flat = samples[key].flatten(start_dim=1)  # [B, N]
        z_samples = sample_flat @ U  # [B, N]
        beta, alpha, _beta_dot = kappa_schedule(t_probe, kappa, lambdas, lambda_star)
        precision = 1.0 / alpha.square().clamp_min(1e-12)  # [N]

        best_distances = torch.full(
            (z_samples.shape[0],),
            float("inf"),
            device=device,
            dtype=dtype,
        )
        best_indices = torch.full((z_samples.shape[0],), -1, device=device, dtype=torch.long)
        best_neighbors = torch.empty_like(sample_flat)

        dataset_offset = 0
        for candidates, z_candidates in candidate_bank.chunks():
            residual = z_samples.unsqueeze(1) - beta.view(1, 1, -1) * z_candidates.unsqueeze(0)
            chunk_distances = (residual.square() * precision.view(1, 1, -1)).sum(dim=2)
            min_distances, min_indices = chunk_distances.min(dim=1)

            improved = min_distances < best_distances
            if improved.any():
                best_distances[improved] = min_distances[improved]
                best_indices[improved] = min_indices[improved] + dataset_offset
                best_neighbors[improved] = candidates[min_indices[improved]]

            dataset_offset += candidates.shape[0]

        neighbors[key] = best_neighbors.view(samples[key].shape[0], *image_shape)
        distances[key] = best_distances
        indices[key] = best_indices

    return neighbors, distances, indices


def save_nearest_neighbor_grid(
    samples: dict[str, torch.Tensor],
    neighbors: dict[str, torch.Tensor],
    postprocess,
    out_path: Path,
    neighbor_space: str,
    neighbor_t: float | None = None,
) -> None:
    """Save generated samples next to their nearest training neighbors."""
    keys = list(samples.keys())
    num_samples = next(iter(samples.values())).shape[0]
    num_cols = 2 * len(keys)
    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(1.9 * num_cols, 1.9 * num_samples),
        squeeze=False,
    )

    for key_index, key in enumerate(keys):
        gen_images = postprocess(samples[key].detach().cpu()).permute(0, 2, 3, 1).numpy()
        nn_images = postprocess(neighbors[key].detach().cpu()).permute(0, 2, 3, 1).numpy()
        gen_col = 2 * key_index
        nn_col = gen_col + 1
        axes[0, gen_col].set_title(f"{key}\ngen", fontsize=13)
        neighbor_label = neighbor_space
        if neighbor_t is not None:
            neighbor_label = f"{neighbor_space} at t={neighbor_t:g}"
        axes[0, nn_col].set_title(f"nearest\n{neighbor_label}", fontsize=13)
        for row in range(num_samples):
            axes[row, gen_col].imshow(gen_images[row], interpolation="nearest")
            axes[row, nn_col].imshow(nn_images[row], interpolation="nearest")
            axes[row, gen_col].set_xticks([])
            axes[row, gen_col].set_yticks([])
            axes[row, nn_col].set_xticks([])
            axes[row, nn_col].set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_nearest_neighbor_metrics(
    distances: dict[str, torch.Tensor],
    indices: dict[str, torch.Tensor],
    normalizer: int,
    neighbor_space: str,
    out_path: Path,
    neighbor_t: float | None = None,
) -> None:
    """Save nearest-neighbor distances and candidate dataset indices."""
    rows: list[dict[str, Any]] = []
    for key in distances:
        key_distances = distances[key].detach().cpu()
        key_indices = indices[key].detach().cpu()
        for sample_index in range(key_distances.shape[0]):
            rows.append(
                {
                    "estimator": key,
                    "sample_index": sample_index,
                    "nearest_dataset_index": int(key_indices[sample_index].item()),
                    "neighbor_space": neighbor_space,
                    "neighbor_t": neighbor_t,
                    "squared_l2_distance": float(key_distances[sample_index].item()),
                    "rmse": float(
                        torch.sqrt(key_distances[sample_index] / normalizer).item()
                    ),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def load_inception_feature_model(device: torch.device):
    """Load torchvision InceptionV3 as a 2048-dimensional feature extractor."""
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Inception neighbor search requires torchvision, but torchvision "
            "is not installed in this Python environment. Install it in the "
            "environment used to run this script, or keep --neighbor-space pixel."
        ) from exc

    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """Resize [0,1] images to Inception input size and apply ImageNet stats."""
    import torch.nn.functional as F

    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    return (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


@torch.no_grad()
def extract_inception_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    postprocess,
) -> torch.Tensor:
    """Extract L2-normalized Inception features from normalized dataset tensors."""
    import torch.nn.functional as F

    rgb_images = postprocess(images.detach().cpu()).to(device=images.device, dtype=images.dtype)
    features = model(imagenet_normalize(rgb_images))
    if features.ndim == 4:
        features = features.flatten(start_dim=1)
    return F.normalize(features, dim=1)


@torch.no_grad()
def compute_or_load_candidate_inception_features(
    candidate_dataset,
    model: torch.nn.Module,
    postprocess,
    device: torch.device,
    cache_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate Inception features and matching images."""
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["features"], payload["images"]

    feature_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []
    for batch in candidate_dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device=device)
        feature_chunks.append(
            extract_inception_features(images, model=model, postprocess=postprocess).cpu()
        )
        image_chunks.append(images.cpu())

    features = torch.cat(feature_chunks, dim=0)
    images = torch.cat(image_chunks, dim=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "images": images}, cache_path)
    return features, images


@torch.no_grad()
def find_inception_nearest_dataset_neighbors(
    samples: dict[str, torch.Tensor],
    candidate_dataset,
    postprocess,
    cache_path: Path | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Find nearest dataset neighbors in normalized Inception feature space."""
    keys = list(samples.keys())
    first = samples[keys[0]]
    device = first.device
    model = load_inception_feature_model(device)
    candidate_features_cpu, candidate_images_cpu = compute_or_load_candidate_inception_features(
        candidate_dataset=candidate_dataset,
        model=model,
        postprocess=postprocess,
        device=device,
        cache_path=cache_path,
    )

    query_images = torch.cat([samples[key] for key in keys], dim=0)
    query_features = extract_inception_features(
        query_images,
        model=model,
        postprocess=postprocess,
    )

    best_distances = torch.full(
        (query_features.shape[0],),
        float("inf"),
        device=device,
        dtype=query_features.dtype,
    )
    best_indices = torch.full((query_features.shape[0],), -1, device=device, dtype=torch.long)
    chunk_size = 4096
    for start in range(0, candidate_features_cpu.shape[0], chunk_size):
        end = min(start + chunk_size, candidate_features_cpu.shape[0])
        candidate_features = candidate_features_cpu[start:end].to(device=device)
        distances = torch.cdist(query_features, candidate_features).square()
        chunk_distances, chunk_indices = distances.min(dim=1)
        improved = chunk_distances < best_distances
        if improved.any():
            best_distances[improved] = chunk_distances[improved]
            best_indices[improved] = chunk_indices[improved] + start

    neighbors: dict[str, torch.Tensor] = {}
    distances: dict[str, torch.Tensor] = {}
    indices: dict[str, torch.Tensor] = {}
    batch_size = first.shape[0]
    for key_index, key in enumerate(keys):
        start = key_index * batch_size
        end = start + batch_size
        selected = best_indices[start:end].cpu()
        neighbors[key] = candidate_images_cpu[selected].to(device=device, dtype=first.dtype)
        distances[key] = best_distances[start:end]
        indices[key] = best_indices[start:end]

    return neighbors, distances, indices


def save_config(
    args: argparse.Namespace,
    out_path: Path,
    *,
    resolved_num_candidates: int,
    spectral_cache_megabytes: float,
) -> None:
    """Save generation arguments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload["resolved_num_candidates"] = resolved_num_candidates
    payload["spectral_cache_megabytes"] = spectral_cache_megabytes
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kappas = parse_kappas(args.kappas)
    out_dir = args.out_dir or Path("spectral_paths/results") / (
        f"kappa_unconstrained_generation_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_experiment_config(
        args.dataset,
        args.batch_size,
        subset_size=None,
        num_workers=args.num_workers,
    )
    candidate_cfg = load_experiment_config(
        args.dataset,
        args.batch_size,
        subset_size=args.num_xi,
        num_workers=args.num_workers,
    )
    dataset = load_dataset(cfg)
    candidate_dataset = load_dataset(candidate_cfg)
    U, eigenvalues, _Vh, _mean = load_or_compute_wiener_svd(candidate_dataset, device)
    candidate_bank = build_spectral_candidate_bank(
        candidate_dataset=candidate_dataset,
        U=U,
        device=device,
        dtype=U.dtype,
    )

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"kappas: {kappas}")
    print(f"num_samples: {args.num_samples}")
    print(f"num_steps: {args.num_steps}")
    print(f"trajectory_frames: {args.trajectory_frames}")
    print(f"trajectory_sample_index: {args.trajectory_sample_index}")
    print(f"num_xi: {args.num_xi}")
    print(f"temperature: {args.temperature}")
    print(
        f"spectral candidate cache: {candidate_bank.images_flat.shape[0]} images, "
        f"{candidate_bank.storage_megabytes:.1f} MiB"
    )
    print(f"neighbor_space: {args.neighbor_space}")
    print(f"out_dir: {out_dir}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        device=device,
        generator=generator,
    )

    probe_t = args.neighbor_t if args.neighbor_space == "posterior" else None
    samples, trajectories, trajectory_times, probe_states = euler_generate(
        initial_noise=initial_noise,
        kappas=kappas,
        candidate_bank=candidate_bank,
        U=U,
        eigenvalues=eigenvalues,
        num_steps=args.num_steps,
        temperature=args.temperature,
        trajectory_frames=args.trajectory_frames,
        trajectory_sample_index=args.trajectory_sample_index,
        probe_t=probe_t,
    )

    save_final_grid(samples, dataset.postprocess, out_dir / "final_samples.png")
    save_trajectory_grid(
        trajectories=trajectories,
        trajectory_times=trajectory_times,
        postprocess=dataset.postprocess,
        out_path=out_dir / "trajectories.png",
    )
    torch.save(
        {
            "times": trajectory_times,
            "sample_index": args.trajectory_sample_index,
            "trajectories": trajectories,
        },
        out_dir / "trajectories.pt",
    )

    if args.neighbor_space == "pixel":
        neighbors, distances, indices = find_nearest_dataset_neighbors(samples, candidate_dataset)
        metric_normalizer = initial_noise[0].numel()
        neighbor_t = None
    elif args.neighbor_space == "inception":
        cache_path = args.inception_cache
        if cache_path is None:
            suffix = "full" if args.num_xi is None else f"num_xi_{args.num_xi}"
            cache_path = (
                Path("spectral_paths/cache")
                / f"inception_features_{dataset_output_slug(args.dataset)}_{suffix}.pt"
            )
        print(f"inception_cache: {cache_path}")
        neighbors, distances, indices = find_inception_nearest_dataset_neighbors(
            samples=samples,
            candidate_dataset=candidate_dataset,
            postprocess=dataset.postprocess,
            cache_path=cache_path,
        )
        metric_normalizer = 2048
        neighbor_t = None
    else:
        if probe_states is None:
            raise RuntimeError("Posterior neighbor search requires saved probe states.")
        lambdas = spectral_stds(eigenvalues).to(device=device, dtype=initial_noise.dtype)
        lambda_star = geometric_mean(lambdas)
        neighbors, distances, indices = find_posterior_dataset_neighbors(
            samples=probe_states,
            kappas=kappas,
            candidate_bank=candidate_bank,
            U=U,
            lambdas=lambdas,
            lambda_star=lambda_star,
            t_probe=args.neighbor_t,
        )
        metric_normalizer = initial_noise[0].numel()
        neighbor_t = args.neighbor_t
        torch.save(
            {
                "time": neighbor_t,
                "states": {
                    key: value.detach().cpu() for key, value in probe_states.items()
                },
            },
            out_dir / "posterior_probe_states.pt",
        )

    save_nearest_neighbor_grid(
        samples=samples,
        neighbors=neighbors,
        postprocess=dataset.postprocess,
        out_path=out_dir / "nearest_neighbors.png",
        neighbor_space=args.neighbor_space,
        neighbor_t=neighbor_t,
    )
    save_nearest_neighbor_metrics(
        distances=distances,
        indices=indices,
        normalizer=metric_normalizer,
        neighbor_space=args.neighbor_space,
        out_path=out_dir / "nearest_neighbors.json",
        neighbor_t=neighbor_t,
    )

    save_config(
        args,
        out_dir / "generation_config.json",
        resolved_num_candidates=candidate_bank.images_flat.shape[0],
        spectral_cache_megabytes=candidate_bank.storage_megabytes,
    )
    print(f"saved plot to {out_dir / 'final_samples.png'}")
    print(f"saved trajectory plot to {out_dir / 'trajectories.png'}")
    print(f"saved nearest-neighbor plot to {out_dir / 'nearest_neighbors.png'}")


if __name__ == "__main__":
    main()
