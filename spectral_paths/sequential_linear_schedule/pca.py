"""Population-PCA preparation and caching for the spectral experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from local_diffusion.data import DatasetBundle


@dataclass
class PCABasis:
    """Full orthogonal PCA basis with non-centered coordinate conversions."""

    mean: torch.Tensor
    U: torch.Tensor
    lambda_sq: torch.Tensor
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int:
        return self.mean.numel()

    @property
    def lambda_std(self) -> torch.Tensor:
        return torch.sqrt(self.lambda_sq.clamp_min(0.0))

    @property
    def mu_z(self) -> torch.Tensor:
        return self.mean @ self.U

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "PCABasis":
        """Return this basis on a selected device and dtype."""
        return PCABasis(
            mean=self.mean.to(device=device, dtype=dtype),
            U=self.U.to(device=device, dtype=dtype),
            lambda_sq=self.lambda_sq.to(device=device, dtype=dtype),
            metadata=self.metadata,
        )

    def to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Map non-centered pixel row vectors ``x`` to ``z = U^T x``."""
        if x.shape[-1] != self.dimension:
            raise ValueError("The last pixel dimension does not match the PCA basis.")
        return x @ self.U

    def to_pixel(self, z: torch.Tensor) -> torch.Tensor:
        """Map spectral row vectors back to pixels with ``x = U z``."""
        if z.shape[-1] != self.dimension:
            raise ValueError("The last spectral dimension does not match the PCA basis.")
        return z @ self.U.T


def load_pca_cache(
    cache_dir: Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> PCABasis:
    """Load a population-PCA cache produced by this experiment."""
    metadata_path = cache_dir / "metadata.json"
    required = [
        cache_dir / "mean.pt",
        cache_dir / "U.pt",
        cache_dir / "lambda_sq.pt",
        metadata_path,
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete PCA cache at {cache_dir}.")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    mean = torch.load(required[0], map_location=device, weights_only=True)
    U = torch.load(required[1], map_location=device, weights_only=True)
    lambda_sq = torch.load(required[2], map_location=device, weights_only=True)
    if dtype is not None:
        mean = mean.to(dtype=dtype)
        U = U.to(dtype=dtype)
        lambda_sq = lambda_sq.to(dtype=dtype)
    return PCABasis(mean=mean, U=U, lambda_sq=lambda_sq, metadata=metadata)


def save_pca_cache(basis: PCABasis, cache_dir: Path) -> None:
    """Persist tensors and provenance metadata for a PCA basis."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(basis.mean.detach().cpu().contiguous(), cache_dir / "mean.pt")
    torch.save(basis.U.detach().cpu().contiguous(), cache_dir / "U.pt")
    torch.save(
        basis.lambda_sq.detach().cpu().contiguous(),
        cache_dir / "lambda_sq.pt",
    )
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(basis.metadata, handle, indent=2, sort_keys=True)


@torch.no_grad()
def prepare_population_pca(
    dataset: DatasetBundle,
    cache_dir: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    force_recompute: bool = False,
    source_wiener_dir: Path | None = None,
    reuse_sample_covariance_cache: bool = True,
) -> PCABasis:
    """Load, convert, or compute the centered population PCA.

    Artem's Wiener cache was computed with denominator ``N - 1``. For the
    full CIFAR-10 training split, its eigenvectors and mean are reused while
    eigenvalues are multiplied by ``(N - 1) / N``. The converted tensors are
    written to a separate cache with explicit provenance.
    """
    expected_dimension = dataset.in_channels * dataset.resolution**2
    num_samples = len(dataset.dataset)

    if cache_dir.exists() and not force_recompute:
        basis = load_pca_cache(cache_dir, device=device, dtype=dtype)
        _validate_cache(basis, dataset, num_samples, expected_dimension)
        return basis

    can_reuse_source = (
        reuse_sample_covariance_cache
        and source_wiener_dir is not None
        and dataset.name.lower() == "cifar10"
        and dataset.split == "train"
        and num_samples == 50_000
        and not isinstance(dataset.dataset, Subset)
    )
    if can_reuse_source:
        basis = _convert_artem_wiener_cache(
            dataset=dataset,
            source_dir=source_wiener_dir,
            num_samples=num_samples,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    else:
        basis = _compute_population_pca(
            dataset=dataset,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if basis.dimension != expected_dimension:
        raise RuntimeError("Computed PCA dimension does not match the dataset.")
    save_pca_cache(basis, cache_dir)
    return basis


@torch.no_grad()
def _compute_population_pca(
    dataset: DatasetBundle,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> PCABasis:
    dimension = dataset.in_channels * dataset.resolution**2
    total = 0
    sample_sum = torch.zeros(dimension, device=device, dtype=dtype)

    for batch in dataset.dataloader:
        images = _images_from_batch(batch).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        flattened = images.flatten(start_dim=1)
        sample_sum += flattened.sum(dim=0)
        total += flattened.shape[0]
    if total == 0:
        raise RuntimeError("Cannot compute PCA from an empty dataset.")
    mean = sample_sum / total

    covariance = torch.zeros(
        dimension,
        dimension,
        device=device,
        dtype=dtype,
    )
    for batch in dataset.dataloader:
        images = _images_from_batch(batch).to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
        centered = images.flatten(start_dim=1) - mean
        covariance.addmm_(centered.T, centered)
    covariance /= total
    covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    min_eigenvalue = float(eigenvalues.min().detach().cpu())
    lambda_sq = eigenvalues[order].clamp_min(0.0)
    U = eigenvectors[:, order]
    metadata = _metadata(
        dataset=dataset,
        num_samples=total,
        dimension=dimension,
        dtype=dtype,
        seed=seed,
        method="batch_population_covariance_eigh",
        extra={"minimum_eigenvalue_before_clamp": min_eigenvalue},
    )
    return PCABasis(mean=mean, U=U, lambda_sq=lambda_sq, metadata=metadata)


def _convert_artem_wiener_cache(
    dataset: DatasetBundle,
    source_dir: Path,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> PCABasis:
    required = [source_dir / "U.pt", source_dir / "LA.pt", source_dir / "mean.pt"]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete Artem Wiener cache at {source_dir}.")

    U = torch.load(required[0], map_location=device, weights_only=True).to(dtype=dtype)
    sample_lambda_sq = torch.load(
        required[1], map_location=device, weights_only=True
    ).to(dtype=dtype)
    mean = torch.load(required[2], map_location=device, weights_only=True).to(dtype=dtype)
    conversion_factor = (num_samples - 1) / num_samples
    lambda_sq = (sample_lambda_sq * conversion_factor).clamp_min(0.0)
    metadata = _metadata(
        dataset=dataset,
        num_samples=num_samples,
        dimension=mean.numel(),
        dtype=dtype,
        seed=seed,
        method="converted_artem_wiener_svd",
        extra={
            "source_cache": str(source_dir.resolve()),
            "source_covariance_convention": "sample_1_over_N_minus_1",
            "population_conversion_factor": conversion_factor,
        },
    )
    return PCABasis(mean=mean, U=U, lambda_sq=lambda_sq, metadata=metadata)


def _metadata(
    *,
    dataset: DatasetBundle,
    num_samples: int,
    dimension: int,
    dtype: torch.dtype,
    seed: int,
    method: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    subset_indices: list[int] | None = None
    if isinstance(dataset.dataset, Subset):
        subset_indices = [int(index) for index in dataset.dataset.indices]
    return {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_samples": num_samples,
        "subset_indices": subset_indices,
        "resolution": dataset.resolution,
        "channels": dataset.in_channels,
        "dimension": dimension,
        "normalization": {
            "input_range": [-1.0, 1.0],
            "torchvision_mean": [0.5] * dataset.in_channels,
            "torchvision_std": [0.5] * dataset.in_channels,
        },
        "covariance_convention": "population_1_over_N",
        "spectral_coordinates": "non_centered_z_equals_x_times_U",
        "seed": seed,
        "dtype": str(dtype).removeprefix("torch."),
        "method": method,
        **extra,
    }


def _validate_cache(
    basis: PCABasis,
    dataset: DatasetBundle,
    num_samples: int,
    dimension: int,
) -> None:
    expected = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_samples": num_samples,
        "dimension": dimension,
        "covariance_convention": "population_1_over_N",
    }
    mismatches = {
        key: (basis.metadata.get(key), value)
        for key, value in expected.items()
        if basis.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"PCA cache metadata mismatch: {mismatches}")
    if basis.mean.shape != (dimension,):
        raise ValueError("Cached mean has an invalid shape.")
    if basis.lambda_sq.shape != (dimension,):
        raise ValueError("Cached eigenvalues have an invalid shape.")
    if basis.U.shape != (dimension, dimension):
        raise ValueError("Cached PCA basis has an invalid shape.")


def _images_from_batch(batch: Any) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not isinstance(images, torch.Tensor):
        raise TypeError("Dataset batches must contain image tensors.")
    return images
