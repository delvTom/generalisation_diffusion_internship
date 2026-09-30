"""Standard FID evaluation for diagonal affine spectral generators."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from .euler import DiagonalAffineMap
from .pca import PCABasis


@dataclass(frozen=True)
class FIDEvaluation:
    """FID values and raw-range clipping diagnostics for every generator."""

    values: dict[str, float]
    clipping_fractions: dict[str, float]
    real_statistics_source: str


@torch.no_grad()
def evaluate_affine_maps_fid(
    maps: Mapping[str, DiagonalAffineMap],
    *,
    basis: PCABasis,
    dataset,
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    num_generated: int,
    num_real: int,
    batch_size: int,
    feature: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    real_statistics_cache: Path,
    feature_extractor_weights_path: Path | None = None,
) -> FIDEvaluation:
    """Evaluate maps with TorchMetrics' standard torch-fidelity Inception FID.

    Every map receives the same pseudorandom Gaussian sequence. Real Inception
    sufficient statistics are cached with provenance; generated features are
    intentionally recomputed because they depend on the Euler configuration.
    """
    if not maps:
        raise ValueError("At least one affine map is required for FID.")
    if num_generated < 2 or num_real < 2:
        raise ValueError("FID requires at least two real and generated samples.")
    if batch_size <= 0:
        raise ValueError("FID batch_size must be positive.")
    if num_real > len(dataset.dataset):
        raise ValueError(
            f"Requested {num_real} real samples, but the dataset has "
            f"{len(dataset.dataset)}."
        )

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FID requires both torchmetrics and torch-fidelity in the active "
            "environment. Install them with `pip install torchmetrics torch-fidelity`."
        ) from exc

    weights_path = (
        str(feature_extractor_weights_path.resolve())
        if feature_extractor_weights_path is not None
        else None
    )
    try:
        metric = FrechetInceptionDistance(
            feature=feature,
            normalize=True,
            reset_real_features=False,
            feature_extractor_weights_path=weights_path,
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the standard FID Inception-v3 extractor. "
            "On its first use, torch-fidelity downloads pretrained weights; "
            "alternatively set fid.feature_extractor_weights_path to a local file."
        ) from exc

    expected_metadata = _real_cache_metadata(
        dataset=dataset,
        num_real=num_real,
        feature=feature,
        weights_path=weights_path,
    )
    cache_was_loaded = _load_real_statistics_cache(
        metric,
        real_statistics_cache,
        expected_metadata,
        device=device,
    )
    if cache_was_loaded:
        print(f"Loaded real FID statistics: {real_statistics_cache}")
        real_source = "cache"
    else:
        print(f"Computing real FID statistics from {num_real:,} images")
        _update_real_statistics(
            metric,
            dataset,
            postprocess=postprocess,
            num_real=num_real,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        _save_real_statistics_cache(
            metric,
            real_statistics_cache,
            expected_metadata,
        )
        real_source = "computed"

    values: dict[str, float] = {}
    clipping_fractions: dict[str, float] = {}
    dimension = basis.dimension
    image_shape = (dataset.in_channels, dataset.resolution, dataset.resolution)

    for method_name, affine_map in maps.items():
        metric.reset()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        clipped_values = 0
        total_values = 0
        processed = 0
        print(f"Computing FID: {method_name} ({num_generated:,} generated images)")

        while processed < num_generated:
            current_batch = min(batch_size, num_generated - processed)
            z0 = torch.randn(
                current_batch,
                dimension,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            generated_z = affine_map.apply(z0)
            generated_x = basis.to_pixel(generated_z).view(current_batch, *image_shape)
            if not torch.isfinite(generated_x).all():
                raise RuntimeError(
                    f"Non-finite generated values encountered for FID method {method_name}."
                )
            clipped_values += int(
                ((generated_x < -1.0) | (generated_x > 1.0)).sum().item()
            )
            total_values += generated_x.numel()
            metric.update(postprocess(generated_x).to(torch.float32), real=False)
            processed += current_batch
            if processed == num_generated or processed % 2_000 == 0:
                print(f"  generated {processed:,}/{num_generated:,}")

        values[method_name] = float(metric.compute().detach().cpu())
        clipping_fractions[method_name] = clipped_values / total_values

    return FIDEvaluation(
        values=values,
        clipping_fractions=clipping_fractions,
        real_statistics_source=real_source,
    )


@torch.no_grad()
def _update_real_statistics(
    metric,
    dataset,
    *,
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    num_real: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    processed = 0
    for batch in dataset.dataloader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        offset = 0
        while offset < images.shape[0] and processed < num_real:
            current_batch = min(batch_size, images.shape[0] - offset, num_real - processed)
            real_images = postprocess(images[offset : offset + current_batch]).to(torch.float32)
            metric.update(real_images, real=True)
            processed += current_batch
            offset += current_batch
            if processed == num_real or processed % 5_000 == 0:
                print(f"  real {processed:,}/{num_real:,}")
        if processed == num_real:
            break
    if processed != num_real:
        raise RuntimeError(f"Only collected {processed} of {num_real} real FID samples.")


def _real_cache_metadata(
    *,
    dataset,
    num_real: int,
    feature: int,
    weights_path: str | None,
) -> dict[str, object]:
    return {
        "dataset": str(dataset.name),
        "split": str(dataset.split),
        "dataset_size": int(len(dataset.dataset)),
        "num_real": int(num_real),
        "resolution": int(dataset.resolution),
        "channels": int(dataset.in_channels),
        "feature": int(feature),
        "normalize_inputs": True,
        "feature_extractor_weights_path": weights_path,
        "torchmetrics_version": importlib.metadata.version("torchmetrics"),
        "torch_fidelity_version": importlib.metadata.version("torch-fidelity"),
    }


def _load_real_statistics_cache(
    metric,
    cache_path: Path,
    expected_metadata: Mapping[str, object],
    *,
    device: torch.device,
) -> bool:
    if not cache_path.exists():
        return False
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if payload.get("metadata") != dict(expected_metadata):
        print(f"Ignoring incompatible real FID cache: {cache_path}")
        return False
    state_names = (
        "real_features_sum",
        "real_features_cov_sum",
        "real_features_num_samples",
    )
    if any(name not in payload for name in state_names):
        print(f"Ignoring incomplete real FID cache: {cache_path}")
        return False
    sources: dict[str, torch.Tensor] = {}
    for name in state_names:
        target = getattr(metric, name)
        source = payload[name].to(device=device, dtype=target.dtype)
        if source.shape != target.shape:
            print(f"Ignoring real FID cache with invalid {name}: {cache_path}")
            return False
        sources[name] = source
    for name, source in sources.items():
        getattr(metric, name).copy_(source)
    return True


def _save_real_statistics_cache(
    metric,
    cache_path: Path,
    metadata: Mapping[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "real_features_sum": metric.real_features_sum.detach().cpu(),
            "real_features_cov_sum": metric.real_features_cov_sum.detach().cpu(),
            "real_features_num_samples": metric.real_features_num_samples.detach().cpu(),
        },
        cache_path,
    )
    print(f"Saved real FID statistics: {cache_path}")
