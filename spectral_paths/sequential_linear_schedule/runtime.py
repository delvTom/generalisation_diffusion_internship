"""Configuration, dataset, and run-directory helpers for the experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from local_diffusion.configuration import load_config
from local_diffusion.data import DatasetBundle, build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class AffineRunPaths:
    """Directories and files belonging to one affine experiment run."""

    run_dir: Path
    images: Path
    tensors: Path
    config: Path
    metrics: Path


def load_affine_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    overrides: Sequence[str] | None = None,
    *,
    pilot: bool = False,
) -> DictConfig:
    """Load the experiment YAML and optional OmegaConf dot-list overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if pilot:
        cfg = _pilot_config(cfg)
    _resolve_paths(cfg)
    return cfg


def load_dataset_from_config(cfg: DictConfig) -> DatasetBundle:
    """Build CIFAR-10 through the repository's existing data pipeline."""
    overrides = [
        f"dataset.batch_size={int(cfg.dataset.batch_size)}",
        f"dataset.num_workers={int(cfg.dataset.num_workers)}",
    ]
    subset_size = cfg.dataset.subset_size
    overrides.append(
        "dataset.subset_size=null"
        if subset_size is None
        else f"dataset.subset_size={int(subset_size)}"
    )
    dataset_cfg = load_config(str(cfg.dataset.config_path), overrides=overrides)
    return build_dataset(dataset_cfg.dataset)


def select_device(cfg: DictConfig) -> torch.device:
    """Resolve the requested device, optionally falling back to CPU."""
    requested = str(cfg.experiment.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if not bool(cfg.experiment.allow_device_fallback):
            raise RuntimeError("CUDA was requested but is not available.")
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def torch_dtype(name: str) -> torch.dtype:
    """Map a configuration string to a supported floating dtype."""
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}. Expected one of {mapping}.") from exc


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_paths(cfg: DictConfig) -> AffineRunPaths:
    """Create a self-contained result directory for one configuration."""
    epsilon_slug = f"{float(cfg.schedule.epsilon_1):g}".replace(".", "p")
    base_name = cfg.experiment.run_name or f"affine_cifar10_eps_{epsilon_slug}"
    if bool(cfg.experiment.append_timestamp):
        base_name = f"{base_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(cfg.experiment.output_root) / base_name
    images = run_dir / "images"
    tensors = run_dir / "tensors"
    images.mkdir(parents=True, exist_ok=True)
    tensors.mkdir(parents=True, exist_ok=True)
    return AffineRunPaths(
        run_dir=run_dir,
        images=images,
        tensors=tensors,
        config=run_dir / "config.yaml",
        metrics=run_dir / "metrics.json",
    )


def representative_indices(dimension: int, fractions: Sequence[float]) -> list[int]:
    """Map fractions in ``[0, 1]`` to unique PCA indices in descending order."""
    indices: list[int] = []
    for fraction in fractions:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("Representative fractions must lie in [0, 1].")
        index = round(float(fraction) * (dimension - 1))
        if index not in indices:
            indices.append(index)
    return indices


def target_pixel_indices(
    targets: Sequence[DictConfig],
    *,
    channels: int,
    height: int,
    width: int,
) -> list[int]:
    """Convert ``(channel, y, x)`` target specifications to flat indices."""
    indices: list[int] = []
    for target in targets:
        channel = int(target.channel)
        y = int(target.y)
        x = int(target.x)
        if not 0 <= channel < channels or not 0 <= y < height or not 0 <= x < width:
            raise ValueError(f"Invalid sensitivity target: {dict(target)}")
        indices.append(channel * height * width + y * width + x)
    return indices


def _resolve_paths(cfg: DictConfig) -> None:
    path_fields = [
        (cfg.dataset, "config_path"),
        (cfg.pca, "cache_dir"),
        (cfg.pca, "source_wiener_dir"),
        (cfg.fid, "real_statistics_cache"),
        (cfg.experiment, "output_root"),
    ]
    for section, key in path_fields:
        value = Path(str(section[key]))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        section[key] = str(value.resolve())
    weights_path = cfg.fid.feature_extractor_weights_path
    if weights_path is not None:
        value = Path(str(weights_path))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        cfg.fid.feature_extractor_weights_path = str(value.resolve())


def _pilot_config(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.evaluation.visual_num_samples = 2
    cfg.evaluation.mse_num_samples = 16
    cfg.evaluation.report_step_counts = [8, 16, 32]
    cfg.evaluation.final_error_step_counts = [4, 8, 16, 32]
    cfg.evaluation.time_curve_step_counts = [8, 16, 32]
    cfg.evaluation.time_curve_max_points = 64
    cfg.evaluation.euler_time_chunk_size = 8
    cfg.fid.enabled = False
    cfg.plot.dpi = 100
    return cfg
