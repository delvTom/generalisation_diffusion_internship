"""Prepare the centered population-PCA cache used by the affine experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from spectral_paths.sequential_linear_schedule.pca import (  # noqa: E402
    prepare_population_pca,
)
from spectral_paths.sequential_linear_schedule.runtime import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_affine_config,
    load_dataset_from_config,
    seed_everything,
    select_device,
    torch_dtype,
)


def parse_args() -> argparse.Namespace:
    """Parse the config path and optional OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides such as pca.force_recompute=true.",
    )
    return parser.parse_args()


def main() -> None:
    """Load CIFAR-10 and create or validate its population-PCA cache."""
    args = parse_args()
    cfg = load_affine_config(args.config, args.overrides)
    seed_everything(int(cfg.experiment.seed))
    device = select_device(cfg)
    dtype = torch_dtype(str(cfg.experiment.dtype))
    dataset = load_dataset_from_config(cfg)
    basis = prepare_population_pca(
        dataset,
        Path(cfg.pca.cache_dir),
        device=device,
        dtype=dtype,
        seed=int(cfg.experiment.seed),
        force_recompute=bool(cfg.pca.force_recompute),
        source_wiener_dir=Path(cfg.pca.source_wiener_dir),
        reuse_sample_covariance_cache=bool(cfg.pca.reuse_sample_covariance_cache),
    )
    explained = basis.lambda_sq.cumsum(0) / basis.lambda_sq.sum().clamp_min(1e-12)
    print(f"PCA cache: {cfg.pca.cache_dir}")
    print(f"shape: U={tuple(basis.U.shape)}, lambda_sq={tuple(basis.lambda_sq.shape)}")
    print(f"population variance captured by first PC: {explained[0].item():.6f}")
    print(f"metadata method: {basis.metadata['method']}")


if __name__ == "__main__":
    main()
