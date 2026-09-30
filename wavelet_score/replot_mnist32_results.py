"""Regenerate MNIST-32 MSE figures from an existing metrics.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavelet_score.run_mnist32_denoising import plot_mse_by_degree, plot_mse_by_family


def parse_args() -> argparse.Namespace:
    """Parse the existing experiment directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("wavelet_score/results/mnist32_denoising"),
    )
    return parser.parse_args()


def main() -> None:
    """Load stored metrics and overwrite only the two MSE figures."""
    args = parse_args()
    metrics_path = args.run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing experiment metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics.json must contain a non-empty list.")

    plot_mse_by_family(metrics, args.run_dir / "mse_vs_sigma_by_family.png")
    plot_mse_by_degree(metrics, args.run_dir / "mse_vs_sigma_by_degree.png")
    print(f"regenerated MSE plots in {args.run_dir}")


if __name__ == "__main__":
    main()
