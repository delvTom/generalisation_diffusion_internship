"""Check installed dependencies and separately obtained research inputs.

Example: python scripts/check_environment.py --fm
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ("numpy", "torch", "torchvision", "pywt", "matplotlib", "scipy")
FM = ("torchcfm", "omegaconf", "torchmetrics", "tqdm", "local_diffusion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fm", action="store_true", help="Also check FM dependencies")
    args = parser.parse_args()

    missing = []
    for name in CORE + (FM if args.fm else ()):
        present = importlib.util.find_spec(name) is not None
        print(f"{'OK' if present else 'MISSING':7} module {name}")
        if not present:
            missing.append(name)

    if args.fm:
        for name in ("src", "configs"):
            path = ROOT / name
            present = path.exists()
            print(f"{'OK' if present else 'MISSING':7} upstream {name}/")
            if not present:
                missing.append(name)
        for name in (
            "data/datasets",
            "data/models/fm_unet/cifar10/fm_cifar10_weights_step_400000.pt",
            "data/models/wiener/cifar10_32/U.pt",
        ):
            present = (ROOT / name).exists()
            print(f"{'FOUND' if present else 'OPTIONAL':7} input {name}")

    if missing:
        parser.exit(1, f"Missing required setup: {', '.join(missing)}\n")
    print("Required environment checks passed.")


if __name__ == "__main__":
    main()
