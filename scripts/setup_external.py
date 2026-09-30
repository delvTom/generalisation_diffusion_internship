"""Fetch the pinned upstream locality package and link its runtime files.

The upstream source remains a separate checkout under external/. No code from
that project is vendored in this repository.

Example: python scripts/setup_external.py
Offline check: python scripts/setup_external.py --source /path/to/upstream --skip-install
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream_locality"
SOURCE = "https://github.com/ottogin/locality-in-diffusion-models.git"
COMMIT = "48d69957a668eeaeff8efad50eb26830299e6484"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def link(name: str) -> None:
    destination = ROOT / name
    target = UPSTREAM / name
    if not target.exists():
        raise FileNotFoundError(f"Expected upstream directory: {target}")
    if destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise RuntimeError(f"{destination} points elsewhere; will not replace it")
        return
    if destination.exists():
        raise RuntimeError(f"{destination} already exists; will not replace it")
    destination.symlink_to(target.relative_to(ROOT), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE, help="Git URL or local clone")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    if not UPSTREAM.exists():
        UPSTREAM.mkdir(parents=True)
        run("git", "init", "-q", str(UPSTREAM))
        run("git", "-C", str(UPSTREAM), "fetch", "--depth=1", args.source, COMMIT)
        run("git", "-C", str(UPSTREAM), "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Upstream checkout is {actual}, expected {COMMIT}")

    link("src")
    link("configs")
    if not args.skip_install:
        run(sys.executable, "-m", "pip", "install", "-e", str(UPSTREAM))
    print(f"Upstream locality ready at {UPSTREAM} ({COMMIT[:7]})")


if __name__ == "__main__":
    main()
