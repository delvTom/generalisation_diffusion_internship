"""Plot sensitivity fields and binary masks induced by Wiener_asym.

For a PCA rank r, Wiener_asym is the affine closed-form estimator of the raw
AsymFlow target

    u_A = x_1 - P_r eps

under the Flow Matching convention x_t = t x_1 + (1 - t) eps.  This script
visualizes the row q of its linear sensitivity operator A_asym,r,t, and the
binary mask obtained from this operator with Artem's normalization/thresholding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (  # noqa: E402
    dataset_output_slug,
    load_dataset,
    parse_t_values,
)
from fm_comparisons.compare_wiener_asym_direct_cifar10 import (  # noqa: E402
    build_asymflow_affine_diagonal,
    load_experiment_config,
    parse_ranks,
)


def parse_args() -> argparse.Namespace:
    """Parse plotting options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ranks", default="0,8,32,256,3072")
    parser.add_argument("--t-values", default="0.10,0.25,0.40,0.60,0.85,0.95")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--sqrt-display", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used to load the Wiener basis. Use cpu if the GPU is busy.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def pixel_index(channel: int, y: int, x: int, height: int, width: int) -> int:
    """Map image coordinates (channel, y, x) to a flattened coordinate."""
    return channel * height * width + y * width + x


def build_wiener_asym_operator(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    rank: int,
    t: float,
) -> torch.Tensor:
    """Build the linear sensitivity A_asym,r,t in pixel coordinates."""
    diagonal = build_asymflow_affine_diagonal(eigenvalues, rank, t)
    return (U * diagonal.unsqueeze(0)) @ U.T


def load_wiener_basis(dataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load only the PCA basis and eigenvalues needed for sensitivity rows."""
    wiener_path = Path("data/models/wiener") / f"{dataset.name}_{dataset.resolution}"
    U_path = wiener_path / "U.pt"
    eigenvalues_path = wiener_path / "LA.pt"
    if not (U_path.exists() and eigenvalues_path.exists()):
        raise FileNotFoundError(
            f"Missing cached Wiener basis at {wiener_path}. "
            "Run a script that computes the Wiener SVD first."
        )

    U = torch.load(U_path, map_location=device, weights_only=True)
    eigenvalues = torch.load(eigenvalues_path, map_location=device, weights_only=True)
    return U, eigenvalues


def build_wiener_asym_row(
    U: torch.Tensor,
    eigenvalues: torch.Tensor,
    rank: int,
    t: float,
    q: int,
) -> torch.Tensor:
    """Build only row q of A_asym,r,t without materializing the full matrix."""
    diagonal = build_asymflow_affine_diagonal(eigenvalues, rank, t)
    return (U[q, :] * diagonal) @ U.T


def normalized_sensitivity_field(
    row: torch.Tensor,
    channel: int,
    num_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return |A[q,p]| reshaped as a same-channel spatial field in [0, 1]."""
    channel_row = row.abs().reshape(num_channels, height, width)
    field = channel_row[channel]
    return field / (field.max() + 1e-12)


def binary_mask_field(
    q: int,
    row: torch.Tensor,
    channel: int,
    num_channels: int,
    height: int,
    width: int,
    threshold: float,
) -> torch.Tensor:
    """Return a row-local binary Artem-style mask for output q.

    The generation code uses the full-matrix max after row normalization.  For
    CelebA, materializing the full 12288 x 12288 operator is too expensive for
    a diagnostic plot, so here we apply the same row normalization and threshold
    on the single displayed row.
    """
    diagonal = row[q]
    if diagonal.abs() < 1e-6:
        diagonal = torch.ones((), device=row.device, dtype=row.dtype)
    normalized = row / diagonal
    cutoff = normalized.abs().max() * threshold
    mask_row = (normalized.abs() >= cutoff).float()
    mask_row[q] = 1.0
    field = mask_row.reshape(num_channels, height, width)
    return field[channel]


def maybe_empty_cuda_cache() -> None:
    """Release cached CUDA memory after row computations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_grid(
    fields: dict[tuple[str, int, float], torch.Tensor],
    ranks: list[int],
    t_values: list[float],
    title_prefix: str,
    out_path: Path,
    sqrt_display: bool,
    cmap: str,
) -> None:
    """Save a grid with rows=ranks and columns=t values."""
    fig, axes = plt.subplots(
        len(ranks),
        len(t_values),
        figsize=(2.0 * len(t_values), 2.0 * len(ranks)),
        squeeze=False,
    )

    for col, t in enumerate(t_values):
        axes[0, col].set_title(f"t={t:.2f}", fontsize=11)

    image = None
    for row, rank in enumerate(ranks):
        for col, t in enumerate(t_values):
            ax = axes[row, col]
            field = fields[(title_prefix, rank, t)].detach().cpu()
            if sqrt_display:
                field = field.sqrt()
            image = ax.imshow(field, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"r={rank}", fontsize=13)

    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    """Build and plot Wiener_asym sensitivity fields."""
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = args.out_dir or Path("fm_comparisons/results") / (
        f"wiener_asym_sensitivity_{dataset_output_slug(args.dataset)}"
    )

    cfg = load_experiment_config(args.dataset, args.batch_size, args.num_workers)
    dataset = load_dataset(cfg)
    if not 0 <= args.channel < dataset.in_channels:
        raise ValueError(f"channel must be in [0, {dataset.in_channels - 1}]")

    U, eigenvalues = load_wiener_basis(dataset, device)
    ranks = parse_ranks(args.ranks, max_rank=U.shape[1])
    t_values = parse_t_values(args.t_values)
    pca_params = cfg.model.params or {}
    threshold = args.mask_threshold
    if threshold is None:
        threshold = float(pca_params.get("mask_threshold", 0.02))

    height = dataset.resolution
    width = dataset.resolution
    y = args.y if args.y is not None else height // 2
    x = args.x if args.x is not None else width // 2
    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"target pixel must be inside image, got y={y}, x={x}")
    q = pixel_index(args.channel, y, x, height, width)

    print(f"dataset: {dataset.name}")
    print(f"device: {device}")
    print(f"ranks: {ranks}")
    print(f"t_values: {t_values}")
    print(f"channel: {args.channel}")
    print(f"target pixel: (y={y}, x={x})")
    print(f"mask_threshold: {threshold}")
    print(f"out_dir: {out_dir}")

    sensitivity_fields: dict[tuple[str, int, float], torch.Tensor] = {}
    binary_fields: dict[tuple[str, int, float], torch.Tensor] = {}
    for rank in ranks:
        for t in t_values:
            print(f"computing rank={rank}, t={t:.2f}")
            row = build_wiener_asym_row(U, eigenvalues, rank, t, q)
            sensitivity_fields[("sensitivity", rank, t)] = normalized_sensitivity_field(
                row=row,
                channel=args.channel,
                num_channels=dataset.in_channels,
                height=height,
                width=width,
            )
            binary_fields[("mask", rank, t)] = binary_mask_field(
                q=q,
                row=row,
                channel=args.channel,
                num_channels=dataset.in_channels,
                height=height,
                width=width,
                threshold=threshold,
            )
            maybe_empty_cuda_cache()

    suffix = "_sqrt" if args.sqrt_display else ""
    sensitivity_path = out_dir / f"sensitivity_channel_{args.channel}{suffix}.png"
    mask_path = out_dir / f"binary_mask_channel_{args.channel}.png"
    plot_grid(
        fields=sensitivity_fields,
        ranks=ranks,
        t_values=t_values,
        title_prefix="sensitivity",
        out_path=sensitivity_path,
        sqrt_display=args.sqrt_display,
        cmap="turbo",
    )
    plot_grid(
        fields=binary_fields,
        ranks=ranks,
        t_values=t_values,
        title_prefix="mask",
        out_path=mask_path,
        sqrt_display=False,
        cmap="gray",
    )
    print(f"saved sensitivity plot to {sensitivity_path}")
    print(f"saved binary mask plot to {mask_path}")


if __name__ == "__main__":
    main()
