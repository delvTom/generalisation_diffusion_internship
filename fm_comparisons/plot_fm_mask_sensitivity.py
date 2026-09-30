"""Plot sensitivity fields for closed-form FM masks and the FM U-Net.

For selected FM times t, target pixels q, and one color channel, this script
visualizes the raw sensitivity fields associated with the x1_mask and
velocity_mask closed-form estimators, together with input-gradient sensitivity
fields of the pretrained FM U-Net. Each field is normalized to [0, 1] only for
visualization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fm_comparisons.compare_fm_unet_closed_form_mse import (
    MASK_KINDS,
    build_fm_wiener_operator,
    build_mask_operator,
    collect_images,
    dataset_output_slug,
    default_checkpoint_for_dataset,
    fm_unet_velocity_forward,
    load_dataset,
    load_experiment_config,
    load_fm_unet,
    load_or_compute_wiener_svd,
    parse_t_values,
)

PLOT_KINDS = ("x1_mask", "velocity_mask", "U-Net")
PLOT_LABELS = {
    "x1_mask": "D\u00e9bruiteur affine\nde Wiener",
    "velocity_mask": "Estimateur affine\nde v\u00e9locit\u00e9",
    "U-Net": "U-Net FM\n(v\u00e9locit\u00e9)",
}


def pixel_index(channel: int, y: int, x: int, height: int, width: int) -> int:
    """Map image coordinates (channel, y, x) to a flattened pixel index."""
    return channel * height * width + y * width + x


def default_target_pixels(height: int, width: int) -> list[tuple[int, int]]:
    """Return a small set of spatial target pixels q=(y, x)."""
    center_y, center_x = height // 2, width // 2
    offset = max(1, min(height, width) // 5)
    return [
        (center_y, center_x),
        (max(0, center_y - offset), center_x),
        (min(height - 1, center_y + offset), center_x),
        (center_y, max(0, center_x - offset)),
        (center_y, min(width - 1, center_x + offset)),
    ]


def parse_target_pixels(value: str | None, height: int, width: int) -> list[tuple[int, int]]:
    """Parse target pixels formatted as ``y:x,y:x`` or use the defaults."""
    if value is None:
        return default_target_pixels(height, width)

    pixels: list[tuple[int, int]] = []
    for item in value.split(","):
        coordinates = item.strip().split(":")
        if len(coordinates) != 2:
            raise ValueError("--target-pixels must use the format y:x,y:x")
        y, x = (int(coordinate) for coordinate in coordinates)
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError(f"Target pixel {(y, x)} lies outside {height}x{width}.")
        pixels.append((y, x))
    if not pixels:
        raise ValueError("At least one target pixel is required.")
    return pixels

def spatial_sensitivity_field(
    operator: torch.Tensor,
    y: int,
    x: int,
    channel: int,
    num_channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Extract same-channel sensitivity at target pixel q=(y, x)."""
    q = pixel_index(channel, y, x, height, width)

    # row[p] is d output[channel, y, x] / d input[p].
    row = operator[q, :]  # [N]
    channel_rows = row.abs().reshape(num_channels, height, width)  # [C_in, H, W]
    field = channel_rows[channel]  # [H, W]
    return field / (field.max() + 1e-12)


def unet_spatial_sensitivity_field(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t: float,
    y: int,
    x: int,
    channel: int,
) -> torch.Tensor:
    """Estimate same-channel U-Net sensitivity at q=(y, x)."""
    xt = xt.detach().clone().requires_grad_(True)  # [B, C, H, W]
    velocity = fm_unet_velocity_forward(model, xt, t)  # [B, C, H, W]

    scalar_output = velocity[:, channel, y, x].sum()
    grad = torch.autograd.grad(scalar_output, xt)[0]  # [B, C_in, H, W]
    field = grad[:, channel].abs().mean(dim=0)  # [H, W], averaged over B only.
    return field / (field.max() + 1e-12)


def plot_sensitivity_grid(
    fields: dict[tuple[str, tuple[int, int], float], torch.Tensor],
    target_pixels: list[tuple[int, int]],
    t_values: list[float],
    out_path: Path,
    sqrt_display: bool,
) -> None:
    """Save one grid comparing closed-form and U-Net sensitivities."""
    num_rows = len(target_pixels) * len(PLOT_KINDS)
    num_cols = len(t_values)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(2.0 * num_cols, 2.0 * num_rows),
        squeeze=False,
    )

    for col, t in enumerate(t_values):
        axes[0, col].set_title(f"t={t:.2f}", fontsize=10)

    for pixel_id, (y, x) in enumerate(target_pixels):
        for kind_id, plot_kind in enumerate(PLOT_KINDS):
            row = pixel_id * len(PLOT_KINDS) + kind_id
            for col, t in enumerate(t_values):
                ax = axes[row, col]
                field = fields[(plot_kind, (y, x), t)].detach().cpu()
                if sqrt_display:
                    field = field.sqrt()
                image = ax.imshow(
                    field,
                    cmap="turbo",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax.scatter([x], [y], s=8, c="red")
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(
                        PLOT_LABELS[plot_kind],
                        fontsize=12,
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=12,
                    )

        group_center = (pixel_id * len(PLOT_KINDS) + 1 + 0.5) / num_rows
        fig.text(
            0.012,
            1.0 - group_center,
            rf"$q_{pixel_id + 1}=({y},{x})$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.015)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-unet-samples", type=int, default=64)
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Input and output color channel: 0=red, 1=green, 2=blue.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-values", default="0.10,0.25,0.40,0.55,0.70,0.85,0.95")
    parser.add_argument(
        "--target-pixels",
        default=None,
        help="Comma-separated target pixels formatted as y:x,y:x.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sqrt-display",
        action="store_true",
        help="Apply sqrt to normalized sensitivity fields before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or Path(
        "fm_comparisons/results"
    ) / f"fm_mask_sensitivity_{dataset_output_slug(args.dataset)}"
    checkpoint_path = args.checkpoint or default_checkpoint_for_dataset(args.dataset)

    cfg = load_experiment_config(args.dataset, args.batch_size, subset_size=None)
    dataset = load_dataset(cfg)
    if not 0 <= args.channel < dataset.in_channels:
        raise ValueError(
            f"channel must be between 0 and {dataset.in_channels - 1}, got {args.channel}."
        )
    t_values = parse_t_values(args.t_values)
    U, eigenvalues, Vh, _ = load_or_compute_wiener_svd(dataset, device)
    model = load_fm_unet(checkpoint_path, device, args.dataset)
    target_pixels = parse_target_pixels(
        args.target_pixels,
        dataset.resolution,
        dataset.resolution,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x1_unet = collect_images(dataset, args.num_unet_samples, device)
    x0_unet = torch.randn(x1_unet.shape, device=device, generator=generator)

    print(f"dataset: {dataset.name}")
    print(f"resolution: {dataset.resolution}")
    print(f"channels: {dataset.in_channels}")
    print(f"selected channel: {args.channel}")
    print(f"t_values: {t_values}")
    print(f"target_pixels: {target_pixels}")
    print(f"num_unet_samples: {args.num_unet_samples}")

    fields: dict[tuple[str, tuple[int, int], float], torch.Tensor] = {}
    for t in t_values:
        print(f"computing sensitivity fields for t={t:.2f}")
        wiener_operator = build_fm_wiener_operator(U, eigenvalues, Vh, t)
        for mask_kind in MASK_KINDS:
            operator = build_mask_operator(wiener_operator, t, mask_kind)
            for y, x in target_pixels:
                fields[(mask_kind, (y, x), t)] = spatial_sensitivity_field(
                    operator=operator,
                    y=y,
                    x=x,
                    channel=args.channel,
                    num_channels=dataset.in_channels,
                    height=dataset.resolution,
                    width=dataset.resolution,
                )

        xt_unet = (1.0 - t) * x0_unet + t * x1_unet
        for y, x in target_pixels:
            fields[("U-Net", (y, x), t)] = unet_spatial_sensitivity_field(
                model=model,
                xt=xt_unet,
                t=t,
                y=y,
                x=x,
                channel=args.channel,
            )

    suffix = "_sqrt" if args.sqrt_display else ""
    out_path = out_dir / f"sensitivity_channel_{args.channel}{suffix}.png"
    plot_sensitivity_grid(fields, target_pixels, t_values, out_path, args.sqrt_display)
    torch.save(
        {
            "fields": {key: value.detach().cpu() for key, value in fields.items()},
            "target_pixels": target_pixels,
            "t_values": t_values,
            "channel": args.channel,
            "sqrt_display": args.sqrt_display,
        },
        out_path.with_suffix(".pt"),
    )
    with out_path.with_name(f"sensitivity_channel_{args.channel}{suffix}_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "checkpoint": str(checkpoint_path),
                "channel": args.channel,
                "target_pixels_yx": target_pixels,
                "times": t_values,
                "num_unet_samples": args.num_unet_samples,
                "seed": args.seed,
                "normalization": "each sensitivity map divided by its own maximum",
                "display_transform": "sqrt" if args.sqrt_display else "identity",
            },
            handle,
            indent=2,
        )
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
