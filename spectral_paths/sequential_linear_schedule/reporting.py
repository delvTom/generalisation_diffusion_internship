"""Compact, self-contained figures for the affine Euler experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter


PATH_COLORS = {
    "coupled": "#2878B5",
    "isotropic": "#E07A2D",
}
PATH_LABELS = {
    "coupled": "Coupled noise path",
    "isotropic": "Isotropic noise path",
}


def plot_final_sample_grid(
    rows: Sequence[tuple[str, str]],
    samples: Mapping[str, torch.Tensor],
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Compare exact, classic Wiener, and Euler endpoints with matched noise."""
    if not rows:
        raise ValueError("At least one sample row is required.")
    sample_count = samples[rows[0][0]].shape[0]
    if sample_count <= 0:
        raise ValueError("At least one visual sample is required.")
    if any(samples[key].shape[0] != sample_count for key, _ in rows):
        raise ValueError("All sample rows must contain the same number of seeds.")

    figure, axes = plt.subplots(
        len(rows),
        sample_count,
        figsize=(max(12.5, 2.05 * sample_count + 2.5), 2.0 * len(rows) + 1.1),
        squeeze=False,
    )
    for row_index, (key, label) in enumerate(rows):
        display = postprocess(samples[key]).detach().cpu()
        for column in range(sample_count):
            axis = axes[row_index, column]
            axis.imshow(_image_for_matplotlib(display[column]), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"Noise seed {column + 1}", fontsize=11, pad=7)
        axes[row_index, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=10.5,
            labelpad=12,
        )

    figure.suptitle(
        "Affine generation: exact endpoint, classic Wiener, and sequential paths",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.955,
        "Matched noise by column: legacy pixel epsilon (seed 42), with "
        "z0 = U^T epsilon for every spectral generator.",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.012,
        "Images are mapped from the model range [-1, 1] and clipped to [0, 1] only for display.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.19,
        right=0.99,
        top=0.91,
        bottom=0.04,
        wspace=0.06,
        hspace=0.13,
    )
    _save_figure(figure, output_path, dpi)


def plot_final_mse_vs_steps(
    statistics: Mapping[str, Mapping[int, Mapping[str, float]]],
    output_path: Path,
    *,
    batch_size: int,
    dimension: int,
    dpi: int,
) -> None:
    """Plot endpoint MSE and its Monte-Carlo standard error versus Euler steps."""
    figure, axis = plt.subplots(figsize=(10.8, 6.8))
    for path_name in ("coupled", "isotropic"):
        path_values = statistics[path_name]
        steps = np.asarray(sorted(path_values), dtype=np.int64)
        means = np.asarray([path_values[int(step)]["mean"] for step in steps])
        standard_errors = np.asarray(
            [path_values[int(step)]["standard_error"] for step in steps]
        )
        axis.errorbar(
            steps,
            means,
            yerr=standard_errors,
            color=PATH_COLORS[path_name],
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            capsize=3,
            label=PATH_LABELS[path_name],
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of explicit-Euler steps N (field evaluations)", fontsize=11)
    axis.set_ylabel("Final mean squared error at t = 1", fontsize=11)
    displayed_steps = np.asarray(sorted(statistics["coupled"]), dtype=np.int64)
    axis.set_xticks(displayed_steps)
    axis.set_xticklabels(
        [f"{int(step):,}" for step in displayed_steps],
        rotation=35,
        ha="right",
    )
    axis.xaxis.set_minor_formatter(ScalarFormatter())
    axis.tick_params(axis="x", which="minor", labelbottom=False)
    axis.grid(which="both", alpha=0.23)
    axis.legend(title="Probability path", fontsize=10, title_fontsize=10)
    figure.suptitle("Euler endpoint error versus computational budget", fontsize=15, y=0.98)
    figure.text(
        0.5,
        0.915,
        (
            r"$\mathrm{MSE}_N(1)=\frac{1}{BD}\sum_{b=1}^{B}"
            r"\|z^{\mathrm{Euler},N}_b(1)-z_b^\star(1)\|_2^2$"
            f"    with B={batch_size} and D={dimension}"
        ),
        ha="center",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.025,
        "Error bars show +/- one standard error over the matched initial noises. "
        "Orthogonality of the PCA basis makes this equal to raw pixel-space MSE.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.035, 0.065, 0.995, 0.89))
    _save_figure(figure, output_path, dpi)


def plot_mse_vs_time(
    curves: Mapping[str, Mapping[int, Mapping[str, torch.Tensor | str]]],
    output_path: Path,
    *,
    batch_size: int,
    dimension: int,
    dpi: int,
) -> None:
    """Plot Euler-to-exact MSE throughout the trajectory for selected budgets."""
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.3), sharey=True)
    line_colors = ["#3B4CC0", "#2AA198", "#C44E52", "#7A5195"]
    for axis, path_name in zip(axes, ("coupled", "isotropic")):
        for color, num_steps in zip(line_colors, sorted(curves[path_name])):
            payload = curves[path_name][num_steps]
            times = _as_numpy(payload["times"])
            mse = _as_numpy(payload["mse"])
            positive = mse[mse > 0.0]
            floor = max(float(positive.min()) * 0.25, 1e-18) if positive.size else 1e-18
            axis.plot(
                times,
                np.maximum(mse, floor),
                color=color,
                linewidth=2.0,
                label=str(payload["label"]),
            )
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Flow time t", fontsize=11)
        axis.set_title(PATH_LABELS[path_name], fontsize=13)
        axis.grid(which="both", alpha=0.23)
        axis.legend(title="Euler budget", fontsize=9.5, title_fontsize=10)
    axes[0].set_ylabel("Mean squared error at time t", fontsize=11)

    figure.suptitle("Where does explicit Euler depart from the exact affine flow?", fontsize=15, y=0.98)
    figure.text(
        0.5,
        0.91,
        (
            r"$\mathrm{MSE}_N(t)=\frac{1}{BD}\sum_{b=1}^{B}"
            r"\|z^{\mathrm{Euler},N}_b(t)-z_b^\star(t)\|_2^2$"
            f"    with B={batch_size} and D={dimension}"
        ),
        ha="center",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.025,
        "Euler and the exact solution are compared at the same time n/N. "
        "Long trajectories are subsampled for display only.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.03, 0.065, 0.995, 0.88))
    _save_figure(figure, output_path, dpi)


def plot_fid_comparison(
    methods: Sequence[tuple[str, str, str]],
    fid_values: Mapping[str, float],
    output_path: Path,
    *,
    num_generated: int,
    num_real: int,
    dpi: int,
) -> None:
    """Plot FID for the exact endpoint and selected Euler endpoints."""
    keys = [key for key, _, _ in methods]
    labels = [label for _, label, _ in methods]
    colors = [
        "#777777" if path_name == "exact" else PATH_COLORS[path_name]
        for _, _, path_name in methods
    ]
    values = np.asarray([fid_values[key] for key in keys], dtype=np.float64)
    positions = np.arange(len(values), dtype=np.float64)

    figure, axis = plt.subplots(figsize=(12.8, 6.8))
    bars = axis.bar(positions, values, color=colors, width=0.72)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9.5)
    axis.set_xticks(positions, labels, fontsize=9.5)
    axis.set_ylabel("Fréchet Inception Distance (lower is better)", fontsize=11)
    axis.set_ylim(0.0, max(values.max() * 1.17, 1.0))
    axis.grid(axis="y", alpha=0.23)

    exact_proxy = plt.Rectangle((0, 0), 1, 1, color="#777777")
    coupled_proxy = plt.Rectangle((0, 0), 1, 1, color=PATH_COLORS["coupled"])
    isotropic_proxy = plt.Rectangle((0, 0), 1, 1, color=PATH_COLORS["isotropic"])
    axis.legend(
        [exact_proxy, coupled_proxy, isotropic_proxy],
        ["Exact common endpoint", PATH_LABELS["coupled"], PATH_LABELS["isotropic"]],
        loc="upper right",
        fontsize=9.5,
    )
    figure.suptitle("Distribution quality of exact and Euler affine generators", fontsize=15, y=0.98)
    figure.text(
        0.5,
        0.915,
        (
            f"Standard Inception-v3 FID: {num_generated:,} generated samples "
            f"against {num_real:,} CIFAR-10 training images"
        ),
        ha="center",
        fontsize=10.5,
    )
    figure.text(
        0.5,
        0.025,
        "All methods use the same generated noise sequence; outputs are clipped to [0, 1] before FID.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.025, 0.07, 0.995, 0.89))
    _save_figure(figure, output_path, dpi)


def _image_for_matplotlib(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu()
    if array.ndim != 3:
        raise ValueError("Expected an image tensor with shape [C, H, W].")
    if array.shape[0] == 1:
        return array[0].numpy()
    return array.permute(1, 2, 0).numpy()


def _as_numpy(value: torch.Tensor | np.ndarray | object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_figure(figure: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)
