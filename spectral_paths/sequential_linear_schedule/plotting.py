"""Figures for schedules, affine trajectories, Euler, and sensitivity."""

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
from matplotlib.colors import SymLogNorm

from .affine import AffineSpectralVelocity
from .schedule import SequentialLinearSchedule


PATH_LABELS = {
    "coupled": "Coupled noise",
    "isotropic": "Isotropic noise",
}


@torch.no_grad()
def plot_schedule_curves(
    schedule: SequentialLinearSchedule,
    indices: Sequence[int],
    output_path: Path,
    *,
    points: int,
    device: torch.device,
    dtype: torch.dtype,
    dpi: int,
) -> None:
    """Plot beta and its derivative for representative PCs."""
    times, beta, dot_beta = _selected_schedule_curves(
        schedule,
        indices,
        points=points,
        device=device,
        dtype=dtype,
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for column, index in enumerate(indices):
        label = f"PC {index + 1}"
        axes[0].plot(times, beta[:, column], label=label)
        axes[1].plot(times, dot_beta[:, column], label=label)
    axes[0].set_ylabel(r"$\beta_k(t)$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel(r"$\dot{\beta}_k(t)$")
    axes[1].set_xlabel("t")
    axes[1].set_yscale("symlog", linthresh=max(schedule.rho, 1e-6))
    axes[1].set_ylim(0.0, schedule.dimension * 1.1)
    axes[0].legend(ncol=min(5, len(indices)), fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


@torch.no_grad()
def plot_affine_coefficient_curves(
    models: Mapping[str, AffineSpectralVelocity],
    indices: Sequence[int],
    output_path: Path,
    *,
    points: int,
    dpi: int,
) -> None:
    """Compare variance, standard deviation, and affine slope over time."""
    figure, axes = plt.subplots(
        len(models),
        4,
        figsize=(19, 4.2 * len(models)),
        sharex=True,
        squeeze=False,
    )
    for row_index, (path_name, model) in enumerate(models.items()):
        times, variance, slope, bias = _selected_affine_curves(model, indices, points)
        for column, component_index in enumerate(indices):
            label = f"PC {component_index + 1}"
            axes[row_index, 0].plot(times, variance[:, column], label=label)
            axes[row_index, 1].plot(
                times,
                np.sqrt(np.clip(variance[:, column], 0.0, None)),
                label=label,
            )
            axes[row_index, 2].plot(times, slope[:, column], label=label)
            axes[row_index, 3].plot(times, bias[:, column], label=label)
        axes[row_index, 0].set_ylabel(PATH_LABELS[path_name])
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 2].set_yscale("symlog", linthresh=1.0)
        axes[row_index, 3].set_yscale("symlog", linthresh=1.0)
        for axis in axes[row_index]:
            axis.grid(alpha=0.25)
    axes[0, 0].set_title(r"Variance $v_k(t)$")
    axes[0, 1].set_title(r"Scale $\sqrt{v_k(t)}$")
    axes[0, 2].set_title(r"Slope $s_k(t)=\dot v_k(t)/(2v_k(t))$")
    axes[0, 3].set_title(r"Bias $c_k(t)=\dot m_k(t)-s_k(t)m_k(t)$")
    for axis in axes[-1]:
        axis.set_xlabel("t")
    axes[0, 0].legend(ncol=1, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_exact_trajectories(
    pixel_trajectories: Mapping[str, torch.Tensor],
    times: Sequence[float],
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    sample_index: int,
    dpi: int,
) -> None:
    """Plot exact intermediate images for the same noise and both paths."""
    figure, axes = plt.subplots(
        len(pixel_trajectories),
        len(times),
        figsize=(2.3 * len(times), 2.4 * len(pixel_trajectories)),
        squeeze=False,
    )
    for row, (path_name, trajectory) in enumerate(pixel_trajectories.items()):
        display = postprocess(trajectory[:, sample_index]).detach().cpu()
        for column, time in enumerate(times):
            axes[row, column].imshow(_image_for_matplotlib(display[column]))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"t={float(time):.3f}")
        axes[row, 0].set_ylabel(PATH_LABELS[path_name], fontsize=11)
    figure.suptitle(f"Exact affine trajectories, noise sample {sample_index}")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_exact_trajectory_grid(
    trajectory: torch.Tensor,
    times: Sequence[float],
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    path_name: str,
    num_samples: int,
    spectral_ranks: Sequence[int] | None = None,
    dpi: int,
) -> None:
    """Plot several exact trajectories for one probability path."""
    sample_count = min(num_samples, trajectory.shape[1])
    display = postprocess(trajectory[:, :sample_count]).detach().cpu()
    figure, axes = plt.subplots(
        sample_count,
        len(times),
        figsize=(2.05 * len(times), 2.05 * sample_count),
        squeeze=False,
    )
    for row in range(sample_count):
        for column, time in enumerate(times):
            axes[row, column].imshow(_image_for_matplotlib(display[column, row]))
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                rank_label = (
                    f"j={spectral_ranks[column]}\n" if spectral_ranks is not None else ""
                )
                axes[row, column].set_title(
                    f"{rank_label}t={float(time):.4f}",
                    fontsize=9,
                )
        axes[row, 0].set_ylabel(f"seed {row}", fontsize=9)
    figure.suptitle(f"Exact affine trajectories - {PATH_LABELS[path_name]}")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_final_samples(
    samples: torch.Tensor,
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot the common exact endpoint of both affine paths."""
    display = postprocess(samples).detach().cpu()
    columns = min(8, display.shape[0])
    rows = int(np.ceil(display.shape[0] / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.1 * columns, 2.1 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index < display.shape[0]:
            axis.imshow(_image_for_matplotlib(display[index]))
    figure.suptitle("Common exact affine endpoint")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_euler_errors(
    results: Mapping[str, Mapping[str, object]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot knot-wise Euler error for each path and substep count."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for path_name, path_results in results.items():
        for solver_name, result in path_results.items():
            times = result.boundary_times.detach().cpu().numpy()
            errors = result.relative_boundary_error.detach().cpu().numpy()
            axis.plot(
                times,
                np.maximum(errors, 1e-16),
                label=f"{PATH_LABELS[path_name]}, {solver_name}",
            )
    axis.set_yscale("log")
    axis.set_xlabel("t")
    axis.set_ylabel("Relative spectral error")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_euler_comparison(
    exact_pixels: torch.Tensor,
    euler_pixels: Mapping[str, Mapping[str, torch.Tensor]],
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    sample_index: int,
    dpi: int,
) -> None:
    """Show exact, Euler, and raw absolute pixel error for each solver."""
    rows = sum(len(values) for values in euler_pixels.values())
    figure, axes = plt.subplots(rows, 3, figsize=(7.2, 2.35 * rows), squeeze=False)
    exact_display = postprocess(exact_pixels).detach().cpu()
    row = 0
    for path_name, values in euler_pixels.items():
        for solver_name, numerical_pixels in values.items():
            numerical_display = postprocess(numerical_pixels).detach().cpu()
            raw_difference = (
                numerical_pixels[sample_index] - exact_pixels[sample_index]
            ).abs().mean(dim=0).detach().cpu()
            axes[row, 0].imshow(_image_for_matplotlib(exact_display[sample_index]))
            axes[row, 1].imshow(_image_for_matplotlib(numerical_display[sample_index]))
            image = axes[row, 2].imshow(raw_difference.numpy(), cmap="magma")
            figure.colorbar(image, ax=axes[row, 2], fraction=0.046, pad=0.04)
            axes[row, 0].set_ylabel(
                f"{PATH_LABELS[path_name]}\n{solver_name}",
                fontsize=9,
            )
            row += 1
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0, 0].set_title("Exact")
    axes[0, 1].set_title("Euler")
    axes[0, 2].set_title("Mean |difference| over channels")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_sensitivity_rows(
    rows_by_path: Mapping[str, torch.Tensor],
    times: Sequence[float],
    output_prefix: Path,
    *,
    channels: int,
    height: int,
    width: int,
    target_y: int,
    target_x: int,
    dpi: int,
) -> None:
    """Plot raw per-channel rows and their spatial L2 aggregation."""
    reshaped = {
        name: rows.view(len(times), channels, height, width).detach().cpu()
        for name, rows in rows_by_path.items()
    }
    l2_fields = {
        name: values.square().sum(dim=1).sqrt()
        for name, values in reshaped.items()
    }
    l2_max = max(float(value.max()) for value in l2_fields.values())
    l2_max = max(l2_max, 1e-12)
    figure, axes = plt.subplots(
        len(reshaped),
        len(times),
        figsize=(2.1 * len(times), 2.2 * len(reshaped)),
        squeeze=False,
    )
    last_image = None
    for row, (path_name, fields) in enumerate(l2_fields.items()):
        for column, time in enumerate(times):
            last_image = axes[row, column].imshow(
                fields[column].numpy(),
                cmap="turbo",
                vmin=0.0,
                vmax=l2_max,
            )
            axes[row, column].scatter(target_x, target_y, c="red", s=8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"t={float(time):.3f}")
        axes[row, 0].set_ylabel(PATH_LABELS[path_name])
    if last_image is not None:
        figure.colorbar(last_image, ax=axes, fraction=0.018, pad=0.015)
    figure.suptitle("L2 sensitivity across input channels")
    _save_figure(figure, output_prefix.with_name(output_prefix.name + "_l2.png"), dpi)

    signed_max = max(float(value.abs().max()) for value in reshaped.values())
    signed_max = max(signed_max, 1e-12)
    figure, axes = plt.subplots(
        len(reshaped) * channels,
        len(times),
        figsize=(2.1 * len(times), 1.9 * len(reshaped) * channels),
        squeeze=False,
    )
    last_image = None
    for path_row, (path_name, fields) in enumerate(reshaped.items()):
        for channel in range(channels):
            row = path_row * channels + channel
            for column, time in enumerate(times):
                last_image = axes[row, column].imshow(
                    fields[column, channel].numpy(),
                    cmap="coolwarm",
                    vmin=-signed_max,
                    vmax=signed_max,
                )
                axes[row, column].scatter(target_x, target_y, c="red", s=7)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    axes[row, column].set_title(f"t={float(time):.3f}")
            axes[row, 0].set_ylabel(f"{PATH_LABELS[path_name]}\ninput c={channel}", fontsize=8)
    if last_image is not None:
        figure.colorbar(last_image, ax=axes, fraction=0.012, pad=0.012)
    figure.suptitle("Signed sensitivity by input channel")
    _save_figure(
        figure,
        output_prefix.with_name(output_prefix.name + "_channels.png"),
        dpi,
    )


def plot_effective_radii(
    radius_metrics: Mapping[str, Mapping[str, Sequence[float]]],
    times: Sequence[float],
    output_path: Path,
    *,
    tau: float,
    dpi: int,
) -> None:
    """Plot effective spatial sensitivity radii for all selected outputs."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for path_name, targets in radius_metrics.items():
        for target_name, radii in targets.items():
            axis.plot(
                times,
                radii,
                marker="o",
                markersize=3,
                label=f"{PATH_LABELS[path_name]} - {target_name}",
            )
    axis.set_xlabel("t")
    axis.set_ylabel(rf"Effective radius $r_q^{{{tau:g}}}(t)$")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_component_activation_magnitudes(
    magnitudes: Mapping[str, torch.Tensor],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot the exact coordinate change during each PC's main ramp."""
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for path_name, values in magnitudes.items():
        ranks = np.arange(1, values.numel() + 1)
        axis.plot(
            ranks,
            np.maximum(values.detach().cpu().numpy(), 1e-12),
            label=PATH_LABELS[path_name],
            linewidth=1.0,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("PCA component rank")
    axis.set_ylabel(r"$|z^k_{k/d} - z^k_{(k-1)/d}|$")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_moment_validation(
    moment_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot empirical errors on affine means, variances, and correlations."""
    keys = (
        "spectral_mean_error_over_total_std",
        "spectral_variance_relative_l2",
        "selected_spectral_off_diagonal_correlation_max_abs",
    )
    titles = (
        "Mean error / total standard deviation",
        "Relative variance error",
        "Maximum selected off-diagonal correlation",
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), squeeze=False)
    for path_name, values_by_time in moment_metrics.items():
        ordered = sorted((float(time), values) for time, values in values_by_time.items())
        times = [item[0] for item in ordered]
        for column, key in enumerate(keys):
            values = [max(float(item[1][key]), 1e-16) for item in ordered]
            axes[0, column].plot(
                times,
                values,
                marker="o",
                label=PATH_LABELS[path_name],
            )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
        axes[0, column].set_xlabel("t")
        axes[0, column].set_yscale("log")
        axes[0, column].grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_moment_sample_convergence(
    convergence_metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Show Monte-Carlo moment errors as the sample count increases."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    sample_counts = sorted(int(value) for value in convergence_metrics)
    for path_name in next(iter(convergence_metrics.values())):
        mean_errors = []
        variance_errors = []
        for sample_count in sample_counts:
            by_time = convergence_metrics[str(sample_count)][path_name]
            mean_errors.append(
                np.mean(
                    [
                        float(values["spectral_mean_error_over_total_std"])
                        for values in by_time.values()
                    ]
                )
            )
            variance_errors.append(
                np.mean(
                    [
                        float(values["spectral_variance_relative_l2"])
                        for values in by_time.values()
                    ]
                )
            )
        axes[0, 0].plot(sample_counts, mean_errors, marker="o", label=PATH_LABELS[path_name])
        axes[0, 1].plot(sample_counts, variance_errors, marker="o", label=PATH_LABELS[path_name])
    axes[0, 0].set_title("Mean error")
    axes[0, 1].set_title("Variance error")
    for axis in axes[0]:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Monte-Carlo samples")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


@torch.no_grad()
def plot_affine_slope_heatmap(
    models: Mapping[str, AffineSpectralVelocity],
    output_path: Path,
    *,
    time_points: int,
    dpi: int,
) -> None:
    """Plot the full component-time structure of the affine slope."""
    if time_points < 2:
        raise ValueError("time_points must be at least two.")
    first_model = next(iter(models.values()))
    device = first_model.lambda_sq.device
    dtype = first_model.lambda_sq.dtype
    if time_points == first_model.path.schedule.dimension:
        times = (
            torch.arange(time_points, device=device, dtype=dtype) + 0.5
        ) / time_points
    else:
        times = torch.linspace(0.0, 1.0, time_points, device=device, dtype=dtype)
    slopes: dict[str, np.ndarray] = {}
    for path_name, model in models.items():
        chunks = [model.coefficients(chunk).slope.cpu() for chunk in times.split(128)]
        slopes[path_name] = torch.cat(chunks).numpy().T

    finite_abs = np.concatenate(
        [np.abs(values[np.isfinite(values)]) for values in slopes.values()]
    )
    vmax = max(float(np.quantile(finite_abs, 0.995)), 1e-6)
    norm = SymLogNorm(linthresh=max(vmax * 1e-3, 1e-6), vmin=-vmax, vmax=vmax)
    figure, axes = plt.subplots(1, len(slopes), figsize=(7.2 * len(slopes), 6), squeeze=False)
    last_image = None
    for column, (path_name, values) in enumerate(slopes.items()):
        last_image = axes[0, column].imshow(
            values,
            aspect="auto",
            origin="upper",
            extent=(0.0, 1.0, values.shape[0], 1),
            cmap="coolwarm",
            norm=norm,
            interpolation="nearest",
        )
        axes[0, column].set_title(PATH_LABELS[path_name])
        axes[0, column].set_xlabel("t")
        axes[0, column].set_ylabel("PCA component rank")
    if last_image is not None:
        figure.colorbar(last_image, ax=axes, fraction=0.025, pad=0.03, label=r"$s_k(t)$")
    figure.suptitle("Affine spectral contraction and expansion")
    _save_figure(figure, output_path, dpi)


def plot_solver_final_errors(
    solver_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Plot final numerical errors against the number of field evaluations."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
    for path_name, metrics_by_solver in solver_metrics.items():
        ordered = sorted(
            (
                int(values["total_field_evaluations"]),
                solver_name,
                values,
            )
            for solver_name, values in metrics_by_solver.items()
            if solver_name != "exact_transitions"
        )
        evaluations = [item[0] for item in ordered]
        spectral = [max(float(item[2]["final_relative_spectral_error"]), 1e-16) for item in ordered]
        pixel = [max(float(item[2]["final_relative_pixel_error"]), 1e-16) for item in ordered]
        axes[0, 0].plot(evaluations, spectral, marker="o", label=PATH_LABELS[path_name])
        axes[0, 1].plot(evaluations, pixel, marker="o", label=PATH_LABELS[path_name])
        for evaluation, solver_name, values in ordered:
            short_name = solver_name.replace("uniform_", "U").replace("aligned_", "A")
            axes[0, 0].annotate(short_name, (evaluation, max(float(values["final_relative_spectral_error"]), 1e-16)), fontsize=7)
    axes[0, 0].set_title("Spectral relative error")
    axes[0, 1].set_title("Pixel relative error")
    for axis in axes[0]:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Field evaluations")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_data_vs_affine_histograms(
    data_coefficients: torch.Tensor,
    affine_coefficients: torch.Tensor,
    component_indices: Sequence[int],
    output_path: Path,
    *,
    bins: int,
    dpi: int,
) -> None:
    """Compare true and moment-matched Gaussian PCA marginals."""
    data = data_coefficients.detach().cpu().numpy()
    affine = affine_coefficients.detach().cpu().numpy()
    figure, axes = plt.subplots(
        len(component_indices),
        1,
        figsize=(9, 2.7 * len(component_indices)),
        squeeze=False,
    )
    for row, component_index in enumerate(component_indices):
        axis = axes[row, 0]
        axis.hist(data[:, row], bins=bins, density=True, histtype="step", linewidth=1.5, label="CIFAR-10")
        axis.hist(affine[:, row], bins=bins, density=True, histtype="step", linewidth=1.5, label="Affine Gaussian")
        axis.set_ylabel(f"PC {component_index + 1}\ndensity")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Non-centered PCA coefficient")
    figure.suptitle("Data marginals versus affine Gaussian endpoint")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_data_vs_affine_samples(
    real_samples: torch.Tensor,
    affine_samples: torch.Tensor,
    postprocess: Callable[[torch.Tensor], torch.Tensor],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """Show real images and moment-matched affine endpoint samples."""
    count = min(real_samples.shape[0], affine_samples.shape[0], 16)
    real = postprocess(real_samples[:count]).detach().cpu()
    affine = postprocess(affine_samples[:count]).detach().cpu()
    columns = min(8, count)
    groups = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        2 * groups,
        columns,
        figsize=(1.8 * columns, 3.65 * groups),
        squeeze=False,
    )
    for index in range(groups * columns):
        group = index // columns
        column = index % columns
        real_axis = axes[2 * group, column]
        affine_axis = axes[2 * group + 1, column]
        if index < count:
            real_axis.imshow(_image_for_matplotlib(real[index]))
            affine_axis.imshow(_image_for_matplotlib(affine[index]))
        for axis in (real_axis, affine_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            if index >= count:
                axis.set_axis_off()
        if column == 0:
            real_axis.set_ylabel("CIFAR-10", fontsize=9)
            affine_axis.set_ylabel("Affine", fontsize=9)
    figure.suptitle("True data versus moment-matched Gaussian samples")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def plot_component_spatial_dynamics(
    U: torch.Tensor,
    activation_payload: Mapping[str, object],
    component_indices: Sequence[int],
    output_path: Path,
    *,
    path_name: str,
    channels: int,
    height: int,
    width: int,
    dpi: int,
) -> None:
    """Show spatial PC contributions and changes during their main ramps."""
    path_payload = activation_payload[path_name]
    if not isinstance(path_payload, Mapping):
        raise TypeError("Invalid component activation payload.")
    end_coordinate = path_payload["end_coordinate"]
    ramp_delta = path_payload["main_ramp_delta"]
    if not isinstance(end_coordinate, torch.Tensor) or not isinstance(ramp_delta, torch.Tensor):
        raise TypeError("Component activation coordinates must be tensors.")

    basis = U.detach().cpu()
    figure, axes = plt.subplots(
        2,
        len(component_indices),
        figsize=(2.3 * len(component_indices), 4.8),
        squeeze=False,
    )
    for column, component_index in enumerate(component_indices):
        end_map = (
            basis[:, component_index] * end_coordinate[component_index]
        ).view(channels, height, width)
        delta_map = (
            basis[:, component_index] * ramp_delta[component_index]
        ).view(channels, height, width)
        axes[0, column].imshow(_signed_component_for_matplotlib(end_map))
        axes[1, column].imshow(_signed_component_for_matplotlib(delta_map))
        axes[0, column].set_title(f"PC {component_index + 1}")
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    axes[0, 0].set_ylabel("End contribution", fontsize=9)
    axes[1, 0].set_ylabel("Main-ramp change", fontsize=9)
    figure.suptitle(f"Spatial component dynamics - {PATH_LABELS[path_name]}")
    figure.tight_layout()
    _save_figure(figure, output_path, dpi)


def _selected_schedule_curves(
    schedule: SequentialLinearSchedule,
    indices: Sequence[int],
    *,
    points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = _curve_times(schedule.dimension, indices, points, device, dtype)
    beta_chunks = []
    derivative_chunks = []
    index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
    for chunk in times.split(512):
        beta_chunks.append(schedule.beta(chunk).index_select(1, index_tensor).cpu())
        derivative_chunks.append(
            schedule.dot_beta(chunk).index_select(1, index_tensor).cpu()
        )
    return (
        times.cpu().numpy(),
        torch.cat(beta_chunks).numpy(),
        torch.cat(derivative_chunks).numpy(),
    )


def _selected_affine_curves(
    model: AffineSpectralVelocity,
    indices: Sequence[int],
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = model.lambda_sq.device
    dtype = model.lambda_sq.dtype
    times = _curve_times(model.path.schedule.dimension, indices, points, device, dtype)
    index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
    variance_chunks = []
    slope_chunks = []
    bias_chunks = []
    for chunk in times.split(512):
        coefficients = model.coefficients(chunk)
        variance_chunks.append(coefficients.variance.index_select(1, index_tensor).cpu())
        slope_chunks.append(coefficients.slope.index_select(1, index_tensor).cpu())
        bias_chunks.append(coefficients.bias.index_select(1, index_tensor).cpu())
    return (
        times.cpu().numpy(),
        torch.cat(variance_chunks).numpy(),
        torch.cat(slope_chunks).numpy(),
        torch.cat(bias_chunks).numpy(),
    )


def _curve_times(
    dimension: int,
    indices: Sequence[int],
    points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, points, device=device, dtype=dtype)
    event_times = []
    for index in indices:
        start = index / dimension
        end = (index + 1) / dimension
        event_times.extend([start, 0.5 * (start + end), end])
    events = torch.as_tensor(event_times, device=device, dtype=dtype)
    return torch.unique(torch.cat([base, events]), sorted=True)


def _image_for_matplotlib(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Expected an image tensor with shape [C, H, W].")
    if image.shape[0] == 1:
        return image[0].numpy()
    return image.permute(1, 2, 0).numpy()


def _signed_component_for_matplotlib(component: torch.Tensor) -> np.ndarray:
    """Map a signed spatial component to RGB around neutral gray."""
    scale = max(float(component.abs().max()), 1e-12)
    normalized = (0.5 + 0.5 * component / scale).clamp(0.0, 1.0)
    if normalized.shape[0] == 1:
        return normalized[0].numpy()
    return normalized.permute(1, 2, 0).numpy()


def _save_figure(figure: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
