"""Evaluate exact and explicit-Euler affine generation on sequential paths.

The report is intentionally restricted to four outputs: matched final samples,
endpoint MSE versus Euler budget, MSE throughout time, and FID. Both spectral
paths use the same PCA, signal schedule, and initial Gaussian noises.

Input: ``--config`` YAML specifying CIFAR-10, a PCA cache and solver budgets.
Outputs: sample grids, endpoint/time MSE and optional FID under ``--output-root``.
Example (requires the external upstream dataset/config runtime)::

    python -m spectral_paths.sequential_linear_schedule.run_affine_experiment --pilot --skip-fid
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from local_diffusion.data import DatasetBundle  # noqa: E402
from spectral_paths.sequential_linear_schedule.affine import (  # noqa: E402
    AffineSpectralVelocity,
)
from spectral_paths.sequential_linear_schedule.euler import (  # noqa: E402
    AffineEulerMapResult,
    DiagonalAffineMap,
    exact_affine_map,
    integrate_classic_fm_wiener_legacy_map,
    integrate_uniform_affine_map,
)
from spectral_paths.sequential_linear_schedule.fid import (  # noqa: E402
    evaluate_affine_maps_fid,
)
from spectral_paths.sequential_linear_schedule.paths import (  # noqa: E402
    SpectralPath,
)
from spectral_paths.sequential_linear_schedule.pca import (  # noqa: E402
    PCABasis,
    prepare_population_pca,
)
from spectral_paths.sequential_linear_schedule.reporting import (  # noqa: E402
    plot_fid_comparison,
    plot_final_mse_vs_steps,
    plot_final_sample_grid,
    plot_mse_vs_time,
)
from spectral_paths.sequential_linear_schedule.runtime import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    create_run_paths,
    load_affine_config,
    load_dataset_from_config,
    seed_everything,
    select_device,
    torch_dtype,
)
from spectral_paths.sequential_linear_schedule.schedule import (  # noqa: E402
    SequentialLinearSchedule,
)


def parse_args() -> argparse.Namespace:
    """Parse the YAML configuration and optional OmegaConf overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a lightweight CPU-compatible check and skip FID.",
    )
    parser.add_argument(
        "--skip-fid",
        action="store_true",
        help="Produce the three numerical figures without running Inception.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured result root.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, for example schedule.epsilon_1=0.1.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the four-part affine solver and generation report."""
    args = parse_args()
    cfg = load_affine_config(args.config, args.overrides, pilot=args.pilot)
    if args.output_root is not None:
        cfg.experiment.output_root = str(args.output_root.resolve())
    if args.skip_fid:
        cfg.fid.enabled = False

    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    device = select_device(cfg)
    dtype = torch_dtype(str(cfg.experiment.dtype))
    started_at = time.perf_counter()

    dataset = load_dataset_from_config(cfg)
    if dataset.name.lower() != "cifar10":
        raise ValueError("This experiment is currently restricted to CIFAR-10.")
    dimension = dataset.in_channels * dataset.resolution**2
    basis = prepare_population_pca(
        dataset,
        Path(cfg.pca.cache_dir),
        device=device,
        dtype=dtype,
        seed=seed,
        force_recompute=bool(cfg.pca.force_recompute),
        source_wiener_dir=Path(cfg.pca.source_wiener_dir),
        reuse_sample_covariance_cache=bool(cfg.pca.reuse_sample_covariance_cache),
    ).to(device, dtype)
    schedule = SequentialLinearSchedule(
        dimension=dimension,
        epsilon_1=float(cfg.schedule.epsilon_1),
    )
    models = {
        path_name: AffineSpectralVelocity(
            path=SpectralPath(schedule=schedule, kind=path_name),
            lambda_sq=basis.lambda_sq,
            mu_z=basis.mu_z,
        )
        for path_name in ("coupled", "isotropic")
    }

    report_steps = _resolve_step_specs(cfg.evaluation.report_step_counts, dimension)
    final_error_steps = _resolve_step_specs(
        cfg.evaluation.final_error_step_counts,
        dimension,
    )
    time_curve_steps = _resolve_step_specs(
        cfg.evaluation.time_curve_step_counts,
        dimension,
    )
    classic_wiener_steps = _resolve_step_specs(
        cfg.evaluation.classic_wiener_step_counts,
        dimension,
    )
    _require_subset("report_step_counts", report_steps, final_error_steps)
    _require_subset("time_curve_step_counts", time_curve_steps, final_error_steps)
    required_steps = sorted(final_error_steps)

    run_paths = create_run_paths(cfg)
    OmegaConf.save(cfg, run_paths.config)
    dpi = int(cfg.plot.dpi)
    print(f"dataset: {dataset.name} ({len(dataset.dataset):,} {dataset.split} images)")
    print(f"device: {device}, dtype: {dtype}, dimension: {dimension}")
    print(f"epsilon_1: {schedule.epsilon_1}, PCA source: {basis.metadata['method']}")
    print(f"report Euler budgets: {[ _step_label(value, dimension) for value in report_steps ]}")
    print(f"classic FM Wiener reference budgets: {classic_wiener_steps}")
    print(f"run directory: {run_paths.run_dir}")

    mse_sample_count = int(cfg.evaluation.mse_num_samples)
    noise_seed = int(cfg.evaluation.noise_seed)
    mse_z0 = _standard_normal(
        mse_sample_count,
        dimension,
        device=device,
        dtype=dtype,
        seed=noise_seed,
    )
    noise_mean = mse_z0.mean(dim=0)
    noise_second_moment = mse_z0.square().mean(dim=0)
    exact_map = exact_affine_map(models["coupled"], 1.0)
    other_exact_map = exact_affine_map(models["isotropic"], 1.0)
    exact_path_difference = max(
        float((exact_map.scale - other_exact_map.scale).abs().max().cpu()),
        float((exact_map.bias - other_exact_map.bias).abs().max().cpu()),
    )
    exact_mse_state = exact_map.apply(mse_z0)

    classic_wiener_maps: dict[int, DiagonalAffineMap] = {}
    classic_wiener_mse_statistics: dict[int, dict[str, float]] = {}
    classic_wiener_mse_per_sample: dict[int, torch.Tensor] = {}
    for num_steps in classic_wiener_steps:
        print(
            "Integrating classic FM Wiener reference with legacy midpoint "
            f"sampler: N={num_steps}"
        )
        affine_map = integrate_classic_fm_wiener_legacy_map(
            basis.lambda_sq,
            basis.mu_z,
            num_steps,
        )
        _require_finite_map(affine_map, "classic_fm_wiener", num_steps)
        classic_wiener_maps[num_steps] = affine_map
        errors = (affine_map.apply(mse_z0) - exact_mse_state).square().mean(dim=1)
        classic_wiener_mse_per_sample[num_steps] = errors.detach().cpu()
        classic_wiener_mse_statistics[num_steps] = _sample_statistics(errors)

    solver_results: dict[str, dict[int, AffineEulerMapResult]] = {
        path_name: {} for path_name in models
    }
    final_mse_statistics: dict[str, dict[int, dict[str, float]]] = {
        path_name: {} for path_name in models
    }
    final_mse_per_sample: dict[str, dict[int, torch.Tensor]] = {
        path_name: {} for path_name in models
    }
    time_curve_step_set = set(time_curve_steps)

    for path_name, model in models.items():
        for num_steps in required_steps:
            observations = (
                _evenly_spaced_steps(
                    num_steps,
                    int(cfg.evaluation.time_curve_max_points),
                )
                if num_steps in time_curve_step_set
                else None
            )
            print(
                f"Integrating affine Euler map: path={path_name}, "
                f"{_step_label(num_steps, dimension)}"
            )
            result = integrate_uniform_affine_map(
                model,
                num_steps,
                observation_steps=observations,
                noise_mean=(noise_mean if observations is not None else None),
                noise_second_moment=(
                    noise_second_moment if observations is not None else None
                ),
                time_chunk_size=int(cfg.evaluation.euler_time_chunk_size),
            )
            _require_finite_map(result.affine_map, path_name, num_steps)
            solver_results[path_name][num_steps] = result

            errors = (
                result.affine_map.apply(mse_z0) - exact_mse_state
            ).square().mean(dim=1)
            final_mse_per_sample[path_name][num_steps] = errors.detach().cpu()
            final_mse_statistics[path_name][num_steps] = _sample_statistics(errors)

    visual_sample_count = int(cfg.evaluation.visual_num_samples)
    visual_noise_seed = int(cfg.evaluation.visual_noise_seed)
    visual_z0 = _legacy_pixel_noise_as_spectral(
        visual_sample_count,
        basis=basis,
        dataset=dataset,
        device=device,
        dtype=dtype,
        seed=visual_noise_seed,
    )
    visual_samples: dict[str, torch.Tensor] = {
        "exact": _to_images(exact_map.apply(visual_z0), basis, dataset),
    }
    sample_rows: list[tuple[str, str]] = [
        ("exact", "Exact solution\n(common endpoint)"),
    ]
    for num_steps in classic_wiener_steps:
        key = f"classic_fm_wiener_{num_steps}"
        visual_samples[key] = _to_images(
            classic_wiener_maps[num_steps].apply(visual_z0),
            basis,
            dataset,
        )
        sample_rows.append(
            (
                key,
                f"Classic FM Wiener\nlegacy sampler, N={num_steps:,}",
            )
        )
    for path_name in ("coupled", "isotropic"):
        for num_steps in report_steps:
            key = _method_key(path_name, num_steps)
            visual_samples[key] = _to_images(
                solver_results[path_name][num_steps].affine_map.apply(visual_z0),
                basis,
                dataset,
            )
            sample_rows.append(
                (
                    key,
                    f"{_short_path_label(path_name)}\nEuler, "
                    f"{_step_label(num_steps, dimension)}",
                )
            )

    plot_final_sample_grid(
        sample_rows,
        visual_samples,
        dataset.postprocess,
        run_paths.images / "final_samples.png",
        dpi=dpi,
    )
    plot_final_mse_vs_steps(
        {
            path_name: {
                num_steps: final_mse_statistics[path_name][num_steps]
                for num_steps in final_error_steps
            }
            for path_name in models
        },
        run_paths.images / "final_mse_vs_steps.png",
        batch_size=mse_sample_count,
        dimension=dimension,
        dpi=dpi,
    )

    time_curves: dict[str, dict[int, dict[str, torch.Tensor | str]]] = {
        path_name: {} for path_name in models
    }
    for path_name in models:
        for num_steps in time_curve_steps:
            result = solver_results[path_name][num_steps]
            if result.observation_mse is None:
                raise RuntimeError("Missing MSE observations for a requested curve.")
            time_curves[path_name][num_steps] = {
                "times": result.observation_times.detach().cpu(),
                "mse": result.observation_mse.detach().cpu(),
                "label": _step_label(num_steps, dimension),
            }
    plot_mse_vs_time(
        time_curves,
        run_paths.images / "mse_vs_time.png",
        batch_size=mse_sample_count,
        dimension=dimension,
        dpi=dpi,
    )

    metrics: dict[str, Any] = {
        "dataset": {
            "name": dataset.name,
            "split": dataset.split,
            "num_samples": len(dataset.dataset),
            "resolution": dataset.resolution,
            "channels": dataset.in_channels,
        },
        "pca": basis.metadata,
        "schedule": {
            "dimension": dimension,
            "epsilon_1": schedule.epsilon_1,
            "rho": schedule.rho,
        },
        "solver": {
            "method": "uniform_explicit_euler",
            "report_step_counts": report_steps,
            "final_error_step_counts": final_error_steps,
            "time_curve_step_counts": time_curve_steps,
            "time_curve_max_points": int(cfg.evaluation.time_curve_max_points),
            "map_time_chunk_size": int(cfg.evaluation.euler_time_chunk_size),
            "exact_endpoint_max_difference_between_paths": exact_path_difference,
            "classic_fm_wiener_reference": {
                "method": "legacy_midpoint_field_evaluation",
                "step_counts": classic_wiener_steps,
                "matched_noise_convention": "epsilon_pixel_equals_U_times_z0_spectral",
            },
        },
        "mse": {
            "definition": "mean_over_samples_and_spectral_coordinates_of_squared_raw_error",
            "num_noise_samples": mse_sample_count,
            "noise_seed": noise_seed,
            "orthogonal_pixel_space_equivalence": True,
            "final": {
                path_name: {
                    str(num_steps): final_mse_statistics[path_name][num_steps]
                    for num_steps in final_error_steps
                }
                for path_name in models
            },
            "classic_fm_wiener_reference": {
                str(num_steps): classic_wiener_mse_statistics[num_steps]
                for num_steps in classic_wiener_steps
            },
        },
        "visual_samples": {
            "num_samples": visual_sample_count,
            "noise_seed": visual_noise_seed,
            "noise_convention": (
                "legacy_pixel_epsilon_then_z0_spectral_equals_epsilon_times_U"
            ),
            "outside_model_range_fraction": {
                key: _outside_model_range_fraction(value)
                for key, value in visual_samples.items()
            },
        },
        "fid": {
            "status": "pending" if bool(cfg.fid.enabled) else "skipped",
        },
    }
    _save_raw_results(
        run_paths.tensors / "affine_solver_results.pt",
        visual_z0=visual_z0,
        exact_map=exact_map,
        classic_wiener_maps=classic_wiener_maps,
        solver_results=solver_results,
        final_mse_per_sample=final_mse_per_sample,
        classic_wiener_mse_per_sample=classic_wiener_mse_per_sample,
        time_curve_steps=time_curve_steps,
    )
    _save_metrics(metrics, run_paths.metrics)

    if bool(cfg.fid.enabled):
        fid_maps: dict[str, DiagonalAffineMap] = {"exact": exact_map}
        fid_methods: list[tuple[str, str, str]] = [
            ("exact", "Exact\n(common)", "exact"),
        ]
        for path_name in ("coupled", "isotropic"):
            for num_steps in report_steps:
                key = _method_key(path_name, num_steps)
                fid_maps[key] = solver_results[path_name][num_steps].affine_map
                fid_methods.append(
                    (
                        key,
                        f"{_short_path_label(path_name)}\n"
                        f"{_compact_step_label(num_steps, dimension)}",
                        path_name,
                    )
                )

        weights_value = cfg.fid.feature_extractor_weights_path
        fid_evaluation = evaluate_affine_maps_fid(
            fid_maps,
            basis=basis,
            dataset=dataset,
            postprocess=dataset.postprocess,
            num_generated=int(cfg.fid.num_generated),
            num_real=int(cfg.fid.num_real),
            batch_size=int(cfg.fid.batch_size),
            feature=int(cfg.fid.feature),
            seed=int(cfg.fid.seed),
            device=device,
            dtype=dtype,
            real_statistics_cache=Path(cfg.fid.real_statistics_cache),
            feature_extractor_weights_path=(
                Path(str(weights_value)) if weights_value is not None else None
            ),
        )
        plot_fid_comparison(
            fid_methods,
            fid_evaluation.values,
            run_paths.images / "fid_comparison.png",
            num_generated=int(cfg.fid.num_generated),
            num_real=int(cfg.fid.num_real),
            dpi=dpi,
        )
        metrics["fid"] = {
            "status": "completed",
            "implementation": "torchmetrics_FrechetInceptionDistance_torch_fidelity",
            "feature": int(cfg.fid.feature),
            "num_generated": int(cfg.fid.num_generated),
            "num_real": int(cfg.fid.num_real),
            "generated_seed": int(cfg.fid.seed),
            "real_statistics_source": fid_evaluation.real_statistics_source,
            "real_statistics_cache": str(cfg.fid.real_statistics_cache),
            "values": fid_evaluation.values,
            "outside_model_range_fraction": fid_evaluation.clipping_fractions,
        }

    metrics["elapsed_seconds"] = time.perf_counter() - started_at
    _save_metrics(metrics, run_paths.metrics)
    print(f"Saved affine report to {run_paths.run_dir}")
    print(f"Elapsed: {metrics['elapsed_seconds'] / 60.0:.2f} minutes")


def _resolve_step_specs(specifications: Sequence[Any], dimension: int) -> list[int]:
    """Resolve integers and strings such as ``2D`` into Euler step counts."""
    resolved: list[int] = []
    for specification in specifications:
        if isinstance(specification, bool):
            raise ValueError("Euler step counts cannot be booleans.")
        if isinstance(specification, int):
            value = specification
        else:
            text = str(specification).strip().upper().replace(" ", "")
            if text.endswith("D"):
                multiplier_text = text[:-1]
                multiplier = 1 if multiplier_text == "" else int(multiplier_text)
                value = multiplier * dimension
            else:
                value = int(text)
        if value <= 0:
            raise ValueError("Every Euler step count must be positive.")
        if value in resolved:
            raise ValueError(f"Duplicate resolved Euler step count: {value}.")
        resolved.append(value)
    if not resolved:
        raise ValueError("At least one Euler step count is required.")
    return resolved


def _evenly_spaced_steps(num_steps: int, max_points: int) -> list[int]:
    """Select trajectory steps without changing the underlying integration."""
    if max_points < 2:
        raise ValueError("time_curve_max_points must be at least two.")
    point_count = min(num_steps + 1, max_points)
    return sorted(
        set(
            int(value)
            for value in torch.linspace(0, num_steps, point_count).round().tolist()
        )
    )


def _sample_statistics(values: torch.Tensor) -> dict[str, float]:
    """Summarize one scalar error per matched initial noise."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Expected a non-empty vector of per-sample values.")
    standard_deviation = values.std(unbiased=values.numel() > 1)
    standard_error = standard_deviation / math.sqrt(values.numel())
    return {
        "mean": float(values.mean().cpu()),
        "standard_deviation": float(standard_deviation.cpu()),
        "standard_error": float(standard_error.cpu()),
        "median": float(values.median().cpu()),
        "minimum": float(values.min().cpu()),
        "maximum": float(values.max().cpu()),
    }


def _standard_normal(
    num_samples: int,
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("The number of Gaussian samples must be positive.")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        num_samples,
        dimension,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _legacy_pixel_noise_as_spectral(
    num_samples: int,
    *,
    basis: PCABasis,
    dataset: DatasetBundle,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Reproduce the old pixel-noise draw and express it in PCA coordinates.

    The historical generation script sampled ``[B, C, H, W]`` tensors in the
    pixel basis. Drawing that tensor first, then applying ``z0 = epsilon U``,
    preserves those samplewise initial conditions for the spectral paths.
    """
    if num_samples <= 0:
        raise ValueError("The number of visual Gaussian samples must be positive.")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pixel_noise = torch.randn(
        num_samples,
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
        generator=generator,
        device=device,
        dtype=dtype,
    ).flatten(start_dim=1)
    return basis.to_spectral(pixel_noise)


def _to_images(z: torch.Tensor, basis, dataset) -> torch.Tensor:
    return basis.to_pixel(z).view(
        z.shape[0],
        dataset.in_channels,
        dataset.resolution,
        dataset.resolution,
    )


def _outside_model_range_fraction(images: torch.Tensor) -> float:
    return float(((images < -1.0) | (images > 1.0)).float().mean().cpu())


def _require_finite_map(
    affine_map: DiagonalAffineMap,
    path_name: str,
    num_steps: int,
) -> None:
    if not torch.isfinite(affine_map.scale).all() or not torch.isfinite(
        affine_map.bias
    ).all():
        raise RuntimeError(
            f"Euler produced a non-finite affine map for path={path_name}, N={num_steps}."
        )


def _save_raw_results(
    output_path: Path,
    *,
    visual_z0: torch.Tensor,
    exact_map: DiagonalAffineMap,
    classic_wiener_maps: dict[int, DiagonalAffineMap],
    solver_results: dict[str, dict[int, AffineEulerMapResult]],
    final_mse_per_sample: dict[str, dict[int, torch.Tensor]],
    classic_wiener_mse_per_sample: dict[int, torch.Tensor],
    time_curve_steps: list[int],
) -> None:
    """Save compact sufficient statistics behind the report figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "visual_z0": visual_z0.detach().cpu(),
            "exact_map": {
                "scale": exact_map.scale.detach().cpu(),
                "bias": exact_map.bias.detach().cpu(),
            },
            "classic_fm_wiener_maps": {
                str(num_steps): {
                    "scale": affine_map.scale.detach().cpu(),
                    "bias": affine_map.bias.detach().cpu(),
                }
                for num_steps, affine_map in classic_wiener_maps.items()
            },
            "euler_maps": {
                path_name: {
                    str(num_steps): {
                        "scale": result.affine_map.scale.detach().cpu(),
                        "bias": result.affine_map.bias.detach().cpu(),
                    }
                    for num_steps, result in path_results.items()
                }
                for path_name, path_results in solver_results.items()
            },
            "final_mse_per_sample": final_mse_per_sample,
            "classic_fm_wiener_mse_per_sample": classic_wiener_mse_per_sample,
            "time_curves": {
                path_name: {
                    str(num_steps): {
                        "steps": path_results[num_steps].observation_steps.detach().cpu(),
                        "times": path_results[num_steps].observation_times.detach().cpu(),
                        "mse": path_results[num_steps].observation_mse.detach().cpu(),
                    }
                    for num_steps in time_curve_steps
                }
                for path_name, path_results in solver_results.items()
            },
        },
        output_path,
    )


def _save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _require_subset(name: str, values: list[int], available: list[int]) -> None:
    missing = sorted(set(values) - set(available))
    if missing:
        raise ValueError(f"{name} contains unavailable step counts: {missing}")


def _method_key(path_name: str, num_steps: int) -> str:
    return f"{path_name}_euler_{num_steps}"


def _short_path_label(path_name: str) -> str:
    return "Coupled path" if path_name == "coupled" else "Isotropic path"


def _step_label(num_steps: int, dimension: int) -> str:
    if num_steps % dimension == 0:
        multiplier = num_steps // dimension
        symbolic = "D" if multiplier == 1 else f"{multiplier}D"
        return f"N={symbolic}={num_steps:,}"
    return f"N={num_steps:,}"


def _compact_step_label(num_steps: int, dimension: int) -> str:
    if num_steps % dimension == 0:
        multiplier = num_steps // dimension
        symbolic = "D" if multiplier == 1 else f"{multiplier}D"
        return f"N={symbolic}"
    return f"N={num_steps:,}"


if __name__ == "__main__":
    main()
