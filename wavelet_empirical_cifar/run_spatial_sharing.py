"""Study 2: same-band posterior with fixed, equal-size shared, or full shared banks."""

from __future__ import annotations

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from wavelet_empirical_cifar.common import (
    TIMES, make_model, mse, parser, prepare, save_generation_grid, save_json,
    save_rows, state_and_targets,
)
from wavelet_empirical_cifar.shared import predict_subband
from wavelet_fm_structure.rgb_wavelets import analyze_rgb, flatten_rgb, synthesize_rgb, unflatten_rgb
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import model_wavelet_state_and_velocity


MODES = ("fixed", "shared_equal", "shared_full")


def plot_metrics(rows, output, sampled):
    selected = [row for row in rows if row["split"] == "evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in dict.fromkeys(row["variant"] for row in selected):
        records = [row for row in selected if row["variant"] == mode]
        axes[0].plot([row["time"] for row in records],
                     [row["clean_mse"] for row in records], marker="o", label=mode)
        axes[1].plot([row["time"] for row in records],
                     [row["dominant_source_fraction"] for row in records], marker="o", label=mode)
    axes[0].set(xlabel="Flow time t", ylabel="Clean-coefficient MSE, test",
                title="Selected sites" if sampled else "All wavelet sites")
    axes[1].set(xlabel="Flow time t", ylabel="Fraction of sites with modal source",
                title="Source-image concentration")
    for axis in axes:
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def predict_selected(state, references, layout, time, mode, device, radius, source_chunk,
                     stride, equal_seed):
    dimension = layout.num_coefficients
    query_y = state.reshape(len(state), 3, dimension)
    reference_y = references.reshape(len(references), 3, dimension)
    prediction = np.full_like(query_y, np.nan)
    count = 0
    weighted_mass = 0.0
    source_ids = []
    for band in layout.subbands:
        height, width = band.shape
        if 2*radius + 1 > min(band.shape):
            raise ValueError(f"radius {radius} aliases {band.shape} band")
        selected = tuple(range(0, band.size, stride))
        query_band = query_y[:, :, band.flat_slice].reshape(len(state), 3, height, width)
        ref_band = reference_y[:, :, band.flat_slice].reshape(len(references), 3, height, width)
        start = perf_counter()
        result, concentration = predict_subband(query_band, ref_band, time,
            radius=radius, mode=mode, device=device, source_chunk=source_chunk,
            target_positions=selected, equal_seed=equal_seed)
        prediction[:, :, band.flat_slice] = result.reshape(len(state), 3, band.size)
        count += concentration.num_targets
        weighted_mass += concentration.mean_max_source_mass*concentration.num_targets
        source_ids.append(concentration.dominant_source_ids)
        print(f"  {mode} band={band.subband_index} targets={len(selected)} "
              f"candidates={len(references)*(band.size if mode == 'shared_full' else 1)} "
              f"seconds={perf_counter()-start:.1f}", flush=True)
    all_ids = np.concatenate(source_ids, axis=1)
    dominant_fraction = np.mean([
        np.bincount(row, minlength=len(references)).max() / len(row)
        for row in all_ids
    ])
    return prediction.reshape(len(state), -1), {
        "source_max_mass": weighted_mass/count,
        "dominant_source_fraction": float(dominant_fraction),
        "evaluated_rgb_sites": count // len(state),
    }


def selected_columns(layout, stride):
    sites = np.concatenate([np.arange(b.flat_start, b.flat_stop, stride) for b in layout.subbands])
    return np.concatenate([channel*layout.num_coefficients + sites for channel in range(3)])


def generate(references, layout, transform, initial, mode, device, steps, radius,
             source_chunk, equal_seed):
    initial_y, check = analyze_rgb(initial, transform)
    if check != layout:
        raise RuntimeError("Generation DWT mismatch")
    state = flatten_rgb(initial_y)
    for step in range(steps):
        time = step / steps
        clean, _ = predict_selected(state, references, layout, time, mode, device,
                                    radius, source_chunk, 1, equal_seed)
        state += (clean - state) / ((1-time)*steps)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite Euler state at step {step}")
        print(f"Euler {mode} {step+1}/{steps}", flush=True)
    return synthesize_rgb(unflatten_rgb(state, layout), layout, transform)


def main():
    argp = parser(__doc__, "wavelet_empirical_cifar/results/spatial_sharing")
    argp.add_argument("--radius", type=int, default=1)
    argp.add_argument("--target-stride", type=int, default=1,
                      help="Explicit site subsampling for screening; 1 evaluates all")
    argp.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    argp.add_argument("--benchmark-only", action="store_true",
                      help="Time one fine-band target per mode, then stop")
    args = argp.parse_args()
    if args.target_stride < 1 or args.radius < 0:
        raise ValueError("target stride and radius must be nonnegative")
    device, transform, references, layout, clean, noise, generation_noise = prepare(args)
    if args.benchmark_only:
        state, _ = state_and_targets(clean["calibration"][:1], noise["calibration"][:1],
                                     .5, transform)
        band = layout.subbands[-1]
        d = layout.num_coefficients
        query = state.reshape(1, 3, d)[:, :, band.flat_slice].reshape(1, 3, *band.shape)
        ref = references.reshape(len(references), 3, d)[:, :, band.flat_slice].reshape(
            len(references), 3, *band.shape)
        benchmark = {}
        for mode in args.modes:
            started = perf_counter()
            predict_subband(query, ref, .5, radius=args.radius, mode=mode,
                device=device, source_chunk=args.reference_chunk, target_positions=(0,),
                equal_seed=args.seed)
            benchmark[mode] = {"seconds_for_one_query_one_fine_band_target": perf_counter()-started,
                "candidates": len(references)*(band.size if mode == "shared_full" else 1)}
        save_json(args.out_dir / "benchmark.json", benchmark)
        print(benchmark, flush=True)
        return
    model = make_model(device, not args.no_unet)
    columns = selected_columns(layout, args.target_stride)
    rows = []
    for split in ("calibration", "evaluation"):
        for time in TIMES:
            state, target = state_and_targets(clean[split], noise[split], time, transform)
            model_target = None
            if model is not None:
                _, velocity = model_wavelet_state_and_velocity(model, clean[split],
                    noise[split], time, transform, device, 32)
                model_target = state + (1-time)*velocity
            for mode in args.modes:
                started = perf_counter()
                prediction, concentration = predict_selected(state, references, layout,
                    time, mode, device, args.radius, args.reference_chunk,
                    args.target_stride, args.seed)
                record = dict(split=split, time=time, variant=mode,
                    clean_mse=mse(prediction[:, columns], target[:, columns]),
                    seconds=perf_counter()-started, **concentration)
                if model_target is not None:
                    record.update(unet_clean_mse=mse(model_target[:, columns], target[:, columns]),
                                  agreement_mse=mse(prediction[:, columns], model_target[:, columns]))
                rows.append(record)
                save_rows(args.out_dir / "metrics.csv", rows)
                print(f"{split} t={time:.2f} {mode}: clean MSE={record['clean_mse']:.6f}", flush=True)
    plot_metrics(rows, args.out_dir / "mse_and_source_concentration.png",
                 args.target_stride != 1)
    if not args.no_generation:
        generated = {mode: generate(references, layout, transform, generation_noise,
            mode, device, args.steps, args.radius, args.reference_chunk, args.seed)
            for mode in args.modes}
        save_generation_grid(generated, references, transform, layout, args.out_dir, device)
    save_json(args.out_dir / "interpretation.json", {
        "question": "Does sharing patch candidates across positions help at fixed radius?",
        "target_stride": args.target_stride,
        "mse_scope": "all wavelet sites" if args.target_stride == 1 else "selected wavelet sites only",
        "fixed_and_shared_equal_candidates_per_site": len(references),
        "shared_full_candidates_per_site": "reference_count * band_site_count",
        "same_band_only": True,
        "translation_equivariance_claim": False,
    })


if __name__ == "__main__":
    main()
