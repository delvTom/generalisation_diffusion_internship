"""Compare Wiener-induced and geometric wavelet masks on CIFAR-10.

Inputs: training-image candidate bank and disjoint test queries; an FM U-Net
checkpoint is needed unless ``--no-unet`` is explicit. Outputs: mask sizes,
MSE tables and optional Euler samples in ``--out-dir``. Example::

    python -m wavelet_empirical_cifar.run_wiener_masks --device cuda --artem-threshold 0.05
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from wavelet_empirical_cifar.common import (
    TIMES, evaluate_estimator, generate_estimator, make_estimator, make_model,
    parser, plot_mse, prepare, save_generation_grid, save_json, save_rows,
    state_and_targets, mse,
)
from wavelet_empirical_cifar.masks import (
    artem_threshold_groups, geometric_groups, sensitivity_groups,
)
from wavelet_empirical_cifar.wiener import (
    covariance_eigensystem, denoiser_from_eigensystem, empirical_moments,
    restricted_wiener_clean, sensitivity_stability,
)
from wavelet_score.indexing import build_metadata
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import model_wavelet_state_and_velocity


def show_masks(layout, groups, output: Path):
    targets = [layout.subbands[index].flat_start + layout.subbands[index].size // 2
               for index in (0, 4, 7)]
    metadata = build_metadata(layout)
    labels = [f"cA{b.pywt_level}" if b.orientation is None
              else f"c{b.orientation}{b.pywt_level}" for b in layout.subbands]
    fig, axes = plt.subplots(len(targets), 2, figsize=(9, 8), sharex=True, sharey=True)
    for row, target in enumerate(targets):
        for col, name in enumerate(("geometric", "sensitivity")):
            chosen = {int(index) % layout.num_coefficients for index in groups[name][target].input_indices}
            for site in chosen:
                item = metadata[site]
                axes[row, col].scatter((item.k2+.5)/item.subband_shape[1],
                                       item.subband_index + .7*((item.k1+.5)/item.subband_shape[0]-.5),
                                       c="tab:red" if site == target else "tab:blue", s=24)
            axes[row, col].set_title(f"{name}, target {target}", fontsize=9)
            axes[row, col].set_ylim(len(layout.subbands)-.5, -.5)
            axes[row, col].grid(alpha=.15)
    axes[-1, 0].set_xlabel("Relative column within subband")
    axes[-1, 1].set_xlabel("Relative column within subband")
    for axis in axes[:, 0]:
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    argp = parser(__doc__, "wavelet_empirical_cifar/results/wiener_masks")
    argp.add_argument("--artem-threshold", type=float, default=None,
                      help="Add Artem's variable-size wavelet mask (CIFAR-10 tau=0.05)")
    argp.add_argument("--mask-size-only", action="store_true",
                      help="For a threshold run, inspect mask sizes without posterior evaluation")
    args = argp.parse_args()
    if args.mask_size_only and args.artem_threshold is None:
        argp.error("--mask-size-only requires --artem-threshold")
    if args.artem_threshold is not None and args.out_dir == Path("wavelet_empirical_cifar/results/wiener_masks"):
        label = f"{args.artem_threshold:g}".replace(".", "p")
        args.out_dir = Path(f"wavelet_empirical_cifar/results/wiener_masks_tau_{label}")
    device, transform, references, layout, clean, noise, generation_noise = prepare(args)
    mean, covariance = empirical_moments(references, device=device)
    values, vectors = covariance_eigensystem(covariance, device)
    if args.mask_size_only:
        mask_sizes = []
        for time in TIMES:
            denoiser = denoiser_from_eigensystem(values, vectors, time)
            groups = artem_threshold_groups(layout, denoiser, args.artem_threshold)
            counts = np.asarray([len(group.input_indices)//3 for group in groups])
            mask_sizes.append(dict(time=time, min_sites=int(counts.min()),
                median_sites=float(np.median(counts)), mean_sites=float(counts.mean()),
                p90_sites=float(np.percentile(counts, 90)), max_sites=int(counts.max())))
            print(mask_sizes[-1], flush=True)
        save_rows(args.out_dir / "artem_mask_sizes.csv", mask_sizes)
        return
    model = make_model(device, not args.no_unet)
    estimator = make_estimator(args, device, references, layout, geometric_groups(layout, 9))
    budgets = (9, 25)
    all_groups = {}
    rows = []
    mask_sizes = []
    for time in TIMES:
        print(f"Sensitivity and calibration t={time:.2f}", flush=True)
        denoiser = denoiser_from_eigensystem(values, vectors, time)
        if args.artem_threshold is not None:
            groups = artem_threshold_groups(layout, denoiser, args.artem_threshold)
            counts = np.asarray([len(group.input_indices)//3 for group in groups])
            mask_sizes.append(dict(time=time, min_sites=int(counts.min()),
                median_sites=float(np.median(counts)), mean_sites=float(counts.mean()),
                p90_sites=float(np.percentile(counts, 90)), max_sites=int(counts.max())))
            all_groups[(time, "artem_threshold")] = groups
            print(f"  Artem tau={args.artem_threshold}: median={np.median(counts):.0f}, "
                  f"p90={np.percentile(counts,90):.0f}, max={counts.max()} RGB sites", flush=True)
        for budget in budgets:
            for name, groups in (("geometric", geometric_groups(layout, budget)),
                                 ("sensitivity", sensitivity_groups(layout, denoiser, budget))):
                variant = f"{name}_{budget}"
                all_groups[(time, variant)] = groups
                result = evaluate_estimator(estimator, groups, clean["calibration"],
                    noise["calibration"], time, transform, model, device)
                rows.append(dict(split="calibration", time=time, variant=variant, **result))
                print(f"  {variant}: clean MSE={result['clean_mse']:.6f} ({result['seconds']:.1f}s)", flush=True)
        if args.artem_threshold is not None:
            result = evaluate_estimator(estimator, all_groups[(time, "artem_threshold")],
                clean["calibration"], noise["calibration"], time, transform, model, device)
            rows.append(dict(split="calibration", time=time, variant="artem_threshold", **result))
        del denoiser
    save_rows(args.out_dir / "calibration.csv", rows)
    if mask_sizes:
        save_rows(args.out_dir / "artem_mask_sizes.csv", mask_sizes)
    # Select one budget from calibration only, comparing like-sized contexts.
    budget_scores = {budget: float(np.mean([r["clean_mse"] for r in rows
                          if r["variant"] in (f"geometric_{budget}", f"sensitivity_{budget}")]))
                     for budget in budgets}
    selected_budget = min(budget_scores, key=budget_scores.get)
    variants = (f"geometric_{selected_budget}", f"sensitivity_{selected_budget}")
    if args.artem_threshold is not None:
        variants += ("artem_threshold",)
    print(f"Selected budget from calibration: {selected_budget} RGB sites", flush=True)

    evaluation = []
    for time in TIMES:
        state, target = state_and_targets(clean["evaluation"], noise["evaluation"], time, transform)
        model_clean = None
        if model is not None:
            _, velocity = model_wavelet_state_and_velocity(model, clean["evaluation"],
                noise["evaluation"], time, transform, device, 32)
            model_clean = state + (1-time)*velocity
        for variant in variants:
            groups = all_groups[(time, variant)]
            result = evaluate_estimator(estimator, groups, clean["evaluation"],
                noise["evaluation"], time, transform, model, device)
            evaluation.append(dict(split="evaluation", time=time, variant=variant, **result))
            started = perf_counter()
            restricted = restricted_wiener_clean(state, mean, covariance, time, groups, device=device)
            extra = ({"unet_clean_mse": mse(model_clean, target),
                      "agreement_mse": mse(restricted, model_clean)} if model_clean is not None else {})
            evaluation.append(dict(split="evaluation", time=time,
                variant=f"restricted_wiener_{variant}", clean_mse=mse(restricted, target),
                seconds=perf_counter()-started, **extra))
        print(f"Evaluation t={time:.2f} complete", flush=True)
    save_rows(args.out_dir / "evaluation.csv", evaluation)
    plot_mse(evaluation, args.out_dir / "mse.png")
    if args.artem_threshold is not None:
        empirical = [row for row in evaluation if not row["variant"].startswith("restricted_wiener_")]
        if model is not None:
            empirical.extend(dict(split="evaluation", time=time, variant="U-Net FM",
                                  clean_mse=next(row["unet_clean_mse"] for row in evaluation
                                                 if row["time"] == time)) for time in TIMES)
        plot_mse(empirical, args.out_dir / "mse_empirical_and_unet.png")
    show_masks(layout, {name: all_groups[(0.50, f"{name}_{selected_budget}")]
                        for name in ("geometric", "sensitivity")}, args.out_dir / "mask_examples.png")

    # Two disjoint halves of training data measure top-site stability at t=0.5.
    half = len(references) // 2
    if half >= 2:
        halves = []
        for subset in (references[:half], references[half:]):
            _, cov = empirical_moments(subset, device=device)
            eigen = covariance_eigensystem(cov, device)
            halves.append(denoiser_from_eigensystem(*eigen, 0.5))
        band_sizes = np.asarray([layout.subbands[item.subband_index].size
                                 for item in build_metadata(layout)])
        stability = sensitivity_stability(*halves, layout.num_coefficients,
                                          selected_budget, band_sizes=band_sizes)
        save_json(args.out_dir / "mask_stability.json", {"time": .5,
                  "budget_sites": selected_budget, "mean_overlap": stability})

    if not args.no_generation:
        generated = {}
        for variant in variants:
            generated[variant] = generate_estimator(estimator,
                lambda t, variant=variant: all_groups[(min(TIMES, key=lambda anchor: abs(anchor-t)), variant)],
                generation_noise, transform, layout, args.steps)
        save_generation_grid(generated, references, transform, layout, args.out_dir, device)
    save_json(args.out_dir / "selection.json", {"selected_budget_rgb_sites": selected_budget,
        "calibration_scores": budget_scores,
        "artem_threshold": args.artem_threshold,
        "artem_mask_rule": "scalar row/diagonal, global max, then RGB-site OR",
        "generation_mask_interpolation": "nearest of five measured times"})


if __name__ == "__main__":
    main()
