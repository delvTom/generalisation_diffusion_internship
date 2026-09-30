"""Study 3: add fixed cA3 context to fine same-band empirical masks."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from wavelet_empirical_cifar.common import (
    TIMES, generate_estimator, make_estimator, make_model,
    parser, plot_mse, prepare, save_generation_grid, save_json, save_rows,
    state_and_targets, mse,
)
from wavelet_fm_structure.run_cifar10_fm_calibrated_radius import model_wavelet_state_and_velocity
from wavelet_empirical_cifar.masks import add_coarse_context, family_groups
from wavelet_score.indexing import build_metadata


def main():
    argp = parser(__doc__, "wavelet_empirical_cifar/results/coarse_context")
    argp.add_argument("--base-family", choices=("same_band", "orientation_augmented",
        "scale_augmented", "full"), default="full",
        help="Fixed reference family; one family is screened per run")
    args = argp.parse_args()
    device, transform, references, layout, clean, noise, generation_noise = prepare(args)
    model = make_model(device, not args.no_unet)
    baseline = family_groups(layout, args.base_family, radius=1)
    variants = {amount: add_coarse_context(layout, baseline, amount)
                for amount in ("none", "center", "local", "all")}
    fine_sites = [item.flat_index for item in build_metadata(layout)
                  if item.coefficient_type == "detail" and item.pywt_level == 1]
    columns = np.asarray([channel*layout.num_coefficients + site
                          for channel in range(3) for site in fine_sites])
    estimator = make_estimator(args, device, references, layout, baseline)
    rows = []
    for split in ("calibration", "evaluation"):
        for time in TIMES:
            state, target = state_and_targets(clean[split], noise[split], time, transform)
            model_target = None
            if model is not None:
                _, velocity = model_wavelet_state_and_velocity(model, clean[split], noise[split],
                    time, transform, device, 32)
                model_target = state + (1-time)*velocity
            for name, groups in variants.items():
                estimator.set_groups(groups)
                started = perf_counter()
                predicted = estimator.predict_clean(state, time)
                result = dict(clean_mse=mse(predicted, target),
                              fine_clean_mse=mse(predicted[:, columns], target[:, columns]),
                              seconds=perf_counter()-started)
                if model_target is not None:
                    result.update(unet_clean_mse=mse(model_target, target),
                                  agreement_mse=mse(predicted, model_target))
                rows.append(dict(split=split, time=time, variant=name, **result))
                print(f"{split} t={time:.2f} {name}: fine MSE={result['fine_clean_mse']:.6f}", flush=True)
    save_rows(args.out_dir / "metrics.csv", rows)
    plot_mse(rows, args.out_dir / "fine_mse.png", metric="fine_clean_mse",
             ylabel="Fine-detail clean-coefficient MSE, test")
    plot_mse(rows, args.out_dir / "full_mse.png")
    save_json(args.out_dir / "contexts.json", {name: {
        "base_family": args.base_family,
        "fine_site_count": len(fine_sites),
        "example_input_rgb_coordinates": len(groups[fine_sites[0]].input_indices),
        "time_dependence": "none",
    } for name, groups in variants.items()})
    if not args.no_generation:
        generated = {name: generate_estimator(estimator, lambda t, groups=groups: groups,
                     generation_noise, transform, layout, args.steps)
                     for name, groups in variants.items()}
        save_generation_grid(generated, references, transform, layout, args.out_dir, device)


if __name__ == "__main__":
    main()
