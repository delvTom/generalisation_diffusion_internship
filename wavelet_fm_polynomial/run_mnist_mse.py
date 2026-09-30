"""Evaluate dense polynomial wavelet velocity regressors on MNIST-32.

Inputs: official MNIST train/test data and degree/ridge settings. Outputs:
fit and held-out velocity-MSE tables/plots in ``--out-dir``. Example::

    python -m wavelet_fm_polynomial.run_mnist_mse --help
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wavelet-fm-polynomial")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wavelet_fm_polynomial.features import (  # noqa: E402
    FeatureFamily,
    build_feature_structure,
    evaluate_features,
    feature_count,
)
from wavelet_fm_polynomial.regression import fit_nested_degrees  # noqa: E402
from wavelet_fm_polynomial.wiener import AffineFMWiener  # noqa: E402
from wavelet_score.data import load_mnist_32  # noqa: E402
from wavelet_score.indexing import WaveletLayout  # noqa: E402
from wavelet_score.transforms import WaveletTransform2D  # noqa: E402

FAMILIES: tuple[FeatureFamily, ...] = ("independent", "band", "local")
DEFAULT_TIMES = (0.05, 0.12, 0.27, 0.50, 0.73, 0.88, 0.95)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from error
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("All times must lie strictly between zero and one.")
    return values


def parse_int_tuple(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
    if not values or any(value not in (1, 2, 3) for value in values):
        raise argparse.ArgumentTypeError("Degrees must be selected from 1, 2, and 3.")
    return tuple(sorted(set(values)))


def parse_family_tuple(text: str) -> tuple[FeatureFamily, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values or any(value not in FAMILIES for value in values):
        raise argparse.ArgumentTypeError(
            "Families must be selected from independent, band, and local."
        )
    return tuple(dict.fromkeys(values))  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wavelet_fm_polynomial/results/mnist32_mse"),
    )
    parser.add_argument("--num-fit", type=int, default=60_000)
    parser.add_argument("--num-validation", type=int, default=10_000)
    parser.add_argument("--times", type=parse_float_tuple, default=DEFAULT_TIMES)
    parser.add_argument("--degrees", type=parse_int_tuple, default=(1, 2, 3))
    parser.add_argument("--families", type=parse_family_tuple, default=FAMILIES)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--mode", default="periodization")
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--moment-dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Precision used to form feature products; float64 is the audited default.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Disabled by default; a nonzero value changes the primary objective.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if min(args.num_fit, args.num_validation, args.batch_size, args.level) < 1:
        parser.error("Sample counts, batch size, and wavelet level must be positive.")
    if args.local_radius < 1 or args.ridge < 0.0:
        parser.error("local-radius must be positive and ridge must be non-negative.")
    if 1 not in args.degrees:
        parser.error("Degree 1 is mandatory because it is the affine reference case.")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def prepare_wavelet_pairs(
    images: np.ndarray,
    *,
    transform: WaveletTransform2D,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, WaveletLayout]:
    """Return clean coefficients and one fixed independent noise per image."""
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(images.shape, dtype=np.float32)
    clean_coefficients, layout = transform.analyze(images)
    noise_coefficients, noise_layout = transform.analyze(noise)
    if noise_layout != layout:
        raise RuntimeError("Clean images and Gaussian noise produced different layouts.")
    return clean_coefficients, noise_coefficients, layout


def write_metrics(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ("split", "model", "family", "degree", "t", "mse")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_mse(
    rows: list[dict[str, object]],
    *,
    split: str,
    families: tuple[FeatureFamily, ...],
    degrees: tuple[int, ...],
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(families),
        figsize=(6.0 * len(families), 5.2),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {2: "#2878B5", 3: "#D95319"}
    baseline = sorted(
        (row for row in rows if row["split"] == split and row["model"] == "wiener_affine"),
        key=lambda row: float(row["t"]),
    )
    for column, family in enumerate(families):
        axis = axes[0, column]
        axis.plot(
            [row["t"] for row in baseline],
            [row["mse"] for row in baseline],
            color="black",
            linewidth=2.4,
            marker="o",
            label="D=1 = affine Wiener",
        )
        for degree in degrees:
            if degree == 1:
                continue
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["split"] == split
                    and row["family"] == family
                    and row["degree"] == degree
                ),
                key=lambda row: float(row["t"]),
            )
            if selected:
                axis.plot(
                    [row["t"] for row in selected],
                    [row["mse"] for row in selected],
                    linewidth=2.1,
                    marker="o",
                    color=colors[degree],
                    label=f"{family}, D={degree}",
                )
        axis.set_title(f"{family.capitalize()} dictionary")
        axis.set_xlabel("Flow time t")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel(r"MSE $=\mathbb{E}\|\hat V_t-(X-\varepsilon)\|_2^2/d$")
    title_split = "fit pairs" if split == "fit" else "held-out MNIST images and noises"
    figure.suptitle(f"Standard FM velocity prediction on {title_split}")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_state(
    out_dir: Path,
    rows: list[dict[str, object]],
    diagnostics: dict[str, object],
    families: tuple[FeatureFamily, ...],
    degrees: tuple[int, ...],
) -> None:
    write_metrics(rows, out_dir / "metrics.csv")
    with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    if rows:
        plot_mse(
            rows,
            split="fit",
            families=families,
            degrees=degrees,
            path=out_dir / "mse_fit.png",
        )
        plot_mse(
            rows,
            split="validation",
            families=families,
            degrees=degrees,
            path=out_dir / "mse_validation.png",
        )
        for degree in degrees:
            plot_degree_comparison(rows, families, degree, out_dir / f"mse_D{degree}.png")


def plot_degree_comparison(rows, families, degree, path) -> None:
    """Compare families at fixed degree on fit and held-out pairs."""
    colors = {"independent": "#0072B2", "band": "#D55E00", "local": "#009E73"}
    markers = {"independent": "o", "band": "s", "local": "^"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, split, title in zip(
        axes, ("fit", "validation"),
        ("Fit / train (adjustment pairs)", "Validation (held-out pairs)"), strict=True,
    ):
        baseline = sorted(
            (r for r in rows if r["split"] == split and r["model"] == "wiener_affine"),
            key=lambda r: float(r["t"]),
        )
        axis.plot(
            [float(r["t"]) for r in baseline], [float(r["mse"]) for r in baseline],
            color="black", linestyle="--", marker="x", linewidth=2,
            label="Wiener affine" if degree != 1 else "Wiener = independent = band = local",
        )
        if degree != 1:
            for family in families:
                selected = sorted(
                    (r for r in rows if r["split"] == split and r["family"] == family
                     and int(r["degree"]) == degree), key=lambda r: float(r["t"]),
                )
                if selected:
                    axis.plot(
                        [float(r["t"]) for r in selected],
                        [float(r["mse"]) for r in selected],
                        color=colors[family], marker=markers[family], linewidth=2,
                        label=family.capitalize(),
                    )
        axis.set(title=title, xlabel="Flow time t", ylabel="Velocity MSE (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(
        rf"Degree D={degree}: family comparison"
        "\n" + r"$\mathrm{MSE}=\frac{1}{Nd}\sum_n\|\hat V(X_t^{(n)},t)-(X^{(n)}-\varepsilon^{(n)})\|_2^2$"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def add_metric(
    rows: list[dict[str, object]],
    *,
    split: str,
    model: str,
    family: str,
    degree: int,
    t: float,
    mse: float,
) -> None:
    rows.append(
        {
            "split": split,
            "model": model,
            "family": family,
            "degree": degree,
            "t": t,
            "mse": mse,
        }
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    moment_dtype = torch.float32 if args.moment_dtype == "float32" else torch.float64
    solve_dtype = torch.float64
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "times": list(args.times),
        "degrees": list(args.degrees),
        "families": list(args.families),
        "device_resolved": str(device),
        "solve_dtype": "float64",
        "path": "X_t = t X + (1-t) epsilon",
        "target": "V = X - epsilon",
        "normalization": "MNIST pixels in [0,1]",
    }
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(f"device: {device}")
    print(f"moment_dtype: {args.moment_dtype}")
    print(f"ridge: {args.ridge:g}")
    print("loading MNIST-32...")
    fit_split = load_mnist_32(
        args.data_root,
        "train",
        batch_size=args.batch_size,
        max_samples=args.num_fit,
        download=args.download,
    )
    validation_split = load_mnist_32(
        args.data_root,
        "test",
        batch_size=args.batch_size,
        max_samples=args.num_validation,
        download=args.download,
    )
    transform = WaveletTransform2D(args.wavelet, args.level, args.mode)
    import pywt

    if not pywt.Wavelet(args.wavelet).orthogonal:
        raise ValueError("This experiment requires an orthogonal wavelet basis.")
    fit_clean, fit_noise, layout = prepare_wavelet_pairs(
        fit_split.images, transform=transform, seed=args.seed
    )
    validation_clean, validation_noise, validation_layout = prepare_wavelet_pairs(
        validation_split.images, transform=transform, seed=args.seed + 1
    )
    if validation_layout != layout:
        raise RuntimeError("Fit and validation wavelet layouts differ.")
    del fit_split, validation_split
    gc.collect()

    structure = build_feature_structure(layout, local_radius=args.local_radius)
    max_degree = max(args.degrees)
    counts = {
        family: {
            degree: feature_count(structure, family, degree)
            for degree in args.degrees
        }
        for family in args.families
    }
    for family in args.families:
        details = ", ".join(f"D={degree}: {count}" for degree, count in counts[family].items())
        print(f"features {family}: {details} (intercept included)")
        max_count = counts[family][max_degree] - 1
        gram_gib = max_count**2 * torch.tensor([], dtype=solve_dtype).element_size() / 2**30
        print(f"  maximum float64 Gram matrix: {gram_gib:.2f} GiB")
        if max_degree >= 2 and len(fit_clean) <= max_count:
            raise ValueError(
                f"{family} D={max_degree} needs more than {max_count} fit samples; "
                f"received {len(fit_clean)}."
            )

    # A direct runtime assertion that all degree-one dictionaries are identical.
    audit_input = torch.as_tensor(fit_clean[: min(8, len(fit_clean))], device=device)
    reference_features = evaluate_features(audit_input, structure, "independent", 1)
    for family in args.families:
        candidate = evaluate_features(audit_input, structure, family, 1)
        if not torch.equal(reference_features, candidate):
            raise RuntimeError(f"The {family} degree-one dictionary is not exactly Y_t.")
    del audit_input, reference_features

    print("computing analytic affine FM Wiener moments...")
    wiener = AffineFMWiener.fit(
        fit_clean,
        device=device,
        batch_size=args.batch_size,
        moment_dtype=moment_dtype,
        solve_dtype=solve_dtype,
    )
    fit_targets = np.ascontiguousarray(fit_clean - fit_noise)
    validation_targets = np.ascontiguousarray(validation_clean - validation_noise)
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "feature_counts_including_intercept": counts,
        "degree_one_dictionary": "exactly [1, Y_t] for every family",
        "fits": {},
        "nesting_checks": [],
    }

    experiment_start = perf_counter()
    for time_index, t in enumerate(args.times):
        print(f"\n=== t={t:.4f} ({time_index + 1}/{len(args.times)}) ===")
        fit_Y_t = np.ascontiguousarray(t * fit_clean + (1.0 - t) * fit_noise)
        validation_Y_t = np.ascontiguousarray(
            t * validation_clean + (1.0 - t) * validation_noise
        )
        fit_wiener_mse = wiener.mse(
            fit_Y_t, fit_targets, t, batch_size=args.batch_size
        )
        validation_wiener_mse = wiener.mse(
            validation_Y_t, validation_targets, t, batch_size=args.batch_size
        )
        print(
            f"Wiener affine: fit={fit_wiener_mse:.8f}, "
            f"validation={validation_wiener_mse:.8f}"
        )
        add_metric(
            rows,
            split="fit",
            model="wiener_affine",
            family="wiener",
            degree=1,
            t=t,
            mse=fit_wiener_mse,
        )
        add_metric(
            rows,
            split="validation",
            model="wiener_affine",
            family="wiener",
            degree=1,
            t=t,
            mse=validation_wiener_mse,
        )
        for family in args.families:
            # D=1 is deliberately the same analytic affine function, not a
            # separate finite-noise fit.
            add_metric(
                rows,
                split="fit",
                model=f"{family}_D1",
                family=family,
                degree=1,
                t=t,
                mse=fit_wiener_mse,
            )
            add_metric(
                rows,
                split="validation",
                model=f"{family}_D1",
                family=family,
                degree=1,
                t=t,
                mse=validation_wiener_mse,
            )

        fitted_degrees = tuple(degree for degree in args.degrees if degree >= 2)
        save_state(args.out_dir, rows, diagnostics, args.families, args.degrees)
        for family in args.families:
            if not fitted_degrees:
                continue
            print(f"fitting {family}, degrees={fitted_degrees}...")
            models = fit_nested_degrees(
                fit_Y_t,
                fit_targets,
                structure=structure,
                family=family,
                degrees=fitted_degrees,
                device=device,
                batch_size=args.batch_size,
                moment_dtype=moment_dtype,
                solve_dtype=solve_dtype,
                ridge=args.ridge,
            )
            previous_fit_mse = fit_wiener_mse
            previous_degree = 1
            for degree, model in models.items():
                fit_mse = model.mse(fit_Y_t, fit_targets, batch_size=args.batch_size)
                validation_mse = model.mse(
                    validation_Y_t, validation_targets, batch_size=args.batch_size
                )
                print(
                    f"  D={degree}: fit={fit_mse:.8f}, "
                    f"validation={validation_mse:.8f}, "
                    f"residual={model.diagnostics['relative_normal_equation_residual']:.3e}"
                )
                add_metric(
                    rows,
                    split="fit",
                    model=f"{family}_D{degree}",
                    family=family,
                    degree=degree,
                    t=t,
                    mse=fit_mse,
                )
                add_metric(
                    rows,
                    split="validation",
                    model=f"{family}_D{degree}",
                    family=family,
                    degree=degree,
                    t=t,
                    mse=validation_mse,
                )
                tolerance = 1e-7 + 1e-5 * max(previous_fit_mse, 1.0)
                passed = args.ridge > 0.0 or fit_mse <= previous_fit_mse + tolerance
                check = {
                    "t": t,
                    "family": family,
                    "previous_degree": previous_degree,
                    "degree": degree,
                    "previous_fit_mse": previous_fit_mse,
                    "fit_mse": fit_mse,
                    "tolerance": tolerance,
                    "passed": passed,
                }
                diagnostics["nesting_checks"].append(check)  # type: ignore[union-attr]
                if not passed:
                    raise RuntimeError(
                        f"Unregularized fit MSE increased for {family} at t={t}: "
                        f"{previous_fit_mse:.8f} -> {fit_mse:.8f}."
                    )
                previous_fit_mse = fit_mse
                previous_degree = degree
                fit_diagnostics = diagnostics["fits"]
                if not isinstance(fit_diagnostics, dict):
                    raise RuntimeError("Malformed fit diagnostics container.")
                fit_diagnostics[f"t={t:g}/{family}/D={degree}"] = model.diagnostics
                save_state(args.out_dir, rows, diagnostics, args.families, args.degrees)
            del models
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        diagnostics["elapsed_seconds"] = perf_counter() - experiment_start
        save_state(args.out_dir, rows, diagnostics, args.families, args.degrees)

    print(f"\nresults: {args.out_dir}")
    print(f"elapsed_seconds: {perf_counter() - experiment_start:.1f}")


if __name__ == "__main__":
    main()
