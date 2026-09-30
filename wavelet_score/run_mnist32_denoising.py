"""Compare audited wavelet polynomial denoisers on MNIST-32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from wavelet_score.band_coupled import BandCoupledPolynomialRegressor
from wavelet_score.data import MNISTSplit, load_mnist_32
from wavelet_score.independent import IndependentPolynomialRegressor
from wavelet_score.local_coupled import LocalCoupledPolynomialRegressor
from wavelet_score.noise import VENoiseBatch, sample_ve_noise
from wavelet_score.transforms import WaveletTransform2D
from wavelet_score.torch_regressors import (
    TorchBandCoupledPolynomialRegressor,
    TorchIndependentPolynomialRegressor,
    TorchLocalCoupledPolynomialRegressor,
    torch_dtype,
)

MODEL_NAMES = ("independent", "band_coupled", "local_coupled")


def parse_number_list(text: str, cast: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    """Parse a non-empty comma-separated list of numeric values."""
    try:
        values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {text!r}.") from error
    if not values:
        raise argparse.ArgumentTypeError("The numeric list cannot be empty.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse all reproducible experiment settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--out-dir", type=Path, default=Path("wavelet_score/results/mnist32_denoising"))
    parser.add_argument("--sigmas", type=str, default="1,2,3,4")
    parser.add_argument("--degrees", type=str, default="1,2,3")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--backend", choices=("torch", "numpy"), default="torch")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--evaluation-split", choices=("train", "test"), default="train")
    parser.add_argument("--num-plot-images", type=int, default=6)
    parser.add_argument(
        "--subband-degree",
        type=int,
        default=None,
        help="Degree used for independent/local subband diagnostics; defaults to max(degrees).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.sigmas = parse_number_list(args.sigmas, float)
    args.degrees = parse_number_list(args.degrees, int)
    if any(sigma <= 0.0 for sigma in args.sigmas):
        parser.error("Every sigma must be positive.")
    if any(degree < 1 for degree in args.degrees):
        parser.error("Every polynomial degree must be at least one.")
    if args.ridge < 0.0 or args.local_radius < 1 or args.num_plot_images < 1:
        parser.error("ridge, local-radius, or num-plot-images is invalid.")
    if args.subband_degree is None:
        args.subband_degree = max(args.degrees)
    if args.subband_degree not in args.degrees:
        parser.error("subband-degree must be included in --degrees.")
    return args


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto/cpu/cuda and fail early when CUDA was requested but unavailable."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def make_model(
    name: str,
    degree: int,
    ridge: float,
    local_radius: int,
    backend: str,
    device: torch.device,
    compute_dtype: torch.dtype,
):
    """Construct one fresh polynomial regressor for a model family."""
    if backend == "torch":
        if name == "independent":
            return TorchIndependentPolynomialRegressor(
                degree, ridge, device, compute_dtype
            )
        if name == "band_coupled":
            return TorchBandCoupledPolynomialRegressor(
                degree, ridge, device, compute_dtype
            )
        if name == "local_coupled":
            return TorchLocalCoupledPolynomialRegressor(
                degree, local_radius, ridge, device, compute_dtype
            )
        raise ValueError(f"Unknown model family: {name}.")
    if backend != "numpy":
        raise ValueError(f"Unknown backend: {backend}.")
    if name == "independent":
        return IndependentPolynomialRegressor(degree=degree, ridge=ridge)
    if name == "band_coupled":
        return BandCoupledPolynomialRegressor(degree=degree, ridge=ridge)
    if name == "local_coupled":
        return LocalCoupledPolynomialRegressor(
            degree=degree,
            radius=local_radius,
            ridge=ridge,
        )
    raise ValueError(f"Unknown model family: {name}.")


def fit_and_predict(
    name: str,
    degree: int,
    ridge: float,
    local_radius: int,
    train_noise: VENoiseBatch,
    eval_noise: VENoiseBatch,
    transform: WaveletTransform2D,
    batch_size: int,
    backend: str,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[np.ndarray, float]:
    """Fit one family and return clipped pixel predictions plus elapsed seconds."""
    model = make_model(
        name,
        degree,
        ridge,
        local_radius,
        backend,
        device,
        compute_dtype,
    )
    start = perf_counter()
    if name == "independent":
        model.fit(train_noise.Y_t, train_noise.Y_0, batch_size=batch_size)
    else:
        model.fit(
            train_noise.Y_t,
            train_noise.Y_0,
            train_noise.layout,
            batch_size=batch_size,
        )
    predicted_Y_0 = model.predict(eval_noise.Y_t, batch_size=batch_size)
    predicted_images = transform.synthesize(predicted_Y_0, eval_noise.layout)
    elapsed_seconds = perf_counter() - start
    return np.clip(predicted_images, 0.0, 1.0), elapsed_seconds


def mean_squared_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the mean squared pixel error over samples and spatial coordinates."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes.")
    return float(np.mean((prediction.astype(np.float64) - target) ** 2))


def first_noise_samples(noise: VENoiseBatch, count: int) -> VENoiseBatch:
    """Take the first samples while preserving one paired noise realization."""
    if count < 1 or count > noise.images_0.shape[0]:
        raise ValueError("count must select a non-empty prefix of the noise batch.")
    return VENoiseBatch(
        images_0=noise.images_0[:count],
        epsilon=noise.epsilon[:count],
        images_t=noise.images_t[:count],
        Y_0=noise.Y_0[:count],
        Eta=noise.Eta[:count],
        Y_t=noise.Y_t[:count],
        sigma=noise.sigma,
        layout=noise.layout,
    )


def plot_mse_by_degree(metrics: list[dict[str, object]], out_path: Path) -> None:
    """Plot grouped MSE bars, with one panel per degree and one bar per family."""
    degrees = sorted({int(row["degree"]) for row in metrics})
    figure, axes = plt.subplots(1, len(degrees), figsize=(5.0 * len(degrees), 4.2), squeeze=False)
    for axis, degree in zip(axes[0], degrees, strict=True):
        rows = [row for row in metrics if int(row["degree"]) == degree]
        sigmas = sorted({float(row["sigma"]) for row in rows})
        positions = np.arange(len(sigmas))
        width = 0.8 / len(MODEL_NAMES)
        for model_index, name in enumerate(MODEL_NAMES):
            selected = sorted(
                (row for row in rows if row["model"] == name),
                key=lambda row: float(row["sigma"]),
            )
            axis.bar(
                positions + (model_index - (len(MODEL_NAMES) - 1) / 2) * width,
                [float(row["mse"]) for row in selected],
                width=width,
                label=name.replace("_", " "),
            )
        axis.set_title(f"Polynomial degree D={degree}")
        axis.set_xlabel("Noise standard deviation sigma")
        axis.set_ylabel("Pixel MSE after clipping to [0,1]")
        axis.set_xticks(positions, [f"{sigma:g}" for sigma in sigmas])
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("MNIST-32 wavelet polynomial denoising", fontsize=14)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mse_by_family(metrics: list[dict[str, object]], out_path: Path) -> None:
    """Reproduce Figure 5 with grouped degree bars for each feature family."""
    degrees = sorted({int(row["degree"]) for row in metrics})
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5.0 * len(MODEL_NAMES), 4.2), squeeze=False)
    for axis, name in zip(axes[0], MODEL_NAMES, strict=True):
        rows = [row for row in metrics if row["model"] == name]
        sigmas = sorted({float(row["sigma"]) for row in rows})
        positions = np.arange(len(sigmas))
        width = 0.8 / len(degrees)
        for degree_index, degree in enumerate(degrees):
            selected = sorted(
                (row for row in rows if int(row["degree"]) == degree),
                key=lambda row: float(row["sigma"]),
            )
            axis.bar(
                positions + (degree_index - (len(degrees) - 1) / 2) * width,
                [float(row["mse"]) for row in selected],
                width=width,
                label=f"D={degree}",
            )
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel("Noise standard deviation sigma")
        axis.set_ylabel("Pixel MSE after clipping to [0,1]")
        axis.set_xticks(positions, [f"{sigma:g}" for sigma in sigmas])
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, title="Polynomial degree")
    figure.suptitle("MNIST-32 denoising by wavelet feature family", fontsize=14)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def sigma_filename(sigma: float) -> str:
    """Return a stable filename fragment for a noise standard deviation."""
    return f"{sigma:g}".replace("-", "minus_").replace(".", "p")


def subband_label(descriptor) -> str:
    """Return the conventional cA_L/cH_L/cV_L/cD_L label of a subband."""
    if descriptor.coefficient_type == "scaling":
        return f"cA{descriptor.pywt_level}"
    return f"c{descriptor.orientation}{descriptor.pywt_level}"


def compute_subband_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    transform: WaveletTransform2D,
    model_name: str,
    sigma: float,
    degree: int,
) -> list[dict[str, object]]:
    """Decompose clipped pixel MSE into orthogonal wavelet subbands."""
    predicted_Y_0, layout = transform.analyze(prediction)
    target_Y_0, target_layout = transform.analyze(target)
    if target_layout != layout:
        raise RuntimeError("Prediction and target produced different wavelet layouts.")

    dimension = layout.num_coefficients
    rows: list[dict[str, object]] = []
    for descriptor in layout.subbands:
        error = (
            predicted_Y_0[:, descriptor.flat_slice].astype(np.float64)
            - target_Y_0[:, descriptor.flat_slice]
        )
        squared_error_sum = float(np.sum(error**2))
        mean_per_coefficient = squared_error_sum / error.size
        contribution_to_pixel_mse = squared_error_sum / (error.shape[0] * dimension)
        rows.append(
            {
                "sigma": sigma,
                "degree": degree,
                "model": model_name,
                "subband": subband_label(descriptor),
                "coefficient_type": descriptor.coefficient_type,
                "orientation": descriptor.orientation,
                "pywt_level": descriptor.pywt_level,
                "math_scale_j": descriptor.math_scale_j,
                "num_coefficients": descriptor.size,
                "mse_per_coefficient": mean_per_coefficient,
                "contribution_to_pixel_mse": contribution_to_pixel_mse,
            }
        )
    return rows


def plot_subband_comparison(
    rows: list[dict[str, object]],
    sigma: float,
    degree: int,
    out_path: Path,
) -> None:
    """Compare independent/local errors and local relative gain by subband."""
    sigma_rows = [
        row
        for row in rows
        if float(row["sigma"]) == sigma and int(row["degree"]) == degree
    ]
    labels = [
        str(row["subband"])
        for row in sigma_rows
        if row["model"] == "independent"
    ]
    if not labels:
        raise ValueError("No independent subband rows were provided.")
    values = {
        name: {
            str(row["subband"]): row
            for row in sigma_rows
            if row["model"] == name
        }
        for name in ("independent", "local_coupled")
    }
    if any(set(model_rows) != set(labels) for model_rows in values.values()):
        raise ValueError("Subband labels differ between model families.")

    positions = np.arange(len(labels))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    for offset, name in zip((-width / 2, width / 2), values, strict=True):
        axes[0].bar(
            positions + offset,
            [float(values[name][label]["mse_per_coefficient"]) for label in labels],
            width=width,
            label=name.replace("_", " "),
        )
    independent = np.array(
        [float(values["independent"][label]["contribution_to_pixel_mse"]) for label in labels]
    )
    local = np.array(
        [float(values["local_coupled"][label]["contribution_to_pixel_mse"]) for label in labels]
    )
    relative_gain = 100.0 * (independent - local) / np.maximum(independent, 1e-15)
    axes[1].bar(positions, relative_gain, color=np.where(relative_gain >= 0.0, "tab:green", "tab:red"))
    axes[1].axhline(0.0, color="black", linewidth=0.8)

    axes[0].set_ylabel("MSE per wavelet coefficient")
    axes[0].set_title("Absolute reconstruction error in each subband")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("Local improvement over independent (%)")
    axes[1].set_xlabel("Wavelet subband")
    axes[1].set_title("Positive values mean that local coupling reduces the subband contribution")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(positions, labels, rotation=45, ha="right")
    figure.suptitle(f"MNIST-32 subband error decomposition: sigma={sigma:g}, D={degree}", fontsize=14)
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_family_reconstructions(
    clean: np.ndarray,
    noisy: np.ndarray,
    predictions_by_degree: dict[int, np.ndarray],
    family_name: str,
    sigma: float,
    out_path: Path,
) -> None:
    """Plot clean/noisy inputs and all polynomial-degree reconstructions."""
    degrees = sorted(predictions_by_degree)
    column_names = ("clean", "noisy", *(f"estimate D={degree}" for degree in degrees))
    arrays = (clean, noisy, *(predictions_by_degree[degree] for degree in degrees))
    num_images = clean.shape[0]
    figure, axes = plt.subplots(
        num_images,
        len(column_names),
        figsize=(2.1 * len(column_names), 2.0 * num_images),
        squeeze=False,
    )
    for column, (title, images) in enumerate(zip(column_names, arrays, strict=True)):
        axes[0, column].set_title(title, fontsize=11)
        for row in range(num_images):
            axes[row, column].imshow(images[row], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row in range(num_images):
        axes[row, 0].set_ylabel(f"image {row}", fontsize=10)
    figure.suptitle(
        f"{family_name.replace('_', ' ').title()} - MNIST-32, sigma={sigma:g}",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_noise_regimes(
    clean: np.ndarray,
    sigmas: tuple[float, ...],
    seed: int,
    out_path: Path,
) -> None:
    """Show clean images and sigma-scaled versions of one shared Gaussian noise draw."""
    if clean.ndim != 3 or clean.shape[0] == 0:
        raise ValueError("clean must have shape [N,H,W] with at least one image.")
    if not sigmas or any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("sigmas must contain positive values.")
    epsilon = np.random.default_rng(seed).standard_normal(clean.shape, dtype=clean.dtype)
    noisy_batches = tuple(np.clip(clean + sigma * epsilon, 0.0, 1.0) for sigma in sigmas)
    column_names = ("clean", *(f"sigma={sigma:g}" for sigma in sigmas))
    arrays = (clean, *noisy_batches)
    figure, axes = plt.subplots(
        clean.shape[0],
        len(arrays),
        figsize=(2.0 * len(arrays), 2.0 * clean.shape[0]),
        squeeze=False,
    )
    for column, (title, images) in enumerate(zip(column_names, arrays, strict=True)):
        axes[0, column].set_title(title, fontsize=11)
        for row in range(clean.shape[0]):
            axes[row, column].imshow(images[row], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row in range(clean.shape[0]):
        axes[row, 0].set_ylabel(f"image {row}", fontsize=10)
    figure.suptitle(
        "MNIST-32 VE corruption: X_sigma = X_0 + sigma Z (shared Z by row)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_splits(args: argparse.Namespace) -> tuple[MNISTSplit, MNISTSplit]:
    """Load the fit split and the explicitly selected evaluation split."""
    train = load_mnist_32(
        root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        download=args.download,
    )
    if args.evaluation_split == "train":
        eval_images = train.images
        if args.max_eval_samples is not None:
            eval_images = eval_images[: args.max_eval_samples]
        evaluation = MNISTSplit(
            images=np.ascontiguousarray(eval_images),
            labels=np.ascontiguousarray(train.labels[: len(eval_images)]),
            split="train",
        )
    else:
        evaluation = load_mnist_32(
            root=args.data_root,
            split="test",
            batch_size=args.batch_size,
            max_samples=args.max_eval_samples,
            download=args.download,
        )
    return train, evaluation


def main() -> None:
    """Run all sigma/degree/model combinations and save reproducible outputs."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    compute_dtype = torch_dtype(args.compute_dtype)
    if args.backend == "numpy" and args.device not in ("auto", "cpu"):
        raise ValueError("The NumPy backend is CPU-only; use --device cpu or auto.")
    train, evaluation = load_splits(args)
    transform = WaveletTransform2D(args.wavelet, args.level, mode="periodization")
    config = {
        **{key: value for key, value in vars(args).items() if key != "out_dir"},
        "data_root": str(args.data_root),
        "out_dir": str(args.out_dir),
        "sigmas": list(args.sigmas),
        "degrees": list(args.degrees),
        "num_train_samples": len(train.images),
        "num_eval_samples": len(evaluation.images),
        "resolved_device": str(device if args.backend == "torch" else "cpu"),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if args.backend == "torch" and device.type == "cuda"
            else None
        ),
        "note": "The paper states a small ridge penalty but does not report its numeric value.",
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    noise_plot_count = min(args.num_plot_images, len(evaluation.images))
    plot_noise_regimes(
        clean=evaluation.images[:noise_plot_count],
        sigmas=args.sigmas,
        seed=args.seed + 100_000,
        out_path=args.out_dir / "noise_regimes.png",
    )

    metrics: list[dict[str, object]] = []
    subband_metrics: list[dict[str, object]] = []
    print(
        f"backend={args.backend} device={device if args.backend == 'torch' else 'cpu'} "
        f"dtype={args.compute_dtype}",
        flush=True,
    )
    for sigma_index, sigma in enumerate(args.sigmas):
        train_noise = sample_ve_noise(
            train.images,
            sigma,
            transform,
            np.random.default_rng(args.seed + 2 * sigma_index),
        )
        if args.evaluation_split == "train":
            # This is the empirical in-sample protocol used for the paper's MNIST curves.
            eval_noise = first_noise_samples(train_noise, len(evaluation.images))
        else:
            eval_noise = sample_ve_noise(
                evaluation.images,
                sigma,
                transform,
                np.random.default_rng(args.seed + 2 * sigma_index + 1),
            )
        predictions_for_sigma: dict[str, dict[int, np.ndarray]] = {
            name: {} for name in MODEL_NAMES
        }
        for degree in args.degrees:
            for name in MODEL_NAMES:
                print(f"sigma={sigma:g} D={degree} model={name}", flush=True)
                prediction, elapsed_seconds = fit_and_predict(
                    name=name,
                    degree=degree,
                    ridge=args.ridge,
                    local_radius=args.local_radius,
                    train_noise=train_noise,
                    eval_noise=eval_noise,
                    transform=transform,
                    batch_size=args.batch_size,
                    backend=args.backend,
                    device=device,
                    compute_dtype=compute_dtype,
                )
                mse = mean_squared_error(prediction, eval_noise.images_0)
                metrics.append(
                    {
                        "sigma": sigma,
                        "degree": degree,
                        "model": name,
                        "mse": mse,
                        "fit_and_predict_seconds": elapsed_seconds,
                    }
                )
                predictions_for_sigma[name][degree] = prediction
                if degree == args.subband_degree and name in (
                    "independent",
                    "local_coupled",
                ):
                    subband_metrics.extend(
                        compute_subband_metrics(
                            prediction=prediction,
                            target=eval_noise.images_0,
                            transform=transform,
                            model_name=name,
                            sigma=sigma,
                            degree=degree,
                        )
                    )
                print(f"  mse={mse:.8f} elapsed={elapsed_seconds:.1f}s", flush=True)
        (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        reconstruction_dir = args.out_dir / "reconstructions"
        reconstruction_dir.mkdir(exist_ok=True)
        count = min(args.num_plot_images, len(eval_noise.images_0))
        for name in MODEL_NAMES:
            plot_family_reconstructions(
                clean=eval_noise.images_0[:count],
                noisy=np.clip(eval_noise.images_t[:count], 0.0, 1.0),
                predictions_by_degree={
                    degree: prediction[:count]
                    for degree, prediction in predictions_for_sigma[name].items()
                },
                family_name=name,
                sigma=sigma,
                out_path=reconstruction_dir
                / f"{name}_sigma_{sigma_filename(sigma)}.png",
            )

        subband_dir = args.out_dir / "subband_analysis"
        subband_dir.mkdir(exist_ok=True)
        plot_subband_comparison(
            rows=subband_metrics,
            sigma=sigma,
            degree=args.subband_degree,
            out_path=subband_dir / f"subband_mse_sigma_{sigma_filename(sigma)}.png",
        )

    plot_mse_by_family(metrics, args.out_dir / "mse_vs_sigma_by_family.png")
    plot_mse_by_degree(metrics, args.out_dir / "mse_vs_sigma_by_degree.png")
    (args.out_dir / "subband_metrics.json").write_text(
        json.dumps(subband_metrics, indent=2) + "\n"
    )
    print(f"saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
