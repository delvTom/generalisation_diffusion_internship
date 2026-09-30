"""Plot the empirical effective SNR of every MNIST wavelet subband."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavelet_fm_structure.protocol import DEFAULT_PROTOCOL
from wavelet_score.data import load_mnist_32
from wavelet_score.indexing import SubbandDescriptor, WaveletLayout
from wavelet_score.transforms import WaveletTransform2D


def subband_name(descriptor: SubbandDescriptor) -> str:
    if descriptor.coefficient_type == "scaling":
        return f"cA{descriptor.pywt_level}"
    return f"c{descriptor.orientation}{descriptor.pywt_level}"


def compute_subband_signal_variances(
    Y_0: np.ndarray,
    layout: WaveletLayout,
) -> dict[str, float]:
    """Return the mean coordinate variance in each wavelet subband."""
    if Y_0.ndim != 2 or Y_0.shape[1] != layout.num_coefficients:
        raise ValueError(
            f"Y_0 must have shape [N,{layout.num_coefficients}], got {Y_0.shape}."
        )
    if len(Y_0) < 2 or not np.isfinite(Y_0).all():
        raise ValueError("Y_0 must contain at least two finite observations.")

    variances: dict[str, float] = {}
    for descriptor in layout.subbands:
        coordinate_variances = np.var(
            Y_0[:, descriptor.flat_slice], axis=0, dtype=np.float64
        )
        variances[subband_name(descriptor)] = float(np.mean(coordinate_variances))
    return variances


def effective_snr(signal_variance: float, times: np.ndarray) -> np.ndarray:
    """Compute t^2 Var(Y_0) / (1-t)^2 for standard Flow Matching."""
    times = np.asarray(times, dtype=np.float64)
    if signal_variance < 0.0 or not np.isfinite(signal_variance):
        raise ValueError("signal_variance must be finite and non-negative.")
    if times.ndim != 1 or np.any((times <= 0.0) | (times >= 1.0)):
        raise ValueError("times must be a vector strictly inside (0,1).")
    return signal_variance * np.square(times / (1.0 - times))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path(
            "wavelet_fm_structure/results/experiment_a_mnist32/"
            "effective_snr_by_subband.png"
        ),
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 2:
        parser.error("max-images must be at least two")
    return args


def plot_subband_snr(
    signal_variances: dict[str, float],
    out_path: Path,
) -> None:
    dense_times = np.linspace(0.01, 0.99, 500, dtype=np.float64)
    evaluation_times = np.asarray(DEFAULT_PROTOCOL.times, dtype=np.float64)
    colors = plt.get_cmap("tab10").colors

    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for color, (name, variance) in zip(colors, signal_variances.items(), strict=True):
        axis.plot(
            dense_times,
            effective_snr(variance, dense_times),
            color=color,
            linewidth=2.0,
            label=rf"{name}  ($\lambda_b={variance:.3g}$)",
        )
        axis.scatter(
            evaluation_times,
            effective_snr(variance, evaluation_times),
            color=color,
            s=17,
            zorder=3,
        )

    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.3, label="SNR = 1")
    axis.set(
        xlabel=r"Flow time $t$ (noise $\rightarrow$ data)",
        ylabel=r"Effective SNR  $t^2\lambda_b/(1-t)^2$",
        xlim=(0.0, 1.0),
    )
    axis.set_title(
        "Effective signal-to-noise ratio by wavelet subband\n"
        r"MNIST-32, globally normalized, db2 level 3; "
        r"$Y_t=(1-t)\xi+tY_0$",
        pad=12,
    )
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(
        title=r"Band signal variance $\lambda_b$",
        ncols=2,
        fontsize=9,
        title_fontsize=9,
        loc="upper left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    split = load_mnist_32(
        args.data_root,
        "train",
        max_samples=args.max_images,
        batch_size=1024,
        download=args.download,
    )
    global_mean = float(np.mean(split.images, dtype=np.float64))
    global_std = float(np.std(split.images, dtype=np.float64))
    if not np.isfinite(global_std) or global_std <= 0.0:
        raise RuntimeError("MNIST global standard deviation is invalid.")

    normalized_images = np.ascontiguousarray(
        (split.images.astype(np.float64) - global_mean) / global_std
    )
    transform = WaveletTransform2D(
        DEFAULT_PROTOCOL.wavelet_name,
        DEFAULT_PROTOCOL.decomposition_level,
        DEFAULT_PROTOCOL.mode,
    )
    Y_0, layout = transform.analyze(normalized_images)
    signal_variances = compute_subband_signal_variances(Y_0, layout)
    plot_subband_snr(signal_variances, args.out_path)

    print("mean empirical signal variance by subband:", flush=True)
    for name, variance in signal_variances.items():
        crossing_time = 1.0 / (1.0 + np.sqrt(variance))
        print(
            f"  {name}: lambda_b={variance:.6f}, SNR=1 at t={crossing_time:.4f}",
            flush=True,
        )
    print(f"saved plot to {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
