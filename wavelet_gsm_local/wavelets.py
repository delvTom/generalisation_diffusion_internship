"""MNIST loading and audited two-level Haar analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pywt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

BAND_NAMES = ("H1", "V1", "D1", "H2", "V2", "D2")


@dataclass(frozen=True, slots=True)
class MNISTImages:
    images: np.ndarray  # [N,H,W], float32 in [0,1]
    labels: np.ndarray  # [N], int64
    split: str

    def __post_init__(self) -> None:
        if self.images.ndim != 3 or self.images.shape[1] != self.images.shape[2]:
            raise ValueError("MNIST images must have square shape [N,H,H].")
        if self.images.shape[1] % 4 != 0:
            raise ValueError("MNIST image dimensions must be divisible by four.")
        if self.images.dtype != np.float32:
            raise TypeError("MNIST images must use float32.")
        if self.labels.shape != (len(self.images),):
            raise ValueError("MNIST labels and images have inconsistent lengths.")
        if self.images.size and (self.images.min() < 0.0 or self.images.max() > 1.0):
            raise ValueError("MNIST images must lie in [0,1].")


def load_mnist(
    root: str | Path,
    split: str,
    *,
    image_size: int = 28,
    batch_size: int = 1024,
    max_images: int | None = None,
    download: bool = False,
) -> MNISTImages:
    """Load MNIST in [0,1], optionally using Emma's bilinear resizing protocol."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")
    if image_size < 4 or image_size % 4 != 0:
        raise ValueError("image_size must be at least four and divisible by four.")
    transform_steps: list[object] = []
    if image_size != 28:
        transform_steps.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        )
    transform_steps.append(transforms.ToTensor())
    dataset = datasets.MNIST(
        root=str(root),
        train=split == "train",
        transform=transforms.Compose(transform_steps),
        download=download,
    )
    if max_images is not None and max_images < len(dataset):
        dataset = Subset(dataset, range(max_images))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    image_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        image_batches.append(images[:, 0].numpy().astype(np.float32, copy=False))
        label_batches.append(labels.numpy().astype(np.int64, copy=False))
    return MNISTImages(
        images=np.ascontiguousarray(np.concatenate(image_batches)),
        labels=np.ascontiguousarray(np.concatenate(label_batches)),
        split=split,
    )


def load_mnist_28(
    root: str | Path,
    split: str,
    *,
    batch_size: int = 1024,
    max_images: int | None = None,
    download: bool = False,
) -> MNISTImages:
    """Backward-compatible native 28x28 loader."""
    return load_mnist(
        root,
        split,
        image_size=28,
        batch_size=batch_size,
        max_images=max_images,
        download=download,
    )


@dataclass(frozen=True, slots=True)
class HaarLevel2Batch:
    approximation: np.ndarray  # A2 [N,H/4,W/4]
    bands: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.approximation.ndim != 3:
            raise ValueError("The approximation must have shape [N,H/4,W/4].")
        approximation_shape = self.approximation.shape[1:]
        level_1_shape = tuple(2 * value for value in approximation_shape)
        expected = {
            "H1": level_1_shape,
            "V1": level_1_shape,
            "D1": level_1_shape,
            "H2": approximation_shape,
            "V2": approximation_shape,
            "D2": approximation_shape,
        }
        if tuple(self.bands) != BAND_NAMES:
            raise ValueError(f"Expected bands ordered as {BAND_NAMES}.")
        for name, shape in expected.items():
            if self.bands[name].shape != (len(self.approximation), *shape):
                raise ValueError(f"Unexpected shape for {name}: {self.bands[name].shape}.")


def analyze_haar_level2(images: np.ndarray) -> HaarLevel2Batch:
    """Apply the orthogonal periodized Haar DWT to a square image batch."""
    if images.ndim != 3 or images.shape[1] != images.shape[2]:
        raise ValueError("images must have square shape [N,H,H].")
    if images.shape[1] % 4 != 0:
        raise ValueError("image dimensions must be divisible by four.")
    if images.dtype not in (np.float32, np.float64):
        raise TypeError("images must use float32 or float64.")
    cA2, details_level_2, details_level_1 = pywt.wavedec2(
        images, wavelet="haar", mode="periodization", level=2, axes=(-2, -1)
    )
    cH2, cV2, cD2 = details_level_2
    cH1, cV1, cD1 = details_level_1
    return HaarLevel2Batch(
        approximation=cA2,
        bands={
            "H1": cH1,
            "V1": cV1,
            "D1": cD1,
            "H2": cH2,
            "V2": cV2,
            "D2": cD2,
        },
    )


def synthesize_haar_level2(coefficients: HaarLevel2Batch) -> np.ndarray:
    bands = coefficients.bands
    tree = [
        coefficients.approximation,
        (bands["H2"], bands["V2"], bands["D2"]),
        (bands["H1"], bands["V1"], bands["D1"]),
    ]
    return pywt.waverec2(
        tree, wavelet="haar", mode="periodization", axes=(-2, -1)
    )


def audit_transform(
    images: np.ndarray, *, noise_samples: int, seed: int
) -> dict[str, float | int]:
    """Check reconstruction, Parseval, and empirical whitening of Gaussian noise."""
    subset = np.ascontiguousarray(images[: min(len(images), 256)])
    transformed = analyze_haar_level2(subset)
    reconstructed = synthesize_haar_level2(transformed)
    relative_reconstruction = np.linalg.norm((reconstructed - subset).reshape(len(subset), -1), axis=1)
    relative_reconstruction /= np.maximum(
        np.linalg.norm(subset.reshape(len(subset), -1), axis=1), 1e-15
    )
    coefficients = np.concatenate(
        [
            transformed.approximation.reshape(len(subset), -1),
            *(transformed.bands[name].reshape(len(subset), -1) for name in BAND_NAMES),
        ],
        axis=1,
    )
    pixel_energy = np.sum(subset**2, axis=(1, 2))
    coefficient_energy = np.sum(coefficients**2, axis=1)
    parseval_error = np.abs(pixel_energy - coefficient_energy) / np.maximum(pixel_energy, 1e-15)

    generator = np.random.default_rng(seed)
    noise = generator.standard_normal((noise_samples, *images.shape[1:]))
    noise_wavelets = analyze_haar_level2(noise)
    noise_flat = np.concatenate(
        [
            noise_wavelets.approximation.reshape(noise_samples, -1),
            *(noise_wavelets.bands[name].reshape(noise_samples, -1) for name in BAND_NAMES),
        ],
        axis=1,
    )
    # A full 4096x4096 empirical covariance is unnecessary for this diagnostic.
    # Keep all coordinates at 28x28 and a deterministic spread of 784 at 64x64.
    audit_dimension = min(noise_flat.shape[1], 784)
    if audit_dimension < noise_flat.shape[1]:
        indices = np.linspace(
            0, noise_flat.shape[1] - 1, audit_dimension, dtype=np.int64
        )
        noise_flat = noise_flat[:, indices]
    noise_mean = noise_flat.mean(axis=0, keepdims=True)
    noise_flat -= noise_mean
    covariance = noise_flat.T @ noise_flat / noise_samples
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return {
        "max_relative_reconstruction_error": float(relative_reconstruction.max()),
        "max_relative_parseval_error": float(parseval_error.max()),
        "white_noise_mean_abs_max": float(np.abs(noise_mean).max()),
        "white_noise_variance_mean": float(np.diag(covariance).mean()),
        "white_noise_variance_max_abs_error": float(np.abs(np.diag(covariance) - 1.0).max()),
        "white_noise_off_diagonal_rms": float(np.sqrt(np.mean(off_diagonal**2))),
        "white_noise_off_diagonal_abs_max": float(np.abs(off_diagonal).max()),
        "white_noise_audited_dimension": int(audit_dimension),
        "total_coefficient_dimension": int(coefficients.shape[1]),
    }
