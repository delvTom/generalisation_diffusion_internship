"""Audited MNIST-32 loading for the wavelet-score experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


@dataclass(frozen=True, slots=True)
class MNISTSplit:
    """One official MNIST split represented as NumPy arrays."""

    images: np.ndarray  # [N,32,32], float32 in [0,1]
    labels: np.ndarray  # [N], int64
    split: str

    def __post_init__(self) -> None:
        if self.images.ndim != 3 or self.images.shape[1:] != (32, 32):
            raise ValueError("images must have shape [N,32,32].")
        if self.images.dtype != np.dtype(np.float32):
            raise TypeError("images must use float32.")
        if self.labels.shape != (self.images.shape[0],):
            raise ValueError("labels must have shape [N].")
        if self.labels.dtype != np.dtype(np.int64):
            raise TypeError("labels must use int64.")
        if self.split not in ("train", "test"):
            raise ValueError("split must be train or test.")
        if self.images.size and (
            float(self.images.min()) < 0.0 or float(self.images.max()) > 1.0
        ):
            raise ValueError("images must lie in [0,1].")


def mnist_32_transform() -> transforms.Compose:
    """Resize MNIST to 32x32 and convert uint8 pixels to float32 in [0,1]."""
    return transforms.Compose(
        [
            transforms.Resize(
                (32, 32),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )


def load_mnist_32(
    root: str | Path,
    split: str,
    batch_size: int = 1024,
    num_workers: int = 0,
    max_samples: int | None = None,
    download: bool = False,
) -> MNISTSplit:
    """Load an official MNIST split as ordered arrays without normalization to [-1,1]."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")

    dataset = datasets.MNIST(
        root=str(root),
        train=split == "train",
        download=download,
        transform=mnist_32_transform(),
    )
    if max_samples is not None and max_samples < len(dataset):
        dataset = Subset(dataset, range(max_samples))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    image_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for images, labels in loader:
        if images.ndim != 4 or images.shape[1:] != (1, 32, 32):
            raise RuntimeError(f"Unexpected transformed MNIST shape: {tuple(images.shape)}.")
        image_batches.append(images[:, 0].numpy())
        label_batches.append(labels.numpy())

    images_array = np.ascontiguousarray(np.concatenate(image_batches, axis=0), dtype=np.float32)
    labels_array = np.ascontiguousarray(np.concatenate(label_batches, axis=0), dtype=np.int64)
    return MNISTSplit(images=images_array, labels=labels_array, split=split)
