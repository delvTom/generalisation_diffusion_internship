"""Exact masked empirical conditional expectation for wavelet Flow Matching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from wavelet_score.indexing import WaveletLayout
from wavelet_score.mask_groups import MaskFamily, WaveletMaskGroup, build_mask_groups
from wavelet_score.transforms import WaveletTransform2D


def brute_predict_clean(
    reference_Y0: np.ndarray,
    Y_t: np.ndarray,
    t: float,
    groups: tuple[WaveletMaskGroup, ...],
) -> np.ndarray:
    """Small NumPy reference implementation used only by tests."""
    if t == 0.0:
        return np.broadcast_to(reference_Y0.mean(axis=0), Y_t.shape).copy()
    variance = (1.0 - t) ** 2
    result = np.empty_like(Y_t)
    for group in groups:
        query = Y_t[:, group.input_indices]
        centers = t * reference_Y0[:, group.input_indices]
        logits = -np.sum((query[:, None] - centers[None]) ** 2, axis=-1) / (2.0 * variance)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        result[:, group.output_indices] = weights @ reference_Y0[:, group.output_indices]
    return result


@dataclass(slots=True)
class MaskedEmpiricalWaveletVelocity:
    family: MaskFamily
    radius: int = 1
    reference_chunk_size: int = 256
    query_batch_size: int = 16
    group_chunk_size: int = 32
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    endpoint_epsilon: float = 1e-3
    reference_Y0_: torch.Tensor | None = field(default=None, init=False, repr=False)
    groups_: tuple[WaveletMaskGroup, ...] | None = field(default=None, init=False, repr=False)
    layout_: WaveletLayout | None = field(default=None, init=False, repr=False)

    def fit(
        self, reference_Y0: np.ndarray, layout: WaveletLayout
    ) -> MaskedEmpiricalWaveletVelocity:
        if reference_Y0.ndim != 2 or reference_Y0.shape[1] != layout.num_coefficients:
            raise ValueError("reference_Y0 must have shape [N_ref,d] matching layout.")
        self._fit_reference_bank(reference_Y0, layout)
        return self.set_groups(build_mask_groups(layout, self.family, self.radius))

    def fit_with_groups(
        self,
        reference_Y0: np.ndarray,
        layout: WaveletLayout,
        groups: tuple[WaveletMaskGroup, ...],
    ) -> MaskedEmpiricalWaveletVelocity:
        """Fit one reference bank and attach externally constructed mask groups.

        The feature dimension may be an integer multiple of the scalar wavelet
        layout. This supports channel-major RGB coefficients while preserving a
        single spatial layout shared by all channels.
        """
        if (
            reference_Y0.ndim != 2
            or reference_Y0.shape[1] % layout.num_coefficients != 0
        ):
            raise ValueError(
                "reference_Y0 width must be an integer multiple of the layout size."
            )
        self._fit_reference_bank(reference_Y0, layout)
        return self.set_groups(groups)

    def _fit_reference_bank(
        self, reference_Y0: np.ndarray, layout: WaveletLayout
    ) -> None:
        if reference_Y0.ndim != 2:
            raise ValueError("reference_Y0 must have shape [N_ref,d].")
        if len(reference_Y0) == 0:
            raise ValueError("At least one reference is required.")
        if min(self.reference_chunk_size, self.query_batch_size, self.group_chunk_size) < 1:
            raise ValueError("All chunk sizes must be positive.")
        if not 0.0 < self.endpoint_epsilon < 1.0:
            raise ValueError("endpoint_epsilon must lie in (0,1).")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError("Only torch.float32 and torch.float64 are supported.")
        self.reference_Y0_ = torch.as_tensor(
            np.ascontiguousarray(reference_Y0), device=self.device, dtype=self.dtype
        )
        self.layout_ = layout

    def set_groups(
        self, groups: tuple[WaveletMaskGroup, ...]
    ) -> MaskedEmpiricalWaveletVelocity:
        """Replace masks without copying the fitted reference bank."""
        if self.reference_Y0_ is None or self.layout_ is None:
            raise RuntimeError("The reference bank must be fitted before setting groups.")
        if not groups:
            raise ValueError("At least one mask group is required.")
        dimension = self.reference_Y0_.shape[1]
        self._validate_groups(groups, dimension, require_complete=True)
        self.groups_ = tuple(groups)
        return self

    @staticmethod
    def _validate_groups(
        groups: tuple[WaveletMaskGroup, ...],
        dimension: int,
        *,
        require_complete: bool,
    ) -> np.ndarray:
        if not groups:
            raise ValueError("At least one mask group is required.")
        for group in groups:
            if (
                np.any(group.input_indices < 0)
                or np.any(group.input_indices >= dimension)
                or np.any(group.output_indices < 0)
                or np.any(group.output_indices >= dimension)
            ):
                raise ValueError("Mask group indices lie outside the wavelet layout.")
        outputs = np.concatenate([group.output_indices for group in groups])
        if len(np.unique(outputs)) != len(outputs):
            raise ValueError("Mask groups cannot predict an output more than once.")
        if require_complete and sorted(outputs.tolist()) != list(range(dimension)):
            raise ValueError("Mask groups must cover every output exactly once.")
        return np.sort(outputs)

    def _check_ready(self) -> tuple[torch.Tensor, tuple[WaveletMaskGroup, ...]]:
        if self.reference_Y0_ is None or self.groups_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        return self.reference_Y0_, self.groups_

    @property
    def reference_tensor(self) -> torch.Tensor:
        """Return the single fitted reference tensor shared across mask schedules."""
        if self.reference_Y0_ is None:
            raise RuntimeError("The estimator must be fitted before reading references.")
        return self.reference_Y0_

    @torch.no_grad()
    def predict_clean(self, Y_t: np.ndarray, t: float) -> np.ndarray:
        references, groups = self._check_ready()
        output_indices, result = self._predict_clean_groups(Y_t, t, groups)
        expected = np.arange(references.shape[1], dtype=np.int64)
        if not np.array_equal(output_indices, expected):
            raise RuntimeError("Fitted groups unexpectedly omit output coordinates.")
        return result

    @torch.no_grad()
    def predict_clean_subset(
        self,
        Y_t: np.ndarray,
        t: float,
        groups: tuple[WaveletMaskGroup, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict only the unique outputs covered by ``groups``.

        Returns sorted global output indices and a compact array whose columns
        follow that order. Queries still contain the complete coefficient
        vector because the masks may observe coordinates outside the outputs.
        """
        if self.reference_Y0_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        return self._predict_clean_groups(Y_t, t, groups)

    def _predict_clean_groups(
        self,
        Y_t: np.ndarray,
        t: float,
        groups: tuple[WaveletMaskGroup, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        references = self.reference_tensor
        if Y_t.ndim != 2 or Y_t.shape[1] != references.shape[1]:
            raise ValueError("Y_t must have shape [N_query,d].")
        if not 0.0 <= t <= 1.0 - self.endpoint_epsilon:
            raise ValueError("t must lie in [0, 1-endpoint_epsilon].")
        output_indices = self._validate_groups(
            groups, references.shape[1], require_complete=False
        )
        output_dtype = np.float64 if self.dtype == torch.float64 else np.float32
        if t == 0.0:
            tensor_indices = torch.as_tensor(
                output_indices, device=self.device, dtype=torch.long
            )
            mean = (
                references[:, tensor_indices]
                .mean(dim=0)
                .cpu()
                .numpy()
                .astype(output_dtype, copy=False)
            )
            return output_indices, np.broadcast_to(
                mean, (len(Y_t), len(output_indices))
            ).copy()

        buckets: dict[tuple[int, int], list[WaveletMaskGroup]] = defaultdict(list)
        for group in groups:
            buckets[(len(group.input_indices), len(group.output_indices))].append(group)
        result = np.empty((len(Y_t), len(output_indices)), dtype=output_dtype)
        output_positions = np.full(references.shape[1], -1, dtype=np.int64)
        output_positions[output_indices] = np.arange(len(output_indices), dtype=np.int64)
        one_minus_t = torch.as_tensor(1.0 - t, device=self.device, dtype=self.dtype)

        for query_start in range(0, len(Y_t), self.query_batch_size):
            query_stop = min(query_start + self.query_batch_size, len(Y_t))
            query = torch.as_tensor(
                Y_t[query_start:query_stop], device=self.device, dtype=self.dtype
            )
            query_result = torch.empty(
                (len(query), len(output_indices)),
                device=self.device,
                dtype=self.dtype,
            )
            for bucket in buckets.values():
                for group_start in range(0, len(bucket), self.group_chunk_size):
                    selected = bucket[group_start : group_start + self.group_chunk_size]
                    input_indices = torch.as_tensor(
                        np.stack([group.input_indices for group in selected]),
                        device=self.device,
                        dtype=torch.long,
                    )
                    local_output_indices = torch.as_tensor(
                        np.stack(
                            [output_positions[group.output_indices] for group in selected]
                        ),
                        device=self.device,
                        dtype=torch.long,
                    )
                    selected_output_indices = torch.as_tensor(
                        np.stack([group.output_indices for group in selected]),
                        device=self.device,
                        dtype=torch.long,
                    )
                    query_inputs = query[:, input_indices]  # [B,G,K]
                    running_max = torch.full(
                        query_inputs.shape[:2],
                        -torch.inf,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    denominator = torch.zeros_like(running_max)
                    numerator = torch.zeros(
                        (*running_max.shape, selected_output_indices.shape[1]),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    for reference_start in range(0, len(references), self.reference_chunk_size):
                        reference_stop = min(
                            reference_start + self.reference_chunk_size, len(references)
                        )
                        reference = references[reference_start:reference_stop]
                        reference_inputs = reference[:, input_indices]  # [R,G,K]
                        residual = query_inputs[:, None] - t * reference_inputs[None]
                        logits = -residual.square().sum(dim=-1) / (
                            2.0 * one_minus_t.square()
                        )  # [B,R,G]
                        chunk_max = logits.max(dim=1).values
                        new_max = torch.maximum(running_max, chunk_max)
                        old_scale = torch.exp(running_max - new_max)
                        weights = torch.exp(logits - new_max[:, None])
                        denominator = denominator * old_scale + weights.sum(dim=1)
                        reference_outputs = reference[:, selected_output_indices]  # [R,G,O]
                        numerator = numerator * old_scale[..., None] + torch.einsum(
                            "brg,rgo->bgo", weights, reference_outputs
                        )
                        running_max = new_max
                    values = numerator / denominator[..., None]
                    query_result.scatter_(
                        1,
                        local_output_indices.reshape(1, -1).expand(len(query), -1),
                        values.reshape(len(query), -1),
                    )
            result[query_start:query_stop] = query_result.cpu().numpy()
        return output_indices, result

    def predict_velocity(self, Y_t: np.ndarray, t: float) -> np.ndarray:
        clean = self.predict_clean(Y_t, t)
        return (clean - Y_t) / (1.0 - t)

    def predict_velocity_subset(
        self,
        Y_t: np.ndarray,
        t: float,
        groups: tuple[WaveletMaskGroup, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict velocity for only the outputs covered by ``groups``."""
        output_indices, clean = self.predict_clean_subset(Y_t, t, groups)
        return output_indices, (clean - Y_t[:, output_indices]) / (1.0 - t)

    def predict_velocity_pixels(
        self,
        Y_t: np.ndarray,
        t: float,
        transform: WaveletTransform2D,
    ) -> np.ndarray:
        if self.layout_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")
        if self.reference_tensor.shape[1] != self.layout_.num_coefficients:
            raise RuntimeError(
                "predict_velocity_pixels only supports one scalar wavelet channel."
            )
        return transform.synthesize(self.predict_velocity(Y_t, t), self.layout_)
