"""Exact same-band posterior with fixed or translated source patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


CandidateMode = Literal["fixed", "shared_equal", "shared_full"]


@dataclass(frozen=True)
class SourceConcentration:
    mean_max_source_mass: float
    dominant_source_fraction: float
    num_targets: int
    dominant_source_ids: np.ndarray


def _patches(images: torch.Tensor, radius: int) -> torch.Tensor:
    """Circular same-band patches, [N, positions, 3*(2r+1)^2]."""
    if images.ndim != 4 or images.shape[1] != 3 or radius < 0:
        raise ValueError("images must have shape [N,3,H,W] and radius >= 0")
    width = 2 * radius + 1
    if width > min(images.shape[-2:]):
        raise ValueError("patch is wider than the subband")
    return F.unfold(F.pad(images, (radius,) * 4, mode="circular"), width).transpose(1, 2)


def predict_subband(
    queries: np.ndarray,
    references: np.ndarray,
    time: float,
    *,
    radius: int,
    mode: CandidateMode,
    device: torch.device,
    source_chunk: int = 256,
    target_positions: tuple[int, ...] | None = None,
    equal_seed: int = 1729,
) -> tuple[np.ndarray, SourceConcentration]:
    """Predict clean RGB sites with exact softmax over the specified candidate bank.

    Queries and references are [N,3,H,W]. `shared_equal` uses one fixed,
    reproducibly selected position per source image; `shared_full` uses all
    positions. For a target, concentration sums over source positions first.
    """
    if queries.ndim != 4 or references.ndim != 4 or queries.shape[1:] != references.shape[1:]:
        raise ValueError("queries and references must have matching [N,3,H,W] shape")
    if not 0.0 <= time < 1.0 or source_chunk < 1:
        raise ValueError("invalid time or source chunk size")
    if mode not in ("fixed", "shared_equal", "shared_full"):
        raise ValueError("unknown candidate mode")
    height, width = queries.shape[-2:]
    positions = height * width
    selected = tuple(range(positions)) if target_positions is None else target_positions
    if not selected:
        raise ValueError("at least one target position is required")
    if len(set(selected)) != len(selected) or any(not 0 <= q < positions for q in selected):
        raise ValueError("target positions must be unique and inside the subband")
    predicted = np.full((len(queries), 3, height, width), np.nan, dtype=np.float32)
    query_t = torch.as_tensor(np.ascontiguousarray(queries), device=device, dtype=torch.float32)
    query_patches = _patches(query_t, radius)
    candidate_batches = []
    for start in range(0, len(references), source_chunk):
        stop = min(start + source_chunk, len(references))
        batch = torch.as_tensor(
            np.ascontiguousarray(references[start:stop]), device=device, dtype=torch.float32
        )
        patches = _patches(batch, radius)
        centers = batch.permute(0, 2, 3, 1).reshape(stop - start, positions, 3)
        if mode == "shared_equal":
            ids = torch.arange(start, stop, device=device, dtype=torch.long)
            candidate_positions = ((ids * 104729 + equal_seed) % positions).long()
            rows = torch.arange(stop - start, device=device)
            patches = patches[rows, candidate_positions]
            centers = centers[rows, candidate_positions]
        elif mode == "shared_full":
            patches = patches.reshape(-1, patches.shape[-1])
            centers = centers.reshape(-1, 3)
        candidate_batches.append((start, stop, patches, centers))
    max_source_masses: list[float] = []
    modal_source_counts = np.zeros((len(queries), len(references)), dtype=np.int64)
    dominant_source_ids = np.empty((len(queries), len(selected)), dtype=np.int64)
    total_targets = 0
    denominator = 2.0 * (1.0 - time) ** 2

    for target_number, target in enumerate(selected):
        query = query_patches[:, target].to(dtype=torch.float64)
        query_norm = (query * query).sum(dim=-1, keepdim=True)
        count = len(queries)
        running_max = torch.full((count,), -torch.inf, device=device, dtype=torch.float64)
        mass = torch.zeros(count, device=device, dtype=torch.float64)
        numerator = torch.zeros((count, 3), device=device, dtype=torch.float64)
        best_source_logit = torch.full((count,), -torch.inf, device=device, dtype=torch.float64)
        best_source_id = torch.full((count,), -1, device=device, dtype=torch.long)

        for start, stop, stored_patches, stored_centers in candidate_batches:
            if mode == "fixed":
                patches = stored_patches[:, target].to(dtype=torch.float64)
                centers = stored_centers[:, target].to(dtype=torch.float64)
                candidate_count = 1
            elif mode == "shared_equal":
                patches = stored_patches.to(dtype=torch.float64)
                centers = stored_centers.to(dtype=torch.float64)
                candidate_count = 1
            else:
                patches = stored_patches.to(dtype=torch.float64)
                centers = stored_centers.to(dtype=torch.float64)
                candidate_count = positions

            patch_norm = (patches * patches).sum(dim=-1).unsqueeze(0)
            squared = (query_norm + time * time * patch_norm - 2.0 * time * (query @ patches.T)).clamp_min_(0.0)
            logits = -squared / denominator
            batch_max = logits.amax(dim=1)
            new_max = torch.maximum(running_max, batch_max)
            old_scale = torch.exp(running_max - new_max)
            weights = torch.exp(logits - new_max[:, None])
            mass = mass * old_scale + weights.sum(dim=1)
            numerator = numerator * old_scale[:, None] + weights @ centers
            running_max = new_max

            source_scores = torch.logsumexp(logits.reshape(count, stop - start, candidate_count), dim=-1)
            chunk_best, chunk_index = source_scores.max(dim=1)
            replace = chunk_best > best_source_logit
            best_source_logit = torch.where(replace, chunk_best, best_source_logit)
            best_source_id = torch.where(replace, chunk_index + start, best_source_id)

        if not torch.isfinite(numerator).all() or not torch.all(mass > 0):
            raise FloatingPointError("non-finite shared posterior")
        predicted[:, :, target // width, target % width] = (numerator / mass[:, None]).float().cpu().numpy()
        source_mass = torch.exp(best_source_logit - running_max) / mass
        max_source_masses.extend(source_mass.cpu().tolist())
        chosen = best_source_id.cpu().numpy()
        dominant_source_ids[:, target_number] = chosen
        np.add.at(modal_source_counts, (np.arange(count), chosen), 1)
        total_targets += count

    concentration = SourceConcentration(
        mean_max_source_mass=float(np.mean(max_source_masses)),
        dominant_source_fraction=float(np.mean(modal_source_counts.max(axis=1) / len(selected))),
        num_targets=total_targets,
        dominant_source_ids=dominant_source_ids,
    )
    return predicted, concentration
