"""Centralized cross-scale relations for wavelet coefficient indexings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wavelet_score.indexing import WaveletLayout, build_metadata

RelationKind = Literal["parent_child", "scaling_detail"]


@dataclass(frozen=True, order=True, slots=True)
class ScaleInteractionEdge:
    """One unique undirected interaction, stored in coarse-to-fine order."""

    first_index: int
    second_index: int
    kind: RelationKind


def build_layout_scale_interaction_terms(
    layout: WaveletLayout,
) -> tuple[ScaleInteractionEdge, ...]:
    """Build adjacent-scale and scaling/detail edges from a critical layout."""
    metadata = build_metadata(layout)
    details = {
        (item.math_scale_j, item.k1, item.k2, item.orientation): item.flat_index
        for item in metadata
        if item.coefficient_type == "detail"
    }
    edges: set[ScaleInteractionEdge] = set()
    for item in metadata:
        if item.coefficient_type != "detail":
            continue
        parent = details.get(
            (item.math_scale_j - 1, item.k1 // 2, item.k2 // 2, item.orientation)
        )
        if parent is not None:
            edges.add(ScaleInteractionEdge(parent, item.flat_index, "parent_child"))

    scaling = {
        (item.k1, item.k2): item.flat_index
        for item in metadata
        if item.coefficient_type == "scaling"
    }
    coarsest_j = min(key[0] for key in details)
    for (j, k1, k2, _orientation), detail_index in details.items():
        scaling_index = scaling.get((k1, k2))
        if j == coarsest_j and scaling_index is not None:
            edges.add(ScaleInteractionEdge(scaling_index, detail_index, "scaling_detail"))
    return tuple(sorted(edges))


def receptive_indices_from_scale_edges(
    num_coefficients: int,
    edges: tuple[ScaleInteractionEdge, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return self plus every cross-scale neighbor for each output coordinate."""
    visible = [{index} for index in range(num_coefficients)]
    for edge in edges:
        visible[edge.first_index].add(edge.second_index)
        visible[edge.second_index].add(edge.first_index)
    return tuple(tuple(sorted(indices)) for indices in visible)
