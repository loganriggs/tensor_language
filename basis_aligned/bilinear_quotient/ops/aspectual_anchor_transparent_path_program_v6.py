#!/usr/bin/env python3
"""Sparse executable suffix recurrence extending valid aspectual program v5."""

from __future__ import annotations

import aspectual_anchor_transparent_path_program_v5 as v5


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v6"
SUFFIX_BOUNDARIES = tuple(range(10, 18))
SUFFIX_SOURCE_BOUNDARIES = (11, 15)
SUFFIX_MLP_FACTORS_BY_BOUNDARY = {
    11: ("left_change", "right_change"),
    12: ("left_change", "right_change"),
    14: ("left_change", "right_change"),
    15: ("left_change", "bilinear_interaction"),
}

ProgramInputError = v5.ProgramInputError
MLP_FACTORS = v5.MLP_FACTORS
ATTENTION5_HEADS = v5.ATTENTION5_HEADS
ATTENTION9_HEADS = v5.ATTENTION9_HEADS
SOURCE_NAMES = v5.SOURCE_NAMES
ROUTE_CROSSING_BOUNDARIES = v5.ROUTE_CROSSING_BOUNDARIES
SUFFIX_HEAD_BY_BOUNDARY = v5.SUFFIX_HEAD_BY_BOUNDARY
SUFFIX_SOURCE_BANK_BY_BOUNDARY = v5.SUFFIX_SOURCE_BANK_BY_BOUNDARY
ALL_SOURCE_ROLES = v5.ALL_SOURCE_ROLES
mlp4_hidden_response = v5.mlp4_hidden_response
linear_without_bias = v5.linear_without_bias
mlp4_write = v5.mlp4_write
attention_source_term = v5.attention_source_term
attention_source_delta = v5.attention_source_delta
suffix_attention_source_delta = v5.suffix_attention_source_delta
project_selected_head_deltas = v5.project_selected_head_deltas
crossing_delta = v5.crossing_delta
suffix_crossing_delta = v5.suffix_crossing_delta
write_query_delta = v5.write_query_delta
bilinear_hidden_delta = v5.bilinear_hidden_delta


def selected_suffix_mlp_write_delta(
    left_base,
    right_base,
    left_hybrid,
    right_hybrid,
    down_weight,
    *,
    boundary: int,
):
    """Project the frozen response terms for one compiled suffix MLP."""
    if boundary not in SUFFIX_MLP_FACTORS_BY_BOUNDARY:
        raise ProgramInputError("boundary must be one of 11, 12, 14, or 15")
    hidden = bilinear_hidden_delta(
        left_base, right_base, left_hybrid, right_hybrid,
        factors=SUFFIX_MLP_FACTORS_BY_BOUNDARY[boundary],
    )
    return linear_without_bias(hidden, down_weight)


def compiled_sparse_suffix_delta(
    initial_delta,
    *,
    lambda0_by_boundary: dict[int, object],
    source_attention_delta_by_boundary: dict[int, object],
    mlp_states_by_boundary: dict[int, tuple[object, object, object, object]],
    down_weight_by_boundary: dict[int, object],
):
    """Propagate resid10 query delta through the explicit sparse block10-17 recurrence."""
    if set(lambda0_by_boundary) != set(SUFFIX_BOUNDARIES):
        raise ProgramInputError("lambda0 map must contain each boundary 10 through 17 exactly")
    if set(source_attention_delta_by_boundary) != set(SUFFIX_SOURCE_BOUNDARIES):
        raise ProgramInputError("source-attention map must contain boundaries 11 and 15 exactly")
    if set(mlp_states_by_boundary) != set(SUFFIX_MLP_FACTORS_BY_BOUNDARY):
        raise ProgramInputError("MLP-state map must contain boundaries 11, 12, 14, and 15 exactly")
    if set(down_weight_by_boundary) != set(SUFFIX_MLP_FACTORS_BY_BOUNDARY):
        raise ProgramInputError("Down-weight map must contain boundaries 11, 12, 14, and 15 exactly")
    if any(len(states) != 4 for states in mlp_states_by_boundary.values()):
        raise ProgramInputError("each MLP-state entry must be (left_base,right_base,left_hybrid,right_hybrid)")

    delta = initial_delta
    for boundary in SUFFIX_BOUNDARIES:
        delta = lambda0_by_boundary[boundary] * delta
        if boundary in SUFFIX_SOURCE_BOUNDARIES:
            delta = delta + source_attention_delta_by_boundary[boundary]
        if boundary in SUFFIX_MLP_FACTORS_BY_BOUNDARY:
            states = mlp_states_by_boundary[boundary]
            delta = delta + selected_suffix_mlp_write_delta(
                states[0], states[1], states[2], states[3], down_weight_by_boundary[boundary],
                boundary=boundary,
            )
    return delta


def program_manifest() -> dict[str, object]:
    """Return the stable inventory and remaining dependency boundary for v6."""
    manifest = dict(v5.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "compiled_suffix_boundaries": SUFFIX_BOUNDARIES,
        "suffix_source_boundaries": SUFFIX_SOURCE_BOUNDARIES,
        "suffix_mlp_factors_by_boundary": dict(SUFFIX_MLP_FACTORS_BY_BOUNDARY),
        "suffix_recurrence_evidence": "post-outcome block repair plus conditional component/bilinear resolution",
        "runtime_dependencies": (
            "checkpoint weights and lambda0 scalars",
            "paired base/donor MLP4 states",
            "paired base/hybrid attention source terms",
            "paired base/hybrid bilinear MLP states at blocks11,12,14,15",
            "base resid18 state and exact final normalization/unembedding",
        ),
    })
    return manifest
