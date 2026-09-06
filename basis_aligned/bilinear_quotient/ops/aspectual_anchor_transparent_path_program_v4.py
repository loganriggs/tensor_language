#!/usr/bin/env python3
"""Executable source- and MLP-resolved suffix extension of aspectual program v3."""

from __future__ import annotations

import aspectual_anchor_transparent_path_program_v3 as v3


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v4"
MLP_FACTORS = ("left_change", "right_change", "bilinear_interaction")
SUFFIX_MLP_FACTORS_BY_BOUNDARY = {
    11: ("left_change", "right_change"),
    15: ("left_change", "bilinear_interaction"),
}

# Stable v3 API re-exports.
ProgramInputError = v3.ProgramInputError
ATTENTION5_HEADS = v3.ATTENTION5_HEADS
ATTENTION9_HEADS = v3.ATTENTION9_HEADS
SOURCE_NAMES = v3.SOURCE_NAMES
ROUTE_CROSSING_BOUNDARIES = v3.ROUTE_CROSSING_BOUNDARIES
SUFFIX_HEAD_BY_BOUNDARY = v3.SUFFIX_HEAD_BY_BOUNDARY
SUFFIX_SOURCE_BANK_BY_BOUNDARY = v3.SUFFIX_SOURCE_BANK_BY_BOUNDARY
ALL_SOURCE_ROLES = v3.ALL_SOURCE_ROLES
mlp4_hidden_response = v3.mlp4_hidden_response
linear_without_bias = v3.linear_without_bias
mlp4_write = v3.mlp4_write
attention_source_term = v3.attention_source_term
attention_source_delta = v3.attention_source_delta
suffix_attention_source_delta = v3.suffix_attention_source_delta
project_selected_head_deltas = v3.project_selected_head_deltas
crossing_delta = v3.crossing_delta
suffix_crossing_delta = v3.suffix_crossing_delta
write_query_delta = v3.write_query_delta


def bilinear_hidden_delta(
    left_base,
    right_base,
    left_hybrid,
    right_hybrid,
    *,
    factors: tuple[str, ...],
):
    """Compose a unique subset of exact Left/Right bilinear response terms."""
    if len(set(factors)) != len(factors) or any(factor not in MLP_FACTORS for factor in factors):
        raise ProgramInputError("factors must be a unique subset of the three bilinear terms")
    delta_left = left_hybrid - left_base
    delta_right = right_hybrid - right_base
    terms = {
        "left_change": delta_left * right_base,
        "right_change": left_base * delta_right,
        "bilinear_interaction": delta_left * delta_right,
    }
    hidden = None
    for factor in MLP_FACTORS:
        if factor in factors:
            hidden = terms[factor] if hidden is None else hidden + terms[factor]
    if hidden is None:
        return left_base - left_base
    return hidden


def suffix_mlp_write_delta(
    left_base,
    right_base,
    left_hybrid,
    right_hybrid,
    down_weight,
    *,
    boundary: int,
):
    """Project the frozen two-term MLP11 or MLP15 response without bias."""
    if boundary not in SUFFIX_MLP_FACTORS_BY_BOUNDARY:
        raise ProgramInputError("boundary must be 11 or 15")
    hidden = bilinear_hidden_delta(
        left_base, right_base, left_hybrid, right_hybrid,
        factors=SUFFIX_MLP_FACTORS_BY_BOUNDARY[boundary],
    )
    return linear_without_bias(hidden, down_weight)


def compiled_suffix_crossing_delta(
    lambda0,
    base_resid,
    hybrid_resid,
    projected_source_attention_delta,
    left_base,
    right_base,
    left_hybrid,
    right_hybrid,
    down_weight,
    *,
    boundary: int,
):
    """Compose carried state, source-resolved attention, and compressed MLP response."""
    return (
        lambda0 * (hybrid_resid - base_resid)
        + projected_source_attention_delta
        + suffix_mlp_write_delta(
            left_base, right_base, left_hybrid, right_hybrid,
            down_weight, boundary=boundary,
        )
    )


def program_manifest() -> dict[str, object]:
    """Return the exact stable inventory exposed by executable program v4."""
    manifest = dict(v3.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "suffix_mlp_factors_by_boundary": dict(SUFFIX_MLP_FACTORS_BY_BOUNDARY),
        "runtime_dependencies": (
            "checkpoint weights",
            "paired base/donor MLP4 states",
            "paired base/hybrid attention captures",
            "paired base/hybrid suffix residual and bilinear MLP states",
            "native checkpoint suffix",
        ),
    })
    return manifest
