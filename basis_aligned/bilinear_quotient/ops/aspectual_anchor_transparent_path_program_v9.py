#!/usr/bin/env python3
"""Rank-one carrier projection extension of transferred aspectual program v8."""

from __future__ import annotations

import torch

import aspectual_anchor_transparent_path_program_v8 as v8


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v9"
RANK1_BASIS_SHA256 = "123c6e098fcccf68bd9b881bb81c6b95858a258baa688b79a947a3043bb61e39"

ProgramInputError = v8.ProgramInputError
SUFFIX_BOUNDARIES = v8.SUFFIX_BOUNDARIES
SUFFIX_SOURCE_BOUNDARIES = v8.SUFFIX_SOURCE_BOUNDARIES
SUFFIX_MLP_FACTORS_BY_BOUNDARY = v8.SUFFIX_MLP_FACTORS_BY_BOUNDARY
TRANSFER_CONSTRUCTIONS = v8.TRANSFER_CONSTRUCTIONS
compiled_sparse_suffix_delta = v8.compiled_sparse_suffix_delta
exact_final_logits = v8.exact_final_logits
exact_scored_pair = v8.exact_scored_pair


def carrier_amplitude(delta18, basis):
    """Return the signed coordinate of a resid18 displacement on the released carrier."""
    if not isinstance(delta18, torch.Tensor) or not isinstance(basis, torch.Tensor):
        raise ProgramInputError("delta18 and basis must be torch tensors")
    q = basis.reshape(-1)
    if delta18.ndim != 1 or q.shape != delta18.shape:
        raise ProgramInputError("delta18 and flattened basis must be equal-length vectors")
    if abs(float(q.float().norm()) - 1.0) > 1.0e-4:
        raise ProgramInputError("basis must be unit norm")
    return torch.dot(delta18, q)


def rank1_carrier_projection(delta18, basis):
    """Project a resid18 displacement onto the released one-dimensional carrier."""
    q = basis.reshape(-1)
    return q * carrier_amplitude(delta18, q)


def compiled_rank1_suffix_scored_pair(
    base_resid18,
    initial_delta,
    basis,
    lm_head,
    *,
    answer_id: int,
    foil_id: int,
    lambda0_by_boundary: dict[int, object],
    source_attention_delta_by_boundary: dict[int, object],
    mlp_states_by_boundary: dict[int, tuple[object, object, object, object]],
    down_weight_by_boundary: dict[int, object],
):
    """Execute v8, retain its rank-one carrier displacement, and score answer versus foil."""
    if not isinstance(base_resid18, torch.Tensor) or not isinstance(initial_delta, torch.Tensor):
        raise ProgramInputError("base_resid18 and initial_delta must be torch tensors")
    if base_resid18.shape != initial_delta.shape:
        raise ProgramInputError("base_resid18 and initial_delta must have identical shapes")
    delta18 = compiled_sparse_suffix_delta(
        initial_delta,
        lambda0_by_boundary=lambda0_by_boundary,
        source_attention_delta_by_boundary=source_attention_delta_by_boundary,
        mlp_states_by_boundary=mlp_states_by_boundary,
        down_weight_by_boundary=down_weight_by_boundary,
    )
    projected = rank1_carrier_projection(delta18, basis)
    return exact_scored_pair(base_resid18 + projected, lm_head, answer_id=answer_id, foil_id=foil_id)


def program_manifest() -> dict[str, object]:
    """Return the v8 program with the independently fitted causal carrier operation."""
    manifest = dict(v8.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "rank1_basis_sha256": RANK1_BASIS_SHA256,
        "carrier_site": "resid:18",
        "carrier_rank": 1,
        "carrier_mediation_recovery_fraction_range": (0.7955221554957826, 0.9345785212979353),
        "orthogonal_absolute_fraction_range": (0.13607285061027674, 0.22008478809153567),
        "stored_fit_scalars": 1152,
    })
    return manifest
