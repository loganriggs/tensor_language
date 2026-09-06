#!/usr/bin/env python3
"""Scored-logit readout extension of executable aspectual program v6."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import aspectual_anchor_transparent_path_program_v6 as v6


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v7"
SOFTCAP = 30.0

ProgramInputError = v6.ProgramInputError
MLP_FACTORS = v6.MLP_FACTORS
SUFFIX_BOUNDARIES = v6.SUFFIX_BOUNDARIES
SUFFIX_SOURCE_BOUNDARIES = v6.SUFFIX_SOURCE_BOUNDARIES
SUFFIX_MLP_FACTORS_BY_BOUNDARY = v6.SUFFIX_MLP_FACTORS_BY_BOUNDARY
compiled_sparse_suffix_delta = v6.compiled_sparse_suffix_delta
selected_suffix_mlp_write_delta = v6.selected_suffix_mlp_write_delta
bilinear_hidden_delta = v6.bilinear_hidden_delta


def exact_final_logits(resid18, lm_head):
    """Apply the checkpoint's exact RMS-normalization, unembedding, and soft cap."""
    if not isinstance(resid18, torch.Tensor) or resid18.ndim < 1:
        raise ProgramInputError("resid18 must be a torch tensor with a feature dimension")
    if not callable(lm_head):
        raise ProgramInputError("lm_head must be callable")
    normalized = F.rms_norm(resid18, (resid18.shape[-1],))
    return SOFTCAP * torch.tanh(lm_head(normalized) / SOFTCAP)


def exact_scored_pair(resid18, lm_head, *, answer_id: int, foil_id: int):
    """Return the answer and foil logits from the exact checkpoint readout."""
    logits = exact_final_logits(resid18, lm_head)
    vocabulary = logits.shape[-1]
    if (
        not isinstance(answer_id, int) or not isinstance(foil_id, int)
        or answer_id == foil_id or not 0 <= answer_id < vocabulary or not 0 <= foil_id < vocabulary
    ):
        raise ProgramInputError("answer_id and foil_id must be distinct in-range integers")
    return logits[..., answer_id], logits[..., foil_id]


def compiled_sparse_suffix_scored_pair(
    base_resid18,
    initial_delta,
    lm_head,
    *,
    answer_id: int,
    foil_id: int,
    lambda0_by_boundary: dict[int, object],
    source_attention_delta_by_boundary: dict[int, object],
    mlp_states_by_boundary: dict[int, tuple[object, object, object, object]],
    down_weight_by_boundary: dict[int, object],
):
    """Execute the sparse suffix recurrence and exact scored-logit readout."""
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
    return exact_scored_pair(base_resid18 + delta18, lm_head, answer_id=answer_id, foil_id=foil_id)


def program_manifest() -> dict[str, object]:
    """Return the exact stable inventory exposed by executable program v7."""
    manifest = dict(v6.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "exact_final_readout": ("rms_norm", "lm_head", "softcap_30", "answer_foil_index"),
        "runtime_dependencies": (
            "checkpoint weights, lambda0 scalars, and lm_head",
            "paired base/donor MLP4 states",
            "paired base/hybrid attention source terms",
            "paired base/hybrid bilinear MLP states at blocks11,12,14,15",
            "base resid18 state",
        ),
    })
    return manifest
