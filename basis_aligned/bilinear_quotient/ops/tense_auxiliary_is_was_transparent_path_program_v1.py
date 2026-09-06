#!/usr/bin/env python3
"""Prospectively validated read-compute-write program for is/was control."""

from __future__ import annotations

import torch

import aspectual_anchor_transparent_path_program_v11 as shared


PROGRAM_ID = "tense_auxiliary.is_vs_was.transparent_path_program_v1"
UPSTREAM_RESULT_SHA256 = "dad39b298a0e89e5e0271149574012ebf7f75e995b83f6ede12ebe3b1aa746e8"
BASIS_SHA256 = "e83ca8d0a89b170edcd334123bd6b25a8f18c39b1e441e4321f2fa96c29d5e1b"
TOKEN_IDS = {"is": 318, "was": 373}
GAIN_COEFFICIENTS = {
    "past_to_present": {"intercept": -147.6980274976786, "slope": -3678.9199345310312},
    "present_to_past": {"intercept": -348.2583777946352, "slope": 3040.696664893123},
}

ProgramInputError = shared.ProgramInputError
exact_selected_margin = shared.exact_selected_margin


def intermediate_unembedding_contrast(resid10, lm_head, *, direction: str):
    """Read the exact local current-minus-other is/was contrast at resid:10."""
    if direction not in GAIN_COEFFICIENTS:
        raise ProgramInputError(f"unknown direction: {direction!r}")
    current = TOKEN_IDS["is"] if direction == "present_to_past" else TOKEN_IDS["was"]
    other = TOKEN_IDS["was"] if direction == "present_to_past" else TOKEN_IDS["is"]
    return exact_selected_margin(resid10, lm_head, target_id=current, foil_id=other)


def predict_writer_gain(resid10, lm_head, *, direction: str):
    """Compute the frozen scalar gain for the selective q_is writer."""
    contrast = intermediate_unembedding_contrast(resid10, lm_head, direction=direction)
    coefficients = GAIN_COEFFICIENTS[direction]
    return coefficients["intercept"] + coefficients["slope"] * contrast


def upstream_writer_actuation(resid10, base_resid18, basis, lm_head, *, direction: str):
    """Execute the tested resid:10 read -> affine gain -> rank-one resid:18 write."""
    if not isinstance(resid10, torch.Tensor) or resid10.ndim != 1:
        raise ProgramInputError("resid10 must be a rank-one tensor")
    if not isinstance(base_resid18, torch.Tensor) or base_resid18.ndim != 1:
        raise ProgramInputError("base_resid18 must be a rank-one tensor")
    if resid10.shape != base_resid18.shape:
        raise ProgramInputError("resid10 and base_resid18 must have the same width")
    q = basis.reshape(-1) if isinstance(basis, torch.Tensor) else None
    if q is None or q.shape != base_resid18.shape:
        raise ProgramInputError("basis must flatten to the state shape")
    if abs(float(q.float().norm()) - 1.0) > 1.0e-4:
        raise ProgramInputError("basis must be unit norm")
    contrast = intermediate_unembedding_contrast(resid10, lm_head, direction=direction)
    coefficients = GAIN_COEFFICIENTS[direction]
    alpha = coefficients["intercept"] + coefficients["slope"] * contrast
    return {
        "patched_resid18": base_resid18 + alpha * q,
        "alpha": alpha,
        "resid10_unembedding_contrast": contrast,
        "direction": direction,
    }


def program_manifest() -> dict[str, object]:
    return {
        "program_id": PROGRAM_ID,
        "read_site": "resid:10",
        "read": "fixed current-minus-other is/was normalized soft-capped unembedding contrast",
        "compute": "direction-specific affine scalar gain",
        "write_site": "resid:18",
        "write": "alpha times the frozen selective rank-one q_is writer",
        "fixed_token_ids": dict(TOKEN_IDS),
        "coefficients": {key: dict(value) for key, value in GAIN_COEFFICIENTS.items()},
        "basis_sha256": BASIS_SHA256,
        "evidence_result_sha256": UPSTREAM_RESULT_SHA256,
        "required_runtime_inputs": ("resid10", "base_resid18", "rank1_basis", "lm_head", "direction"),
        "confirmation_resid18_margin_required": False,
        "donor_activation_required": False,
        "row_id_or_outcome_required": False,
        "stored_fit_scalars": 1156,
        "scope": "prospective disjoint lexicon within registered this/that-moment constructions",
    }
