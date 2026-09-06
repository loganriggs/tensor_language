#!/usr/bin/env python3
"""Prospective upstream read-compute-write extension of aspectual program v11."""

from __future__ import annotations

import torch

import aspectual_anchor_transparent_path_program_v11 as v11


PROGRAM_ID = "aspectual_anchor.has_vs_had.transparent_path_program_v12"
UPSTREAM_RESULT_SHA256 = "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c"
TOKEN_IDS = {"has": 468, "had": 550}
GAIN_COEFFICIENTS = {
    "past_to_present": {"intercept": 3575.0380871196844, "slope": 1181.9271936643886},
    "present_to_past": {"intercept": 3049.77917349804, "slope": -1881.1152323328579},
}

ProgramInputError = v11.ProgramInputError
carrier_amplitude = v11.carrier_amplitude
rank1_carrier_projection = v11.rank1_carrier_projection
compiled_sparse_suffix_delta = v11.compiled_sparse_suffix_delta
exact_final_logits = v11.exact_final_logits
exact_scored_pair = v11.exact_scored_pair
exact_selected_margin = v11.exact_selected_margin
donor_free_margin_reflection = v11.donor_free_margin_reflection
operational_quotient_manifest = v11.operational_quotient_manifest


def intermediate_unembedding_contrast(resid10, lm_head, *, direction: str):
    """Read the fixed current-minus-other has/had contrast locally at resid:10.

    This applies the checkpoint's normalized soft-capped unembedding functional to
    an intermediate state. It is not a replay of the full post-block17 logits.
    """
    if direction not in GAIN_COEFFICIENTS:
        raise ProgramInputError(f"unknown direction: {direction!r}")
    current = TOKEN_IDS["has"] if direction == "present_to_past" else TOKEN_IDS["had"]
    other = TOKEN_IDS["had"] if direction == "present_to_past" else TOKEN_IDS["has"]
    return v11.exact_selected_margin(resid10, lm_head, target_id=current, foil_id=other)


def predict_carrier_gain(resid10, lm_head, *, direction: str):
    """Predict the frozen resid:18 carrier gain from the upstream local read."""
    contrast = intermediate_unembedding_contrast(resid10, lm_head, direction=direction)
    coefficients = GAIN_COEFFICIENTS[direction]
    return coefficients["intercept"] + coefficients["slope"] * contrast


def upstream_carrier_actuation(resid10, base_resid18, basis, lm_head, *, direction: str):
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
    """Return v11 plus the prospectively validated upstream actuator interface."""
    manifest = dict(v11.program_manifest())
    manifest.update({
        "program_id": PROGRAM_ID,
        "upstream_predictive_actuator": {
            "read_site": "resid:10",
            "read": "fixed current-minus-other has/had normalized soft-capped unembedding contrast",
            "compute": "direction-specific affine scalar gain",
            "write_site": "resid:18",
            "write": "alpha times the frozen rank-one carrier",
            "fixed_token_ids": dict(TOKEN_IDS),
            "coefficients": {key: dict(value) for key, value in GAIN_COEFFICIENTS.items()},
            "basis_sha256": manifest["rank1_basis_sha256"],
            "evidence_result_sha256": UPSTREAM_RESULT_SHA256,
            "required_runtime_inputs": ("resid10", "base_resid18", "rank1_basis", "lm_head", "direction"),
            "confirmation_resid18_margin_required": False,
            "donor_activation_required": False,
            "row_outcome_ids_required": False,
        },
        "upstream_predictive_confirmation": {
            "scope": "prospective third lexicon with population capability within tested constructions",
            "A1_recovery": 0.8538454604376456,
            "A2_recovery": 0.8579021545739657,
            "P_margin_reflection": 1.0026248626680097,
            "C_normalized_unrelated_effect": 0.0021410661481911153,
            "direction_fraction": 1.0,
        },
        "preferred_actuator_within_tested_scope": "upstream_carrier_actuation",
        "stored_fit_scalars": 1157,
    })
    return manifest
