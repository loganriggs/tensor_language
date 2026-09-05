#!/usr/bin/env python3
"""Frozen plan for exact mu/delta mediation by the three R549 attention candidates."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as rows_authority


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_r549_attention_mediation"
PATCH_LAYER, PATCH_HEAD = rows_authority.PATCH_LAYER, rows_authority.PATCH_HEAD
ROWS, ROWS_SHA256 = rows_authority.ROWS, rows_authority.ROWS_SHA256
FAMILIES = rows_authority.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = rows_authority.TARGET_FAMILIES, rows_authority.STABILITY_FAMILIES
HEADS = ((14, 1), (15, 3), (16, 1))
PRIOR_ART_SHA256 = "176815d395903cef47f13148d6a2d1fb7190d661e075c3851c263dff32466530"


def compile_plan() -> dict:
    restores = [f"remove_{factor}_restore_l{layer}h{head}"
                for factor in ("mu", "delta") for layer, head in HEADS]
    return {
        "schema": "bracket_l13h8_mu_delta_r549_attention_mediation_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_authority_candidate_id": rows_authority.CANDIDATE_ID,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "fixed_heads": [f"L{layer}H{head}" for layer, head in HEADS],
        "conditions": ["native", "native_replay", "remove_mu", "remove_delta", *restores],
        "price": {"model_forwards": 10, "example_evaluations": 240,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "estimand": "For each factor/head, d=C(native)-C(remove), r=C(restore)-C(remove), projection=dot(r,d)/(dot(d,d)+1e-8), cosine=dot(r,d)/(norm(r)norm(d)+1e-8); CE damage/rescue use the signed correct-answer full-vocabulary CE.",
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "median_projection_recovery_each_factor_family_min": 0.10,
            "median_rescue_cosine_each_factor_family_min": 0.25,
            "positive_projection_fraction_each_factor_family_min": 2 / 3,
        },
        "frozen_predictions": {
            "pred_a": "native capability/replay hold and both upstream removals are live in every construction and stability rewrite",
            "pred_b": "at least one fixed R549 head meets all projection, cosine, and sign bars for each factor separately across both target constructions; matching stability-rewrite reports are descriptive robustness evidence",
            "pred_c": "with live upstream effects, no fixed R549 head meets the target-construction mediation bars for at least one factor, favoring distributed use or bypass over these response-localized heads",
        },
        "interpretation_limit": "causal localization only; no site selection, selectivity proof, or representational decomposition",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
