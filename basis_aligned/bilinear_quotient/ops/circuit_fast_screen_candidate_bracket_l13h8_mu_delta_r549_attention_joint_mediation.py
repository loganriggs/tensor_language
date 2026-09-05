#!/usr/bin/env python3
"""Frozen joint-restoration follow-up for the three response-localized R549 heads."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_r549_attention_mediation as parent


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_r549_attention_joint_mediation"
PATCH_LAYER, PATCH_HEAD = parent.PATCH_LAYER, parent.PATCH_HEAD
ROWS, ROWS_SHA256 = parent.ROWS, parent.ROWS_SHA256
FAMILIES, TARGET_FAMILIES = parent.FAMILIES, parent.TARGET_FAMILIES
STABILITY_FAMILIES, HEADS = parent.STABILITY_FAMILIES, parent.HEADS
PRIOR_ART_SHA256 = "c08ffec32d1aaba60cf081dc145c1645a187e5cf5f2a22e7ea2c6eee580759e1"
INDIVIDUAL_RESULT_SHA256 = "7ae8f092a873b15d3f273196db380e46ef1e7f906b7d837395eb7b3dacf87917"


def compile_plan() -> dict:
    return {
        "schema": "bracket_l13h8_mu_delta_r549_attention_joint_mediation_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": PRIOR_ART_SHA256,
        "individual_result_sha256": INDIVIDUAL_RESULT_SHA256,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "fixed_joint_heads": [f"L{layer}H{head}" for layer, head in HEADS],
        "conditions": ["native", "native_replay", "remove_mu", "remove_delta",
                       "remove_mu_restore_joint_l14h1_l15h3_l16h1",
                       "remove_delta_restore_joint_l14h1_l15h3_l16h1"],
        "price": {"model_forwards": 6, "example_evaluations": 144,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "bound_prior_result": "individual mediation result 7ae8f092...; read only after current outputs for the descriptive interaction diagnostic",
        "estimand": "Joint centered-closer projection/cosine and full-vocabulary correct-answer CE damage/rescue; interaction diagnostics subtract the row-level sum of the three frozen individual projection recoveries and CE rescues from joint restoration.",
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "median_joint_projection_recovery_each_factor_target_family_min": 0.10,
            "median_joint_rescue_cosine_each_factor_target_family_min": 0.25,
            "positive_joint_projection_fraction_each_factor_target_family_min": 2 / 3,
        },
        "frozen_predictions": {
            "pred_a": "native capability/replay hold and both upstream factor removals remain live",
            "pred_b": "the all-three restoration meets the projection, cosine, and sign bars for both mu and delta in both target constructions despite negligible frozen individual rescues",
            "pred_c": "with live upstream effects, joint restoration stays below a mediation bar for at least one factor in a target construction, favoring distributed use or bypass",
        },
        "interpretation_limit": "one fixed joint corner only; no subset factorial, site scan, selectivity proof, or representational decomposition",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
