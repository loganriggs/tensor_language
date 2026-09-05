#!/usr/bin/env python3
"""Frozen complete-module causal mediation plan for the L13H8 opener factors."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as rows_authority


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_downstream_module_mediation"
PATCH_LAYER, PATCH_HEAD = rows_authority.PATCH_LAYER, rows_authority.PATCH_HEAD
ROWS, ROWS_SHA256 = rows_authority.ROWS, rows_authority.ROWS_SHA256
FAMILIES = rows_authority.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = rows_authority.TARGET_FAMILIES, rows_authority.STABILITY_FAMILIES
MODULES = ("mlp13", "attention14", "mlp14", "attention15",
           "attention16", "mlp16", "attention17", "mlp17")
PRIOR_ART_SHA256 = "66fba6861aa6d5bd0048b7658fcf8440f9adde25f45e5e38d774fc83cfca55e4"


def compile_plan() -> dict:
    restores = [f"remove_{factor}_restore_{module}"
                for factor in ("mu", "delta") for module in MODULES]
    return {
        "schema": "bracket_l13h8_mu_delta_downstream_module_mediation_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "fixed_modules": list(MODULES),
        "conditions": ["native", "native_replay", "remove_mu", "remove_delta", *restores],
        "price": {"model_forwards": 20, "example_evaluations": 480,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "estimand": "For each factor/module, restore the complete native final-position module residual write after upstream removal; score centered-closer projection/cosine and signed correct-answer full-vocabulary CE damage/rescue.",
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "median_projection_recovery_each_factor_target_family_min": 0.10,
            "median_rescue_cosine_each_factor_target_family_min": 0.25,
            "positive_projection_fraction_each_factor_target_family_min": 2 / 3,
        },
        "frozen_predictions": {
            "pred_a": "native capability/replay hold and both factor removals are live in every construction and stability rewrite",
            "pred_b": "at least one fixed complete module meets projection, cosine, and sign bars for each factor across both target constructions; stability rewrites are descriptive",
            "pred_c": "with live upstream effects, no fixed complete module meets target-construction mediation bars for at least one factor, favoring distributed use or residual bypass",
        },
        "interpretation_limit": "intermediate causal localization only; no final basis, head response scan, selectivity proof, rank, PCA, or reconstruction",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
