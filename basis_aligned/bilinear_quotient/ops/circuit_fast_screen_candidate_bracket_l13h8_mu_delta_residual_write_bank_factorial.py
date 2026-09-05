#!/usr/bin/env python3
"""Frozen residual-route versus downstream-write-bank 2x2 factorial."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as rows_authority


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_residual_write_bank_factorial"
PATCH_LAYER, PATCH_HEAD = rows_authority.PATCH_LAYER, rows_authority.PATCH_HEAD
ROWS, ROWS_SHA256 = rows_authority.ROWS, rows_authority.ROWS_SHA256
FAMILIES = rows_authority.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = rows_authority.TARGET_FAMILIES, rows_authority.STABILITY_FAMILIES
WRITE_BANK = ("mlp13", "attention14", "mlp14", "attention15", "mlp15",
              "attention16", "mlp16", "attention17", "mlp17")
PRIOR_ART_SHA256 = "eab7b8ef3c54e7583204451cc3659a1c7470c60ac1e6502e508b7269510fae47"


def compile_plan() -> dict:
    return {
        "schema": "bracket_l13h8_mu_delta_residual_write_bank_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_sha256": ROWS_SHA256, "rows": 24, "triplet_groups": 8,
        "write_bank": list(WRITE_BANK),
        "conditions": ["native_model", "native_replay_capture_bank",
                       "remove_mu_capture_bank", "remove_delta_capture_bank",
                       "remove_mu_install_native_bank", "native_mu_install_removed_bank",
                       "remove_delta_install_native_bank", "native_delta_install_removed_bank"],
        "price": {"model_forwards": 8, "example_evaluations": 192,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"], "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "vector_factorial": {
            "total": "C_NN-C_RR", "residual_path": "C_NN-C_RN",
            "write_bank": "C_NN-C_NR", "interaction": "C_NN-C_RN-C_NR+C_RR",
            "identity": "residual_path + write_bank - interaction = total",
        },
        "ce_factorial": {
            "total": "L_RR-L_NN", "residual_path": "L_RN-L_NN",
            "write_bank": "L_NR-L_NN", "loss_interaction": "L_RR-L_RN-L_NR+L_NN",
            "identity": "total = residual_path + write_bank + loss_interaction",
        },
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "native_replay_max_absolute_logit_error_max": 1e-5,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "median_residual_projection_each_factor_target_family_min": 0.75,
            "median_write_projection_or_absolute_interaction_projection_each_factor_target_family_min": 0.25,
            "factorial_vector_identity_max_absolute_error": 1e-5,
            "factorial_ce_identity_max_absolute_error": 1e-6,
        },
        "frozen_predictions": {
            "pred_a": "native capability/replay, live removals, and both exact factorial identities hold",
            "pred_b": "residual-route projection is at least 0.75 for both factors in both target constructions",
            "pred_c": "write-bank projection or absolute interaction projection reaches 0.25 for a target factor/construction",
        },
        "interpretation_limit": "grouped path factorial only; no site, head, rank, PCA, reconstruction, or selectivity claim",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
