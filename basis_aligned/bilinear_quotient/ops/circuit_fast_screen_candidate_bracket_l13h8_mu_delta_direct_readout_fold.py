#!/usr/bin/env python3
"""Frozen direct-readout decomposition of the causal L13H8 mu/delta residual path."""

from __future__ import annotations

import json

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_residual_write_bank_factorial as parent


CANDIDATE_ID = "bracket.pending_opener.l13h8_mu_delta_direct_readout_fold"
PATCH_LAYER, PATCH_HEAD = parent.PATCH_LAYER, parent.PATCH_HEAD
ROWS, ROWS_SHA256 = parent.ROWS, parent.ROWS_SHA256
FAMILIES = parent.FAMILIES
TARGET_FAMILIES, STABILITY_FAMILIES = parent.TARGET_FAMILIES, parent.STABILITY_FAMILIES
WRITE_BANK = parent.WRITE_BANK
PRIOR_ART_SHA256 = "194f9df04b0a264c6510ddae0763c589a827c95a9348591842cd1adfee2bb717"
WRITE_BANK_RESULT_SHA256 = "bc38fe115de9f5dac166bbe7b4451592015df67c700e5b297524a1f66a76b17d"


def compile_plan() -> dict:
    return {
        "schema": "bracket_l13h8_mu_delta_direct_readout_fold_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "rows_sha256": ROWS_SHA256,
        "rows": 24,
        "triplet_groups": 8,
        "write_bank": list(WRITE_BANK),
        "prior_results": {
            "residual_write_bank_factorial_sha256": WRITE_BANK_RESULT_SHA256,
            "direct_lens_method_warning": "BILIN18_CONNECTION.md section 2808",
        },
        "conditions": [
            "NN_native_factor_native_write_bank",
            "RN_remove_mu_install_native_write_bank",
            "RN_remove_delta_install_native_write_bank",
        ],
        "price": {"model_forwards": 3, "example_evaluations": 72,
                  "backwards": 0, "parameter_updates": 0},
        "opened_splits": ["FRESH_BASIC"],
        "closed_splits": ["TEST", "OOD"],
        "outcome_reads": [],
        "exact_computation": {
            "scale_fit": "solve [z_NN,-z_RN] [r_NN,r_RN]^T = f by least squares per row",
            "raw_difference": "W_U(z_NN-z_RN)",
            "direct_folded_factor": "W_U(f/r_NN)",
            "normalization_scale_correction": "W_U((r_RN/r_NN-1) z_RN)",
            "raw_identity": "raw_difference = direct_folded_factor + normalization_scale_correction",
            "softcap": "s(raw)=30*tanh(raw/30)",
            "softcap_correction": "s(raw_NN)-s(raw_RN)-(raw_NN-raw_RN)",
            "final_identity": "final_difference = direct_folded_factor + normalization_scale_correction + softcap_correction",
        },
        "bars": {
            "native_positive_fraction_each_family_min": 0.75,
            "median_live_centered_effect_norm_each_factor_family_min": 0.05,
            "rms_scale_fit_relative_residual_max": 2e-4,
            "raw_logit_identity_max_absolute_error": 2e-4,
            "softcap_output_replay_max_absolute_error": 2e-5,
            "final_logit_identity_max_absolute_error": 2e-4,
            "median_direct_projection_each_factor_target_family_min": 0.75,
            "median_absolute_normalization_or_softcap_projection_target_family_min": 0.25,
        },
        "frozen_predictions": {
            "pred_a": "native capability, live removals, the RMS-scale fit, and raw/final logit identities all hold",
            "pred_b": "direct folded-factor projection is at least 0.75 for both factors in both target constructions",
            "pred_c": "if pred_b fails, normalization-scale or softcap correction has absolute projection at least 0.25 in a target factor/construction",
        },
        "interpretation_limit": (
            "Exact weight-level readout of an already causal residual path; no learned direction, "
            "rank, PCA, activation reconstruction, semantic selectivity, TEST, or OOD claim"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
