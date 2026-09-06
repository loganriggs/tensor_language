#!/usr/bin/env python3
"""Extended-depth exact carrier component programs after v1 hit its ceiling."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument pred_b_both_greedy_paths_are_distributive pred_c_both_paths_generalize_to_a1 pred_d_both_paths_transfer_to_a2 pred_e_shared_program_machinery pred_f_price_and_coverage pred_g_truncation_was_active pred_h_extended_paths_improve_prior_recovery
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import run_aspectual_tense_carrier_component_greedy_program_v1 as inherited


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py"
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_carrier_component_greedy_program_v2.json"
OUT = ROOT / "circuits/followups/aspectual_tense_carrier_component_greedy_program_v2_result.json"
CANDIDATE_ID = "aspectual_tense.carrier_component_greedy_program_v2"
EXPECTED_BASE_RUNNER = "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413"
EXPECTED_PRIOR = "55d3f197a331b367e7c0f6b708115b4eeb430c58e268f778e0bdcfdb88a24f58"
BASELINES = {
    "has": {"heldout": 0.7362649359677497, "a2": 0.742626401152252},
    "is": {"heldout": 0.6542596410264659, "a2": 0.8047553527769007},
}
PREDICTION_CONTRACT = {
    "pred_a_authority_capability_exact_instrument": None,
    "pred_b_both_greedy_paths_are_distributive": None,
    "pred_c_both_paths_generalize_to_a1": None,
    "pred_d_both_paths_transfer_to_a2": None,
    "pred_e_shared_program_machinery": None,
    "pred_f_price_and_coverage": None,
    "pred_g_truncation_was_active": None,
    "pred_h_extended_paths_improve_prior_recovery": None,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    if sha(BASE_RUNNER) != EXPECTED_BASE_RUNNER or sha(PRIOR) != EXPECTED_PRIOR:
        raise inherited.ExperimentError("v2 prior or inherited implementation hash changed")
    inherited.PRIOR = PRIOR
    inherited.OUT = OUT
    inherited.CANDIDATE_ID = CANDIDATE_ID
    inherited.EXPECTED = dict(inherited.EXPECTED, prior=EXPECTED_PRIOR)
    inherited.MAX_STEPS = 10
    inherited.MAX_FORWARDS = 438
    inherited.MAX_EVALUATIONS = 5845
    inherited.MAX_RECORDS = 5663


def main():
    configure()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        inherited.main()
        return
    original_write = inherited.atomic_create_json

    def audited_write(path, result):
        result["schema"] = "aspectual_tense_carrier_component_greedy_program_result_v2"
        pred_g = all(len(result["selected_paths"][task]) >= 7 for task in ("has", "is"))
        pred_h = all(result["final_metrics"][task][panel]["mean_recovery"]
                     >= BASELINES[task][panel]
                     for task in ("has", "is") for panel in ("heldout", "a2"))
        result["predictions"]["pred_g_truncation_was_active"] = pred_g
        result["predictions"]["pred_h_extended_paths_improve_prior_recovery"] = pred_h
        invalid = not result["predictions"]["pred_a_authority_capability_exact_instrument"] \
            or not result["predictions"]["pred_f_price_and_coverage"]
        result["terminal"] = "invalid" if invalid else (
            "screen" if all(result["predictions"].values()) else "null")
        result["reason"] = (
            "extended_source_programs_improve_all_panels" if result["terminal"] == "screen"
            else "extended_source_programs_do_not_improve_all_panels" if result["terminal"] == "null"
            else "authority_capability_instrument_coverage_or_price_invalid"
        )
        original_write(path, result)

    inherited.atomic_create_json = audited_write
    inherited.main()


if __name__ == "__main__":
    main()
