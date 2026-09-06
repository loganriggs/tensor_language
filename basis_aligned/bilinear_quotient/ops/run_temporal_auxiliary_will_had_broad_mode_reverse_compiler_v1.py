#!/usr/bin/env python3
"""Run the shared response-compiler protocol in the broad-to-narrow cue direction."""

# BQGATE: EXPERIMENT pred_a_capability_and_writer_closure pred_b_reverse_coefficient_prediction pred_c_reverse_program_material pred_d_prediction_beats_intercept pred_e_fresh_p_selective pred_f_price_exact
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh
import circuit_fast_screen_candidate_temporal_auxiliary as original
import run_temporal_auxiliary_will_had_summed_writer_compiler_ood_v1 as executor

ROOT = Path(__file__).resolve().parents[1]
executor.PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v1.json"
executor.PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_cross_cue_reader_modes_v1_result.json"
executor.FRESH_BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
executor.OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
executor.OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v1_result.json"
executor.CANDIDATE_ID = "temporal_auxiliary.will_vs_had.broad_mode_reverse_compiler_v1"
executor.RESULT_SCHEMA = "temporal_auxiliary_broad_mode_reverse_compiler_result_v1"
executor.EXPECTED_PARENT_TERMINAL = "shared_mode"
executor.PREDICTION_NAMES = (
    "pred_a_capability_and_writer_closure",
    "pred_b_reverse_coefficient_prediction",
    "pred_c_reverse_program_material",
    "pred_d_prediction_beats_intercept",
    "pred_e_fresh_p_selective",
    "pred_f_price_exact",
)
executor.TERMINAL_NAMES = {"success": "shared_compiler", "transfer_fail": "coefficient_asymmetry",
                           "behavioral_fail": "response_asymmetry"}
executor.fresh_builder = original
executor.ood_builder = fresh
executor.EXPECTED = {
    "prior": "1d1a2f4300c20aabedf30ae6509af42223e1b30bce11e6761336d8d00e7c3866",
    "parent_result": "4c91680557c7c2cdcef67f69586fdc758c5605369a0be7b62477574caf1e4f42",
    "fresh_builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "ood_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "parent_runner": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
}


if __name__ == "__main__":
    executor.main()
