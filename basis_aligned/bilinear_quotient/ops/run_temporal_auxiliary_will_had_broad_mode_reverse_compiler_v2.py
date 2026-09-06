#!/usr/bin/env python3
"""Run broad-to-narrow compiler on prospectively manifested capable rows."""

# BQGATE: EXPERIMENT pred_a_capability_and_writer_closure pred_b_reverse_coefficient_prediction pred_c_reverse_program_material pred_d_prediction_beats_intercept pred_e_fresh_p_selective pred_f_price_exact
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh
import circuit_fast_screen_candidate_temporal_auxiliary as original
import run_temporal_auxiliary_will_had_summed_writer_compiler_ood_v1 as executor

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
manifest = json.loads(MANIFEST.read_text())
executor.PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v2.json"
executor.PARENT_RESULT = MANIFEST
executor.FRESH_BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
executor.OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
executor.OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v2_result.json"
executor.CANDIDATE_ID = "temporal_auxiliary.will_vs_had.broad_mode_reverse_compiler_v2"
executor.RESULT_SCHEMA = "temporal_auxiliary_broad_mode_reverse_compiler_result_v2"
executor.EXPECTED_PARENT_TERMINAL = "manifest"
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
executor.OOD_ALLOWED_ROW_IDS = manifest["jointly_capable_row_ids"]
executor.OOD_FAMILY_COUNTS = {"A1": 29, "A2": 30, "P": 32}
executor.MODEL_FORWARDS = 38
executor.EXAMPLE_EVALUATIONS = 1049
executor.RECORDS = 295
executor.fresh_builder = original
executor.ood_builder = fresh
executor.EXPECTED = {
    "prior": "5c0391194a7633e88a1bf50651d3360552b1386e6a9723f49e088b2b2441a6a9",
    "parent_result": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
    "fresh_builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "ood_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "parent_runner": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
}


if __name__ == "__main__":
    executor.main()
