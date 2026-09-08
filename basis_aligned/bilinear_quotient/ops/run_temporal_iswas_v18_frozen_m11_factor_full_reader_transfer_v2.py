#!/usr/bin/env python3
"""Precision-correct replay of the frozen full-reader M11 factor transfer."""

# BQGATE: EXPERIMENT pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price pred_b_frozen_top32_factors_explain_full_M11_on_both_new_constructions pred_c_frozen_top128_factors_explain_most_full_M11_on_both_new_constructions pred_d_frozen_factor_program_remains_selective_on_P_and_C pred_e_no_U8_refit_reorder_or_v18_selection
import hashlib
import json
from pathlib import Path

import run_temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1 as experiment


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "ops/run_temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1.py"
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1.json"
V1_RESULT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v2_result.json"
EXPECTED = {
    "base_runner": "107ce8b51294e0a74b07016352dc55968369b6dc8612423979229dee49105253",
    "prior": "ad803dae9bd31e8ff0495e935aab515cec1173606fc52e2efc87072f998e600f",
    "v1_result": "d0764a416da08395fbec24361104f46883061ff7b4639e1f01aaf8102a57867e",
}
PREDICTION_KEYS = (
    "pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price",
    "pred_b_frozen_top32_factors_explain_full_M11_on_both_new_constructions",
    "pred_c_frozen_top128_factors_explain_most_full_M11_on_both_new_constructions",
    "pred_d_frozen_factor_program_remains_selective_on_P_and_C",
    "pred_e_no_U8_refit_reorder_or_v18_selection",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    observed = {name: sha(path) for name, path in {
        "base_runner": BASE_RUNNER, "prior": PRIOR, "v1_result": V1_RESULT}.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v2 replay authority changed: {observed}")
    invalid = json.loads(V1_RESULT.read_text())
    if (invalid.get("terminal") != "invalid_instrument"
            or invalid.get("predictions", {}).get(
                "pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price") is not False
            or not all(record["all_hidden_margin_max_abs_error"] <= experiment.BARS["formula"]
                       for record in invalid["instrument"].values())
            or not any(record["all_hidden_output_max_abs_error"] > experiment.BARS["formula"]
                       for record in invalid["instrument"].values())):
        raise RuntimeError("v1 invalid-instrument diagnosis changed")
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = "cross_task.temporal_iswas.v18_frozen_m11_factor_full_reader_transfer_v2"
    experiment.RESULT_SCHEMA = "temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_result_v2"
    experiment.main()


if __name__ == "__main__":
    main()
