#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_native_capability pred_c_fresh_response_replay pred_d_fresh_causal_installation pred_e_fresh_causal_removal pred_f_controls_and_random_null
"""FP64 offline-corner correction for fresh head17.2 factor confirmation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import run_setting2_regional_attention17h2_factor_corner_fresh_v1 as v1


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
CORRECTION = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V2_CORRECTION.md"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V2_BINDING.json"
V1_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v2_result.json"
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_native_capability": None,
    "pred_c_fresh_response_replay": None,
    "pred_d_fresh_causal_installation": None,
    "pred_e_fresh_causal_removal": None,
    "pred_f_controls_and_random_null": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_correction_authority():
    binding = json.loads(BINDING.read_text())
    expected = {"v1_invalid_result": sha(V1_RESULT), "v1_runner": sha(Path(v1.__file__).resolve())}
    if binding.get("correction_authorities") != expected:
        raise ValueError("V1 correction authority changed")
    reference = json.loads(V1_RESULT.read_text())
    if reference["terminal"] != "invalid" or reference["instrument_errors"]["selected_expansion_relative_error"] <= 2e-6:
        raise ValueError("V1 is not the bound precision failure")
    if not all(report[gate] for report in reference["family_reports"].values() for gate in ("passes_capability", "passes_response", "passes_installation", "passes_removal", "passes_controls")):
        raise ValueError("V1 raw scientific gates did not all pass")
    return reference


original_head_write = v1.fold.head_write


def fp64_head_write(score1, score2, value, output_weight):
    return original_head_write(score1.double(), score2.double(), value.double(), output_weight.double())


original_atomic_create_json = v1.managed.atomic_create_json


def corrected_atomic_create_json(path, result):
    reference = check_correction_authority()
    scalar_keys = ("response_relative_l2", "selected_aggregate_installation_relative_l2", "selected_aggregate_removal_relative_l2")
    replay = {key: abs(result[key] - reference[key]) for key in scalar_keys}
    replay["response_cosine"] = abs(result["response_cosine"] - reference["response_cosine"])
    for family in reference["family_reports"]:
        replay[f"family_{family}_response"] = abs(result["family_reports"][family]["response_relative_l2"] - reference["family_reports"][family]["response_relative_l2"])
        replay[f"family_{family}_installation"] = abs(result["family_reports"][family]["installation"]["relative_l2"] - reference["family_reports"][family]["installation"]["relative_l2"])
        replay[f"family_{family}_removal"] = abs(result["family_reports"][family]["removal"]["relative_l2"] - reference["family_reports"][family]["removal"]["relative_l2"])
    if max(replay.values()) > .005:
        raise RuntimeError(f"V2 changed scientific outcomes: {replay}")
    result["schema"] = "setting2_regional_attention17h2_factor_corner_fresh_v2_result"
    result["correction"] = {
        "kind": "offline_factor_corner_fp64",
        "v1_invalid_result_sha256": sha(V1_RESULT),
        "v1_metric_absolute_differences": replay,
        "maximum_allowed_difference": .005,
    }
    result["scope"] += " V2 promotes only offline factor-corner contractions to FP64 and requires replay of V1 raw scientific metrics."
    original_atomic_create_json(path, result)


def configure():
    v1.RUNNER = RUNNER
    v1.PREREG = CORRECTION
    v1.BINDING = BINDING
    v1.OUT = OUT
    v1.fold.head_write = fp64_head_write
    v1.managed.atomic_create_json = corrected_atomic_create_json


def main():
    check_correction_authority()
    configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = v1.plan()
        planned["schema"] = "setting2_regional_attention17h2_factor_corner_fresh_v2_plan"
        planned["precision_correction"] = "offline factor corners and expansion in FP64; native-dtype suffix"
        print(json.dumps(planned, sort_keys=True))
        return
    v1.main()


if __name__ == "__main__":
    main()
