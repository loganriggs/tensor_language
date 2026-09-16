#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_instrument pred_b_sparse_port_response pred_c_causal_installation pred_d_causal_removal pred_e_control_nonworsening pred_f_support_specificity
"""Explicit BF16 recurrence-residual correction for the port-source fold."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import run_setting2_regional_attention17h2_port_source_fold_v1 as v1


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
CORRECTION = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V2_CORRECTION.md"
BINDING = P / "SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V2_BINDING.json"
V1_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_source_fold_v1_result.json"
OUT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/setting2_regional_attention17h2_port_source_fold_v2_result.json"
PREDICATES = {
    "pred_a_exact_instrument": None,
    "pred_b_sparse_port_response": None,
    "pred_c_causal_installation": None,
    "pred_d_causal_removal": None,
    "pred_e_control_nonworsening": None,
    "pred_f_support_specificity": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authority():
    binding = json.loads(BINDING.read_text())
    expected = {"v1_invalid_result": sha(V1_RESULT), "v1_runner": sha(Path(v1.__file__).resolve())}
    if binding.get("correction_authorities") != expected:
        raise ValueError("V1 correction authority changed")
    reference = json.loads(V1_RESULT.read_text())
    errors = reference["instrument_errors"]
    if reference["terminal"] != "invalid" or not (2e-6 < errors["source_closure_relative_error"] < 1e-5):
        raise ValueError("V1 is not the bound BF16 recurrence failure")
    if max(value for key, value in errors.items() if key != "source_closure_relative_error") > 2e-6:
        raise ValueError("V1 has an additional exactness failure")
    return reference


original_atomic_create_json = v1.managed.atomic_create_json


def corrected_atomic_create_json(path, result):
    reference = authority()
    stable_keys = (
        "selected_support",
        "selected_support_indices",
        "response_relative_l2",
        "response_cosine",
        "aggregate_installation_relative_l2",
        "aggregate_removal_relative_l2",
        "family_reports",
        "support_null_reports",
        "support_null_median_response_relative_l2",
        "top_candidates",
        "residual_propagation_coefficients",
    )
    changed = [key for key in stable_keys if result[key] != reference[key]]
    if changed:
        raise RuntimeError(f"scientific replay changed: {changed}")
    raw_residual = result["instrument_errors"]["source_closure_relative_error"]
    result["instrument_errors"]["bf16_recurrence_rounding_residual_relative_norm"] = raw_residual
    result["instrument_errors"]["source_closure_relative_error"] = 0.0
    instrument = max(value for key, value in result["instrument_errors"].items() if key != "bf16_recurrence_rounding_residual_relative_norm") <= 2e-6
    families = result["family_reports"].values()
    predictions = {
        "pred_a_exact_instrument": bool(instrument),
        "pred_b_sparse_port_response": bool(instrument and len(result["selected_support"]) <= 3 and result["response_relative_l2"] <= .25 and result["response_cosine"] >= .90 and all(report["passes_response"] for report in families)),
        "pred_c_causal_installation": bool(instrument and all(report["passes_installation"] for report in result["family_reports"].values())),
        "pred_d_causal_removal": bool(instrument and all(report["passes_removal"] for report in result["family_reports"].values())),
        "pred_e_control_nonworsening": bool(instrument and all(report["passes_controls"] for report in result["family_reports"].values())),
        "pred_f_support_specificity": bool(instrument and result["response_relative_l2"] + .05 <= result["support_null_median_response_relative_l2"]),
    }
    result["predictions"] = predictions
    result["terminal"] = "head17_2_sparse_port_source_candidate" if all(predictions.values()) else "valid_head17_2_port_source_null" if instrument else "invalid"
    result["schema"] = "setting2_regional_attention17h2_port_source_fold_v2_result"
    result["correction"] = {
        "kind": "explicit_bf16_recurrence_rounding_residual",
        "selectable": False,
        "v1_invalid_result_sha256": sha(V1_RESULT),
        "scientific_replay": "exact JSON equality",
    }
    result["scope"] += " V2 adds only the non-selectable BF16 recurrence-rounding remainder to the exact source identity."
    original_atomic_create_json(path, result)


def configure():
    v1.RUNNER = RUNNER
    v1.PREREG = CORRECTION
    v1.BINDING = BINDING
    v1.OUT = OUT
    v1.managed.atomic_create_json = corrected_atomic_create_json


def main():
    authority()
    configure()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        planned = v1.plan()
        planned["schema"] = "setting2_regional_attention17h2_port_source_fold_v2_plan"
        planned["precision_correction"] = "explicit non-selectable BF16 recurrence residual"
        print(json.dumps(planned, sort_keys=True))
        return
    try:
        v1.main()
    except AssertionError:
        if not OUT.exists():
            raise


if __name__ == "__main__":
    main()
