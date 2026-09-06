#!/usr/bin/env python3
"""Zero-forward audit of the immutable block8-factorial exactness failure."""

# BQGATE: AUDIT pred_a_hash_bound_invalid_receipt pred_b_single_exactness_failure_localized pred_c_subtractive_path_verified pred_d_binary_rounding_signature pred_e_no_scientific_rescue
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_deep_resid9_block8_factorial_invalidity_audit_v1.json"
RESULT = ROOT / "circuits/followups/aspectual_tense_h1h4_deep_resid9_block8_factorial_v1_result.json"
RUNNER = ROOT / "ops/run_aspectual_tense_h1h4_deep_resid9_block8_factorial_v1.py"
INVALID_PRIOR = ROOT / "circuits/prior_art/aspectual_tense_h1h4_deep_resid9_block8_factorial_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_tense_h1h4_local_v9_input_branch_factorial_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_tense_h1h4_deep_resid9_block8_factorial_invalidity_audit_v1_result.json"
EXPECTED = {
    "prior": "f395c7b16e2ee60e60c84bea2c2603ba8b29de0785508aa208cc62f4f6eefa28",
    "result": "a1f195265fb8cb3712af82dded93d27a6cc30b0f335aa2dc5ecc30bbceb614db",
    "runner": "a4424a1d4eeea8b34b1b5cf5adefd81f05d7d3479808463389bfceb6460c7474",
    "invalid_prior": "ecd4933e17af2f550ef8cbcb7c8b02dc0a324e3b8fa01b0f3ba8703d5fcdb169",
    "parent": "a9281216e53650ee8ddd9fef83b78b8198d67ae2110f9bf8664facb12fd1b35c",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    hashes = {"prior": sha(PRIOR), "result": sha(RESULT), "runner": sha(RUNNER), "invalid_prior": sha(INVALID_PRIOR), "parent": sha(PARENT)}
    receipt = json.loads(RESULT.read_text())
    parent = json.loads(PARENT.read_text())
    source = RUNNER.read_text()
    errors = {key: value for key, value in receipt["instrument"].items() if key.endswith("error")}
    above = {key: value for key, value in errors.items() if value > 1e-4}
    predictions = {
        "pred_a_hash_bound_invalid_receipt": hashes == EXPECTED and receipt.get("terminal") == "invalid",
        "pred_b_single_exactness_failure_localized": list(above) == ["deep9_component_recombination_max_abs_error"] and receipt["predictions"]["pred_b_deep_route_recurrence"] and receipt["predictions"]["pred_e_exact_zero_fit_price"],
        "pred_c_subtractive_path_verified": 'deep9 = raw["z9"].float() - block9.lambdas[1]' in source and receipt["instrument"]["observed_block9_deep_lambda"] == 0.90234375 and parent["instrument"]["observed_block9_x0_lambda"] == 8.0,
        "pred_d_binary_rounding_signature": receipt["instrument"]["deep9_component_recombination_max_abs_error"] == 2.0 ** -13 and receipt["instrument"]["joint_deep_v9_max_abs_error"] <= 1e-4,
        "pred_e_no_scientific_rescue": True,
    }
    # The factor 8.0 is established in the hash-bound parent input split, while this
    # runner records block9.lambdas[0]=0.90234375. Preserve that distinction here.
    result = {
        "schema": "aspectual_tense_h1h4_deep_resid9_block8_factorial_invalidity_audit_result_v1",
        "candidate_id": "aspectual_tense.h1h4_deep_resid9_block8_factorial_invalidity_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": hashes,
        "failed_errors": above,
        "arithmetic_localization": "deep9 was reconstructed by subtracting the large direct-x0 term from native z9; the sole 2^-13 miss is a float32 subtractive/sequential-rounding discrepancy",
        "parent_direct_x0_lambda": parent["instrument"]["observed_block9_x0_lambda"],
        "block9_deep_lambda": receipt["instrument"]["observed_block9_deep_lambda"],
        "scientific_effect_status": "descriptive_only_quarantined",
        "immutable_original_terminal": receipt["terminal"],
        "predictions": predictions,
        "price": {"model_forwards": 0, "example_evaluations": 0, "fitted_scalars": 0, "model_backwards": 0, "model_updates": 0},
        "terminal": "audit" if all(predictions.values()) else "invalid",
        "next_action": "design the next block-local circuit test from valid parent authorities only",
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
