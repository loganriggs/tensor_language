#!/usr/bin/env python3
"""Model-free scoped precision audit for the invalid v23 block-11 factorial."""

# BQGATE: AUDIT pred_a_hash_schema_inventory_and_price_exact pred_b_original_invalid_verdict_preserved pred_c_block_output_logits_and_behavior_close pred_d_final_hidden_failure_retained_and_excluded_only_from_successor_interface pred_e_downstream_reader_license_is_scoped_and_zero_forward
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT.parent / "polynomial_causal/TEMPORAL_ISWAS_V23_BLOCK11_FACTORIAL_PRECISION_AUDIT_V1.md"
RESULT = ROOT / "circuits/followups/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1_result.json"
OUT = ROOT / "circuits/audits/temporal_iswas_v23_block11_factorial_precision_audit_v1.json"
EXPECTED_AUTHORITY_SHA256 = "c65fcb0a109dd7d843d9678f87ae3bd36d0dcb3ea5aa2949da81d5e3f4d3d85d"
EXPECTED_RESULT_SHA256 = "2a7c5d7fabede998284c1c5ce14276e14a12310624896e3505384e8f1c8cf98e"
PREDICTION_KEYS = (
    "pred_a_hash_schema_inventory_and_price_exact",
    "pred_b_original_invalid_verdict_preserved",
    "pred_c_block_output_logits_and_behavior_close",
    "pred_d_final_hidden_failure_retained_and_excluded_only_from_successor_interface",
    "pred_e_downstream_reader_license_is_scoped_and_zero_forward",
)
ORIGINAL_KEYS = tuple("pred_" + suffix for suffix in (
    "a_authority_hooks_closures_self_patch_finiteness_and_exact_price",
    "b_residual_carrier_is_the_dominant_selective_branch",
    "c_m11_is_a_real_but_minor_branch",
    "d_exact_joint_rescue_closes_the_block11_mediation",
))
BARS = {"interface_restore": 1e-4, "joint_projection": .99}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(value)


def audit_payload(result):
    original_predictions = result.get("predictions", {})
    restore = result.get("joint_restore", {})
    price = result.get("price", {})
    A = bool(
        sha(AUTHORITY) == EXPECTED_AUTHORITY_SHA256
        and sha(RESULT) == EXPECTED_RESULT_SHA256
        and result.get("schema") == "temporal_iswas_v23_block11_residual_mlp_factorial_rescue_result_v1"
        and tuple(original_predictions) == ORIGINAL_KEYS
        and price.get("model_forwards_exact") == 7
        and price.get("sequence_evaluations_exact") == 448
        and finite(result)
    )
    B = bool(result.get("terminal") == "invalid_instrument"
             and original_predictions.get(ORIGINAL_KEYS[3]) is False)
    C = bool(
        original_predictions.get(ORIGINAL_KEYS[0]) is True
        and original_predictions.get(ORIGINAL_KEYS[1]) is True
        and result.get("component_closure_max_abs", math.inf) <= BARS["interface_restore"]
        and result.get("self_patch_max_abs", math.inf) <= BARS["interface_restore"]
        and restore.get("block11_output_max_abs", math.inf) <= BARS["interface_restore"]
        and restore.get("logits_max_abs", math.inf) <= BARS["interface_restore"]
        and result.get("reports", {}).get("joint", {}).get(
            "signed_projection", -math.inf) >= BARS["joint_projection"]
        and result.get("selected_component_or_threshold_after_outcome") is None
    )
    D = bool(restore.get("final_hidden_max_abs", -math.inf) > BARS["interface_restore"])
    E = bool(A and B and C and D)
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    return {
        "schema": "temporal_iswas_v23_block11_factorial_precision_audit_v1",
        "candidate_id": "audit.temporal_iswas.v23_block11_factorial_precision_v1",
        "authority_sha256": sha(AUTHORITY), "factorial_result_sha256": sha(RESULT),
        "original_terminal": result.get("terminal"),
        "original_predictions": original_predictions, "joint_restore": restore,
        "component_closure_max_abs": result.get("component_closure_max_abs"),
        "self_patch_max_abs": result.get("self_patch_max_abs"),
        "joint_signed_projection": result.get("reports", {}).get("joint", {}).get("signed_projection"),
        "predictions": predictions,
        "downstream_reader_license": bool(E),
        "license_scope": "block11_residual_correction_to_A12_M17_reader_atlas_only",
        "original_factorial_relabelled": False,
        "terminal": "licensed_downstream_reader_only" if E else "no_successor_license",
        "bars": BARS,
        "price": {"model_forwards_exact": 0, "sequence_evaluations_exact": 0,
                  "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0},
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    payload = audit_payload(json.loads(RESULT.read_text()))
    atomic_create_json(OUT, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
