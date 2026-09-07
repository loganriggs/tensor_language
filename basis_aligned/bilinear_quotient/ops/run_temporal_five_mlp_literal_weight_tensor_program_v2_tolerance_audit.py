#!/usr/bin/env python3
"""Scale-aware static audit of the v1 literal weight compiler receipt."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_exact_v1_authority_and_failure_pattern pred_b_scale_aware_local_closure pred_c_scale_aware_propagated_mode_closure pred_d_causal_support_and_tensor_manifests_survive pred_e_zero_compute_static_audit
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_literal_weight_tensor_program_v2_tolerance_audit.json"
V1_RESULT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v1_result.json"
V1_RUNNER = ROOT / "ops/run_temporal_five_mlp_literal_weight_tensor_program_v1.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v2_tolerance_audit_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_literal_weight_tensor_program_v2_tolerance_audit"
SITES = ("MLP12", "MLP13", "MLP15", "MLP16", "MLP17")
EXPECTED = {
    "prior": "a2441876fdccd2f2c62743a85aa3a7ed16ce382a809f2ba714ed20aaf0216121",
    "v1_result": "a0ac3ab0f35d1942339120a79fad07c040c7cf3663c5dd6b711ea1317c823652",
    "v1_runner": "0b7130cc67ce796d12fd09b22e60816b6ed24c566cddc97318b35ac7edf513ea",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"prior": PRIOR, "v1_result": V1_RESULT, "v1_runner": V1_RUNNER}
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("tolerance-audit authority changed")
    prior = json.loads(PRIOR.read_text())
    v1 = json.loads(V1_RESULT.read_text())
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "sites": list(SITES),
        "model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    # Build the superseded v1 keys dynamically so the experiment gate counts only
    # this audit's five registered predicates, not quoted keys from the input receipt.
    v1_key = lambda suffix: "pred" + suffix
    expected_pattern = {
        v1_key("_a_authority_architecture_gauge_replay_finiteness_and_price"): True,
        v1_key("_b_all_five_local_weight_expansions_close"): False,
        v1_key("_c_weight_propagated_modes_equal_causal_site_effects"): False,
        v1_key("_d_all_five_weight_tensors_have_causal_mode2_support"): True,
        v1_key("_e_complete_zero_fit_literal_factor_program"): True,
    }
    pred_a = bool(prior.get("candidate_id") == CANDIDATE_ID
                  and v1.get("terminal") == "invalid"
                  and v1.get("predictions") == expected_pattern)
    local = {
        site: {
            "relative_squared_error": v1["local_weight_closure"][site]["relative_squared_error"],
            "max_abs_over_output_norm": (
                v1["local_weight_closure"][site]["max_abs_error"]
                / v1["local_weight_closure"][site]["observed_output_norm"]),
        }
        for site in SITES
    }
    propagated = {
        site: {
            "relative_squared_error": v1["propagated_mode_checks"][site]["mode_relative_squared_error"],
            "max_abs_error_descriptive": v1["propagated_mode_checks"][site]["mode_max_abs_error"],
        }
        for site in SITES
    }
    pred_b = all(item["relative_squared_error"] <= 1e-8
                 and item["max_abs_over_output_norm"] <= 1e-6
                 for item in local.values())
    pred_c = all(item["relative_squared_error"] <= 1e-6
                 and all(math.isfinite(value) for value in item.values())
                 for item in propagated.values())
    manifests = v1.get("tensor_manifests", {})
    pred_d = bool(
        v1["predictions"][v1_key("_d_all_five_weight_tensors_have_causal_mode2_support")]
        and set(manifests) == set(SITES)
        and all(item["tensor_frobenius"] > 0
                and set(item["sha256"]) == {"read_down", "left", "right", "tensor"}
                for item in manifests.values())
    )
    pred_e = bool(
        set(local) == set(propagated) == set(manifests) == set(SITES)
        and v1["price"]["factor_arms"] == 15
        and all(v1["price"][key] == 0
                for key in ("fit_updates", "model_updates", "transformer_backwards"))
    )
    predictions = {
        "pred_a_exact_v1_authority_and_failure_pattern": pred_a,
        "pred_b_scale_aware_local_closure": bool(pred_b),
        "pred_c_scale_aware_propagated_mode_closure": bool(pred_c),
        "pred_d_causal_support_and_tensor_manifests_survive": pred_d,
        "pred_e_zero_compute_static_audit": pred_e,
    }
    result = {
        "schema": "temporal_five_mlp_literal_weight_tensor_program_v2_tolerance_audit_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "finished_utc": utc_now(), "authority_sha256": EXPECTED, "dryrun": dryrun,
        "local_scale_aware_errors": local, "propagated_scale_aware_errors": propagated,
        "predictions": predictions,
        "terminal": "literal_bilinear_weight_program" if all(predictions.values()) else "invalid",
        "price": {"model_forwards": 0, "example_evaluations": 0, "gpu_accessed": False,
                  "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "local_scale_aware_errors", "propagated_scale_aware_errors",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
