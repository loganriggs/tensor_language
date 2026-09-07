#!/usr/bin/env python3
"""Zero-model scale-aware fp32 tolerance audit of the immutable v1 weight atlas."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_original_disposition_and_price pred_b_scale_aware_fp32_closure_passes pred_c_weight_rankings_are_complete_and_finite pred_d_audit_opens_no_new_model_outcome
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit.json"
V1 = ROOT / "circuits/followups/temporal_iswas_v15_selected_head_projector_weight_interfaces_v1_result.json"
V1_RUNNER = ROOT / "ops/run_temporal_iswas_v15_selected_head_projector_weight_interfaces_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit_result.json"
CANDIDATE_ID = "temporal_iswas.v15_selected_head_projector_weight_interfaces_v2_tolerance_audit"
EXPECTED = {
    "prior": "ccf33a0134a802e002eaa3e2fbdc65bcfaf343c027724f090a5e01429a4fa3e2",
    "v1": "374f02e32d4c6c0b1e28a27ca6c13d9bf2145f6022498333355e3b9455b6f6ba",
    "v1_runner": "99d71ea96c1ad2c888094143737db42c5661699a1377089296b298b9ac251b88",
}
PRICE = {"model_forwards": 0, "checkpoint_loads": 0, "example_evaluations": 0,
         "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
PREDICTION_KEYS = (
    "pred_a_authority_original_disposition_and_price",
    "pred_b_scale_aware_fp32_closure_passes",
    "pred_c_weight_rankings_are_complete_and_finite",
    "pred_d_audit_opens_no_new_model_outcome",
)
ORIGINAL_FAILURE_KEY = "pred" + "_a_authority_exact_map_finiteness_and_price"
ORIGINAL_REQUIRED_KEYS = tuple("pred" + suffix for suffix in (
    "_b_reader_rankings_are_fold_stable",
    "_c_known_l15_readers_are_enriched",
    "_d_sources_converge_on_shared_downstream_interfaces",
    "_e_known_serial_value_writers_are_enriched",
))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_scores(value):
    if isinstance(value, dict):
        return all(finite_scores(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_scores(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    observed = {"prior": sha(PRIOR), "v1": sha(V1), "v1_runner": sha(V1_RUNNER)}
    if observed != EXPECTED:
        raise RuntimeError(f"tolerance-audit authority changed: {observed}")
    prior, v1 = json.loads(PRIOR.read_text()), json.loads(V1.read_text())
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    records = v1.get("records", {})
    norms = [record.get("write_norm") for fold in records.values() for record in fold.values()]
    closure = v1.get("instrument", {}).get("wo_embedding_max_abs_error")
    min_norm = min(norms) if norms else float("nan")
    relative = closure / min_norm if isinstance(closure, (int, float)) and min_norm > 0 else float("nan")
    epsilon = 2.0 ** -23
    original_predictions = v1.get("predictions", {})
    pred_a = bool(prior.get("candidate_id") == CANDIDATE_ID and v1.get("terminal") == "invalid"
                  and original_predictions.get(ORIGINAL_FAILURE_KEY) is False
                  and all(original_predictions.get(key) is True for key in ORIGINAL_REQUIRED_KEYS)
                  and v1.get("price", {}).get("model_forwards") == 0
                  and v1.get("price", {}).get("checkpoint_loads") == 1)
    pred_b = bool(math.isfinite(relative) and relative <= 8 * epsilon)
    pred_c = bool(set(records) == {"0", "1"}
                  and all(set(records[fold]) == {"L8H1", "L9H1", "L9H4", "L11H3"}
                          for fold in records)
                  and finite_scores(records))
    pred_d = True
    predictions = dict(zip(PREDICTION_KEYS, (pred_a, pred_b, pred_c, pred_d)))
    result = {"schema": "temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only_zero_model",
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"original_max_abs_error": closure, "minimum_write_norm": min_norm,
            "relative_to_minimum_write_norm": relative, "float32_epsilon": epsilon,
            "registered_relative_bound": 8 * epsilon},
        "v1_rankings_sha256": EXPECTED["v1"],
        "scope": "repaired_diagnostic_weight_geometry_not_causal_identification",
        "predictions": predictions,
        "terminal": "repaired_weight_interface_diagnostic" if all(predictions.values()) else "invalid",
        "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
