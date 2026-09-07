#!/usr/bin/env python3
"""Static causal-path audit of the complete attention-15 background clamp."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authorities_and_weight_scope_match pred_b_complete_attn15_overwrites_computed_head_outputs pred_c_das_fit_and_evaluation_always_enable_complete15 pred_d_l15h5_reader_is_not_causally_identified_by_these_runs
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_selected_projector_attn15_reader_path_audit_v1.json"
PARENT = ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py"
MULTI = ROOT / "ops/run_temporal_iswas_v15_multi_environment_rank1_das_v1.py"
FIT = ROOT / "ops/multi_construction_head_projector_fit.py"
WEIGHT = ROOT / "circuits/followups/temporal_iswas_v15_selected_head_projector_weight_interfaces_v1_result.json"
WEIGHT_AUDIT = ROOT / "circuits/followups/temporal_iswas_v15_selected_head_projector_weight_interfaces_v2_tolerance_audit_result.json"
FACTOR = ROOT / "circuits/followups/temporal_iswas_v15_head_factor_dual_greedy_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_selected_projector_attn15_reader_path_audit_v1_result.json"
CANDIDATE_ID = "temporal_iswas.selected_projector_attn15_reader_path_audit_v1"
EXPECTED = {
    "prior": "3c3ba982a569c2b3005dd76c61738ccd9fc4636407c39caa0a293cc2dc4b7e73",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "multi": "8c14405674b4773303e197be04668f29ca97d2662dbeccc0f9653c119716c556",
    "fit": "db0f87d1cd30b189303ac53500e44868bfeafbf483a47885b8498b5cbf250215",
    "weight": "374f02e32d4c6c0b1e28a27ca6c13d9bf2145f6022498333355e3b9455b6f6ba",
    "weight_audit": "31d8a52f17e8dcef447b47677b04eeff0b14268e1261ce8bae0c4ac06f9e676d",
    "factor": "51c705281cc4fa10aa7d5c3c54bc3b1ee1e9d6d3af3e74f3b98cef3a34ad0285",
}
PRICE = {"model_forwards": 0, "checkpoint_loads": 0, "example_evaluations": 0,
         "interventions": 0}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "multi": MULTI, "fit": FIT,
             "weight": WEIGHT, "weight_audit": WEIGHT_AUDIT, "factor": FACTOR}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"reader-path audit authority changed: {observed}")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    parent_source, multi_source, fit_source = PARENT.read_text(), MULTI.read_text(), FIT.read_text()
    weight, weight_audit, factor = (json.loads(path.read_text())
                                    for path in (WEIGHT, WEIGHT_AUDIT, FACTOR))
    marker = "if complete15 and layer == 15:"
    overwrite = "changed[index, :int(stop) + 1] = donor15[index, :int(stop) + 1]"
    head_write = "changed[index, :int(stop) + 1, head] = absolute"
    pred_a = bool(
        weight.get("scope") == "diagnostic_weight_geometry_not_causal_identification"
        and weight_audit.get("terminal") == "repaired_weight_interface_diagnostic"
        and all(weight_audit.get("predictions", {}).values())
        and all(item["source_top10_count"] == 4
                for fold in weight["shared_top10_interfaces"].values()
                for item in fold if item["label"].startswith("L15H5:")))
    pred_b = bool(marker in parent_source and overwrite in parent_source
                  and parent_source.index(head_write) < parent_source.index(marker)
                  and ", head]" not in overwrite)
    pred_c = bool(
        fit_source.count("complete15=True") >= 2
        and "return parent.run_crossfit" in multi_source
        and parent_source.count("complete15=True") >= 5)
    pred_d = bool(pred_b and pred_c)
    predictions = {
        "pred_a_authorities_and_weight_scope_match": pred_a,
        "pred_b_complete_attn15_overwrites_computed_head_outputs": pred_b,
        "pred_c_das_fit_and_evaluation_always_enable_complete15": pred_c,
        "pred_d_l15h5_reader_is_not_causally_identified_by_these_runs": pred_d,
    }
    attn15_only = factor["reports"]["0"]["report"]
    result = {
        "schema": "temporal_iswas_selected_projector_attn15_reader_path_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "static_path": {
            "upstream_to_l15_attention_head_output": "overwritten_by_fixed_donor_cache",
            "upstream_to_l15_residual_skip_and_mlp15": "remains_live",
            "upstream_to_layers16_17": "remains_live",
            "l15h5_weight_rank": "native_reader_hypothesis_only",
        },
        "attention15_only_reference": {
            panel: attn15_only["targets"][panel]["behavior"]["signed_projection"]
            for panel in ("A1", "A2")
        },
        "required_successor": "selected_projector_x_complete_attention15_2x2_dependency_factorial",
        "predictions": predictions,
        "terminal": "attn15_reader_path_bypassed" if all(predictions.values()) else "invalid",
        "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
