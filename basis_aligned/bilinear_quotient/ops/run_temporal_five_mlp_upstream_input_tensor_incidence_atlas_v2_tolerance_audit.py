#!/usr/bin/env python3
"""Scale-aware static audit of the complete upstream tensor-incidence atlas."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_exact_authority_and_preserved_failure_pattern pred_b_all_775_records_close_at_target_scale pred_c_causal_ranking_signal_survives_unchanged pred_d_failed_distinct_writer_prediction_resolves_to_shared_writer pred_e_static_zero_compute_complete_audit
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v2_tolerance_audit.json"
V1 = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1_result.json"
V1_RUNNER = ROOT / "ops/run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.py"
V1_PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.json"
LITERAL = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v1_result.json"
LITERAL_AUDIT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v2_tolerance_audit_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v2_tolerance_audit_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_upstream_input_tensor_incidence_atlas_v2_tolerance_audit"
TARGETS = ("MLP12", "MLP13", "MLP15", "MLP16", "MLP17")
EXPECTED = {
    "prior": "9a6bd1c0d3f9483cc928afaa9afaf61e69860bed5ed5244fa2c7760193ff623a",
    "v1": "0cc9909dcab7a17b93820300da56a07f4cd9a2610f71a1de1c7008710d064467",
    "v1_runner": "6e28d38ec1446eafb3518c1bfe603a5e3469ceadb6f80266e2c695a274692366",
    "v1_prior": "ba41e39135762ad4bdd90bac4fca42f28ca17d657f3db7e9a4ef8e9c460e1e52",
    "literal": "a0ac3ab0f35d1942339120a79fad07c040c7cf3663c5dd6b711ea1317c823652",
    "literal_audit": "ae86583c3461b151de1c6b25eb586be4e32bb0c9c8670a45710c38693700c10d",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"prior": PRIOR, "v1": V1, "v1_runner": V1_RUNNER,
             "v1_prior": V1_PRIOR, "literal": LITERAL, "literal_audit": LITERAL_AUDIT}
    if {key: sha(path) for key, path in paths.items()} != EXPECTED:
        raise RuntimeError("upstream tensor scale-audit authority changed")
    prior, v1, literal, literal_audit = [json.loads(path.read_text())
                                         for path in (PRIOR, V1, LITERAL, LITERAL_AUDIT)]
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "sites": 179,
        "compatible_records": 775, "targets": list(TARGETS),
        "model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    old_key = lambda suffix: "pred" + suffix
    expected_pattern = {
        old_key("_a_authority_alignment_self_patch_finiteness_and_price"): True,
        old_key("_b_all_compatible_local_tensor_expansions_close"): False,
        old_key("_c_known_writer_frontier_is_enriched"): True,
        old_key("_d_tensor_incidence_predicts_final_causal_modes"): True,
        old_key("_e_factor_and_target_structure_is_nontrivial"): False,
    }
    pred_a = bool(
        prior.get("candidate_id") == CANDIDATE_ID
        and v1.get("terminal") == "invalid"
        and v1.get("predictions") == expected_pattern
        and literal_audit.get("terminal") == "literal_bilinear_weight_program")
    reference = {target: literal["local_weight_closure"][target]["observed_output_norm"]
                 for target in TARGETS}
    records = []
    for site, site_metrics in v1["site_metrics"].items():
        for target, target_metrics in site_metrics["targets"].items():
            closure = target_metrics["local_closure"]
            scale = reference[target]
            records.append({
                "site": site, "target": target,
                "max_abs_over_target_reference": closure["max_abs_error"] / scale,
                "error_energy_over_target_reference": (
                    closure["relative_squared_error"]
                    * closure["observed_output_norm"] ** 2 / scale ** 2),
            })
    finite = all(math.isfinite(value) for record in records for key, value in record.items()
                 if key not in ("site", "target"))
    pred_b = bool(
        len(records) == 775 and finite
        and max(record["max_abs_over_target_reference"] for record in records) <= 1e-6
        and max(record["error_energy_over_target_reference"] for record in records) <= 1e-8)
    summary = v1["summary"]
    pred_c = bool(
        summary["tensor_vs_causal_spearman"] >= 0.90
        and summary["top_vs_bottom_quintile_causal_ratio"] >= 100.0
        and len(summary["known_writers_in_tensor_top20"]) >= 3
        and all(summary["known_writer_positive_cells"][site]
                for site in summary["known_writers_in_tensor_top20"]))
    pred_d = bool(
        set(summary["material_targets"]) == set(TARGETS)
        and all(v1["target_rankings"][target][0] == "MLP1" for target in TARGETS)
        and summary["top20_max_interaction_fraction"] >= 0.10)
    pred_e = bool(
        len(v1["site_metrics"]) == 179 and len(records) == 775
        and v1["price"]["model_forwards"] == 182
        and all(v1["price"][key] == 0
                for key in ("fit_updates", "model_updates", "transformer_backwards")))
    predictions = {
        "pred_a_exact_authority_and_preserved_failure_pattern": pred_a,
        "pred_b_all_775_records_close_at_target_scale": pred_b,
        "pred_c_causal_ranking_signal_survives_unchanged": pred_c,
        "pred_d_failed_distinct_writer_prediction_resolves_to_shared_writer": pred_d,
        "pred_e_static_zero_compute_complete_audit": pred_e,
    }
    result = {
        "schema": "temporal_five_mlp_upstream_input_tensor_incidence_atlas_v2_tolerance_audit_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "finished_utc": utc_now(), "authority_sha256": EXPECTED, "dryrun": dryrun,
        "scale_aware_closure": {
            "records": len(records),
            "max_abs_over_target_reference": max(
                record["max_abs_over_target_reference"] for record in records),
            "max_error_energy_over_target_reference": max(
                record["error_energy_over_target_reference"] for record in records),
            "worst_absolute_records": sorted(
                records, key=lambda record: record["max_abs_over_target_reference"], reverse=True)[:10],
            "worst_energy_records": sorted(
                records, key=lambda record: record["error_energy_over_target_reference"], reverse=True)[:10],
        },
        "scientific_summary": summary, "predictions": predictions,
        "terminal": "shared_upstream_writer_screen" if all(predictions.values()) else (
            "invalid" if not pred_a or not pred_b or not pred_e else "causal_incidence_only"),
        "price": {"model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0, "gpu_accessed": False},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "scale_aware_closure", "scientific_summary",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
