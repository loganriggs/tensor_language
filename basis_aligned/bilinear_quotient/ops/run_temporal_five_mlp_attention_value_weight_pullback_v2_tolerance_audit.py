#!/usr/bin/env python3
"""Hash-bound BF16 tolerance audit of the frozen attention value pullback receipt."""
# BQGATE: EXPERIMENT pred_a_frozen_authority_and_invalid_provenance pred_b_bf16_aware_numerical_closure pred_c_compiled_program_retains_target_and_control_validity pred_d_current_value_branch_alone_is_sufficient pred_e_cached_value_branch_is_a_minor_correction
# BQLANE: cpu
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit.json"
V1 = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v1_result.json"
RUNNER = ROOT / "ops/run_temporal_five_mlp_attention_value_weight_pullback_v1.py"
INVALID = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v1_INVALID.md"
OUT = ROOT / "circuits/followups/temporal_five_mlp_attention_value_weight_pullback_v2_tolerance_audit_result.json"
EXPECTED = {"prior": "707168f85e3ee28197516a9694207108ca46254a285f80865eed9ffd1680337c",
            "v1": "aeede368b1c9c4dabf364e358780191da8074953af7b7c14f4a834bd22cc9a79",
            "runner": "f60d07e8a52b4bb3cb0228ecf41d044548f063f1266f53328c5443206eed7dfc",
            "invalid": "1ed872e1dd0792d69d1a49c3eccc36cacb1af2d1238464f28cefcf9ab23fe378"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {"prior": sha(PRIOR), "v1": sha(V1), "runner": sha(RUNNER), "invalid": sha(INVALID)}
    if observed != EXPECTED: raise RuntimeError(f"tolerance-audit authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_attention_value_weight_pullback_v2_tolerance_audit",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_max": 0, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    v1 = json.loads(V1.read_text()); summary = v1["summary"]
    pa = v1["terminal"] == "invalid" and not v1["predictions"]["pred_a_authority_hash_exact_value_coordinate_closure_finiteness_and_price"] and "formal terminal is invalid" in INVALID.read_text()
    pb = summary["max_value_coordinate_closure_rse"] <= 2e-9 and summary["max_captured_vs_compiled_execution_rse"] <= 1e-10
    pc = v1["predictions"]["pred_c_fully_weight_compiled_program_is_sufficient_and_selective"]
    current = {}; cached = {}; finite = []
    for label, report in v1["reports"].items():
        cur, old = report["target"]["current_only"], report["target"]["cached_only"]
        both = report["target"]["compiled_both"]
        current[label] = {"worst_target_residual": cur["worst_target_residual"], "behavior_signed_projection": cur["behavior_signed_projection"]}
        gains = {task: both["behavior_signed_projection"][task] - cur["behavior_signed_projection"][task] for task in ("temporal", "iswas")}
        cached[label] = {"combined_minus_current_projection": gains, "cached_only_worst_target_residual": old["worst_target_residual"],
                         "cached_only_behavior_signed_projection": old["behavior_signed_projection"]}
        finite += [cur["worst_target_residual"], old["worst_target_residual"], *cur["behavior_signed_projection"].values(), *old["behavior_signed_projection"].values(), *gains.values()]
    pd = all(item["worst_target_residual"] <= .15 and min(item["behavior_signed_projection"].values()) >= .8 for item in current.values())
    pe = all(max(item["combined_minus_current_projection"].values()) <= .02 and (item["cached_only_worst_target_residual"] > .15 or min(item["cached_only_behavior_signed_projection"].values()) < .8) for item in cached.values())
    pa = pa and all(math.isfinite(float(value)) for value in finite)
    predictions = {"pred_a_frozen_authority_and_invalid_provenance": bool(pa),
                   "pred_b_bf16_aware_numerical_closure": bool(pb),
                   "pred_c_compiled_program_retains_target_and_control_validity": bool(pc),
                   "pred_d_current_value_branch_alone_is_sufficient": bool(pd),
                   "pred_e_cached_value_branch_is_a_minor_correction": bool(pe)}
    terminal = "attention_current_value_weight_program" if all(predictions.values()) else "tolerance_audit_failed"
    result = {"schema": "temporal_five_mlp_attention_value_weight_pullback_tolerance_audit_result_v2",
              "started_utc": now(), "finished_utc": now(), "authority_sha256": EXPECTED,
              "audited_numerics": {"local_coordinate_rse": summary["max_value_coordinate_closure_rse"],
                                     "complete_execution_rse": summary["max_captured_vs_compiled_execution_rse"],
                                     "local_rse_tolerance": 2e-9, "execution_rse_tolerance": 1e-10},
              "current_only": current, "cached_branch": cached, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 0, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("audited_numerics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
