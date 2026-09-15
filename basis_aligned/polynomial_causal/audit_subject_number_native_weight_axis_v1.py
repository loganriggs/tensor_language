#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_native_weight_axis_v1_result.json"
AXIS = HERE / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
BINDING = HERE / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_BINDING.json"
OUT = HERE / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_AUDIT.json"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r, a, b = (json.loads(p.read_text()) for p in (RESULT, AXIS, BINDING)); s = r["score"]
    checks = {
        "terminal_held": r["terminal"] == "native_weight_axis_held",
        "all_predictions": all(s["predictions"].values()),
        "complete_panel": len(s["evidence"]) == 512,
        "reproduces_law_axis": s["native_axis_vs_law_axis"]["cosine"] >= .90 and s["native_axis_vs_law_axis"]["relative_l2_error"] <= .50 and s["native_axis_vs_law_axis"]["sign_agreement"] >= .90,
        "substitutes_native": s["native_axis_vs_native"]["cosine"] >= .65 and s["native_axis_vs_native"]["relative_l2_error"] <= .90 and s["native_axis_vs_native"]["sign_agreement"] >= .70,
        "zero_closure": max(s[k] for k in ("role_state_closure_max_absolute_error", "role_normalized_closure_max_absolute_error", "downstream_state_closure_max_absolute_error", "downstream_normalized_closure_max_absolute_error")) == 0.0,
        "native_weight_only_axis": a["construction"] == "top_left_singular_vector_of_native_head_output_projection" and not a["causal_outcomes_read"] and not a["behavior_fit"],
        "bound_outcome_blind": not b["causal_outcomes_read_for_axis"],
    }
    payload = {"schema": "subject_number_native_weight_axis_v1_audit", "result_sha256": sha(RESULT), "axis_sha256": sha(AXIS), "binding_sha256": sha(BINDING), "checks": checks, "all_checks_pass": all(checks.values())}
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); print(json.dumps(payload, indent=2))
if __name__ == "__main__": main()
