#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_coefficient_bilinear_law_v1_result.json"
LAW = HERE / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
BINDING = HERE / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_BINDING.json"
OUT = HERE / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_AUDIT.json"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result, law, binding = (json.loads(p.read_text()) for p in (RESULT, LAW, BINDING))
    score = result["score"]
    checks = {
        "terminal_held": result["terminal"] == "coefficient_bilinear_law_held",
        "all_preregistered_predictions": all(score["predictions"].values()),
        "complete_512_effect_panel": len(score["evidence"]) == 512,
        "law_reproduces_rank1": score["law_vs_rank1"]["cosine"] >= .995 and score["law_vs_rank1"]["relative_l2_error"] <= .15 and score["law_vs_rank1"]["sign_agreement"] >= .95,
        "law_substitutes_native": score["law_vs_native"]["cosine"] >= .75 and score["law_vs_native"]["relative_l2_error"] <= .75 and score["law_vs_native"]["sign_agreement"] >= .75,
        "zero_closure_error": max(result["score"][k] for k in ("role_state_closure_max_absolute_error", "role_normalized_closure_max_absolute_error", "downstream_state_closure_max_absolute_error", "downstream_normalized_closure_max_absolute_error")) == 0.0,
        "four_scalar_formula": law["stored_coefficient_scalars_after"] == 4 and len(law["beta"]) == 4,
        "law_fit_used_no_causal_outcomes": not law["causal_outcomes_read"] and not binding["causal_outcomes_read_for_law_fit"],
    }
    payload = {"schema": "subject_number_coefficient_bilinear_law_v1_audit", "result_sha256": sha(RESULT), "law_sha256": sha(LAW), "binding_sha256": sha(BINDING), "checks": checks, "all_checks_pass": all(checks.values())}
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
