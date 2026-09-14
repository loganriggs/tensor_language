#!/usr/bin/env python3
# BQLANE: cpu
# BQGATE: AUDIT pred_a_primary_receipt_and_evidence_identity pred_b_independent_audit_holds_scientific_null pred_c_exact_source_matched_factorial pred_d_no_selective_coarse_mlp_component pred_e_no_model_and_zero_new_forwards
"""Hash-bound reconciliation of the omitted R590 downstream-use authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/"basis_aligned/bilinear_quotient"
PATHS={
 "rows":BQ/"numbered_list_cached_value_downstream_use_rows_rung582.json",
 "preregistration":HERE/"NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG582_PREREGISTRATION.md",
 "runner":BQ/"ops/execute_numbered_list_cached_value_downstream_use_rung590.py",
 "result":BQ/"numbered_list_cached_value_downstream_use_rung590_results.json",
 "evidence":BQ/"numbered_list_cached_value_downstream_use_rung590_evidence/primitive_evidence.json",
 "receipt":BQ/"numbered_list_cached_value_downstream_use_rung590_receipt.json",
 "postexecution_audit":BQ/"numbered_list_cached_value_downstream_use_rung590_postexecution_audit.json",
}
OUT=HERE/"R590_DOWNSTREAM_USE_AUTHORITY_AUDIT_V1.json"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result=json.loads(PATHS["result"].read_text()); audit=json.loads(PATHS["postexecution_audit"].read_text())
    pred_a=(result["schema"]=="numbered_list_cached_value_downstream_use_rung590_result_v1" and
            audit["result_sha256"]==digest(PATHS["result"]) and audit["evidence_sha256"]==digest(PATHS["evidence"]) and
            audit["receipt_sha256"]==digest(PATHS["receipt"]))
    pred_b=(audit["audit_passed"] is True and audit["scientific_terminal_valid"] is True and audit["instrument_invalid"] is False and
            audit["independently_recomputed_decision"]=="downstream_use_decomposition_null" and not audit["audit_failures"])
    exact=audit["fit_exactness"]
    pred_c=(max(exact.values())<=1e-10 and result["evaluated_splits"]==["FIT"] and result["forbidden_splits_opened"]==[] and
            result["model_forwards"]==379 and result["model_backwards"]==0 and result["model_weights_updated"] is False)
    counts=audit["candidate_cell_pass_counts"]
    pred_d=(result["pred_b_selective_downstream_action_component"] is False and
            result["pred_c_cross_representation_reuse"] is False and result["selected_component"] is None and
            all(name.startswith(("mlp8_","mlp10_","mlp12_","mlp14_")) for name in counts) and len(counts)==12)
    predictions={"pred_a_primary_receipt_and_evidence_identity":bool(pred_a),
      "pred_b_independent_audit_holds_scientific_null":bool(pred_b),
      "pred_c_exact_source_matched_factorial":bool(pred_c),
      "pred_d_no_selective_coarse_mlp_component":bool(pred_d),
      "pred_e_no_model_and_zero_new_forwards":True}
    output={"schema":"r590_downstream_use_authority_audit_v1","terminal":"authority_reconciled" if all(predictions.values()) else "audit_failure",
      "predictions":predictions,"files":{name:{"path":str(path.relative_to(REPO)),"sha256":digest(path)} for name,path in PATHS.items()},
      "registered_science":{"decision":result["decision"],"candidate_sites":[8,10,12,14],
        "components":["background_cross","contrast_self","joint_response"],"selected_component":result["selected_component"],
        "fit_only":result["evaluated_splits"],"forbidden_splits_opened":result["forbidden_splits_opened"],
        "model_forwards":result["model_forwards"],"maximum_exactness_error":max(exact.values()),
        "independent_audit_verdict":audit["audit_verdict"],"postpublication_error_classification":audit["postpublication_error_classification"],
        "candidate_cell_pass_counts":counts},
      "authority_conclusion":("The shared exact H3+H7 numeric carrier has already been split by source-matched successor-versus-copy "
        "bilinear use at MLP8/10/12/14. No coarse C, Q, or joint response is a selective reusable action component. "
        "Do not repeat this response decomposition; a next consumer test must use a finer operational partition or another circuit."),
      "price":{"model_loaded":False,"new_model_forwards":0,"fits":0,"quantization":False}}
    if not all(predictions.values()): raise RuntimeError(json.dumps(output,indent=2))
    OUT.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"terminal":output["terminal"],"predictions":predictions,"maximum_exactness_error":max(exact.values())},indent=2))


if __name__=="__main__":main()
