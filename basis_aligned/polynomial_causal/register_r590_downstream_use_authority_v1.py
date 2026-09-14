#!/usr/bin/env python3
"""Register the independently audited R590 coarse downstream-use null."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/"basis_aligned/bilinear_quotient"
sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2

CHECKPOINT="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT="r590_source_matched_downstream_use_fit"
ARTIFACTS={
 "r590_rows":("basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rows_rung582.json","rows"),
 "r590_prereg":("basis_aligned/polynomial_causal/NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG582_PREREGISTRATION.md","preregistration"),
 "r590_runner":("basis_aligned/bilinear_quotient/ops/execute_numbered_list_cached_value_downstream_use_rung590.py","implementation"),
 "r590_result":("basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung590_results.json","result"),
 "r590_evidence":("basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung590_evidence/primitive_evidence.json","evidence"),
 "r590_receipt":("basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung590_receipt.json","receipt"),
 "r590_postexecution_audit":("basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung590_postexecution_audit.json","audit"),
 "r590_authority_audit_source":("basis_aligned/polynomial_causal/audit_r590_downstream_use_authority_v1.py","audit_implementation"),
 "r590_authority_audit":("basis_aligned/polynomial_causal/R590_DOWNSTREAM_USE_AUTHORITY_AUDIT_V1.json","audit"),
}
DEST={
 "task.numbered_list.index_successor":("numbered_list_index_successor.v11","numbered_list_index_successor.v12",12,"numbered_list"),
 "task.numeric_sequence.continuation":("numeric_sequence_continuation.v9","numeric_sequence_continuation.v10",10,"numeric_sequence"),
}


def artifact(path,kind): return {"path":path,"sha256":file_sha256(REPO/path),"kind":kind,"status":"frozen"}
def metric(name,estimate,bar): return {"name":name,"estimate":estimate,"ci95":None,"bar":bar}
def make_event(record,event_id,claim_id,test_type,verdict,failure,result_id,metrics,notes):
    value={"event_id":event_id,"claim_id":claim_id,"test_type":test_type,"stage":"complete","verdict":verdict,
      "failure_kind":failure,"family_ids":[],"site_id":"final_label_l0_value_through_l8h3_h7","split_plan_id":SPLIT,
      "evaluation_role":"source_matched_downstream_use_fit_primary","metrics":metrics,"prereg_artifact_id":"r590_prereg",
      "result_artifact_id":result_id,"input_artifact_ids":["r590_rows","r590_evidence","r590_receipt"],"seed":None,
      "checkpoint_sha256":CHECKPOINT,"supersedes_event_id":None,"replicates_event_id":None,
      "sections":[ARTIFACTS[result_id][0]],"notes":notes}
    value["design_key"]=design_key(record,value); value["execution_key"]=execution_key(record,value); return value


def build(tag,result,audit):
    old,new,revision,namespace=DEST[tag]; path=circuit_path(tag); record=json.loads(path.read_text())
    if any(x["claim_id"]==new for x in record["claims"]): return record,False
    if record["claims"][-1]["claim_id"]!=old: raise RuntimeError(f"{tag} authority moved")
    for aid,(p,kind) in ARTIFACTS.items(): record["artifacts"][aid]=artifact(p,kind)
    record["split_plans"].append({"split_plan_id":SPLIT,"unit":"source-matched semantic group with successor/copy action, representation, source and surface cells",
      "partition_artifact_id":"r590_rows","builder_artifact_id":"r590_rows","seed":582,
      "groups":{"FIT":16,"SELECT":8,"FINAL_TEST":8,"OOD":8},
      "leakage_group_keys":["semantic group","source token","representation","action","surface rewrite"],
      "sealed_before_outcomes":True,"sealed_at":"2026-09-03T19:16:00Z"})
    ids=[f"{namespace}.r590.exact_downstream_factorial.held.v1",f"{namespace}.r590.coarse_mlp_action_component.null.v1",
         f"{namespace}.r590.cross_representation_reuse.null.v1",f"{namespace}.r590.independent_terminal_audit.held.v1"]
    claim=deepcopy(record["claims"][-1]); claim.update({"claim_id":new,"revision":revision,"supersedes":old,
      "evidence_event_ids":[*claim["evidence_event_ids"],*ids],"split_plan_ids":[*claim["split_plan_ids"],SPLIT]})
    claim["next_missing"]=("The shared exact H3+H7 numeric carrier and its coarse downstream bilinear use are now bounded. R590's "
      "source-matched successor/copy factorial exactly decomposed background-cross, contrast-self and joint responses at "
      "MLP8/10/12/14, but no component was a selective action factor and cross-representation reuse failed; an independent "
      "evidence audit holds the FIT-only scientific null. Do not repeat coarse complete-response or C/Q/joint MLP screens. "
      "Any continuation requires an independently defined finer consumer partition; otherwise move to a distinct circuit.")
    for site in claim["candidate_sites"]:
        if site["site_id"]=="final_label_l0_value_through_l8h3_h7": site["ceiling_event_ids"]=[*site["ceiling_event_ids"],ids[0],ids[1],ids[2],ids[3]]
    record["claims"].append(claim)
    exact=result["fit_exactness"]; counts=audit["candidate_cell_pass_counts"]
    events=[
      make_event(record,ids[0],new,"composition","held",None,"r590_result",
        [metric("maximum_exactness_error",max(exact.values()),"<=1e-10"),metric("model_forwards",result["model_forwards"],"379 FIT-only"),metric("forbidden_split_count",len(result["forbidden_splits_opened"]),"0")],
        "Exact source-matched cached-carrier response decomposition completed at MLP8/10/12/14 with all instrument gates live."),
      make_event(record,ids[1],new,"composition","null","scientific_null","r590_result",
        [metric("selected_component",result["selected_component"],"non-null required"),metric("candidate_count",len(counts),"12 frozen")],
        "None of twelve coarse background-cross, contrast-self or joint MLP responses separates successor action from matched copy."),
      make_event(record,ids[2],new,"cross_family_transfer","null","scientific_null","r590_result",
        [metric("cross_representation_reuse",result["pred_c_cross_representation_reuse"],"true required"),metric("select_opened",int("SELECT" in result["evaluated_splits"]),"1 only after FIT selection")],
        "No FIT component selected, so cross-representation reuse and SELECT correctly remained closed."),
      make_event(record,ids[3],new,"seed_stability","held",None,"r590_authority_audit",
        [metric("independent_audit_passed",audit["audit_passed"],"true"),metric("independently_recomputed_forwards",audit["independently_recomputed_model_forwards"],"379"),metric("audit_failure_count",len(audit["audit_failures"]),"0")],
        "Independent primitive-evidence audit reconstructs the valid scientific null; the managed wrapper failed only after atomic publication."),
    ]
    record["evidence_events"].extend(events); validate_v2(record); return record,True


def main():
    result=json.loads((BQ/"numbered_list_cached_value_downstream_use_rung590_results.json").read_text())
    audit=json.loads((BQ/"numbered_list_cached_value_downstream_use_rung590_postexecution_audit.json").read_text())
    authority=json.loads((HERE/"R590_DOWNSTREAM_USE_AUTHORITY_AUDIT_V1.json").read_text())
    if result["decision"]!="downstream_use_decomposition_null" or not audit["audit_passed"] or authority["terminal"]!="authority_reconciled": raise RuntimeError("authority changed")
    built={tag:build(tag,result,audit) for tag in DEST}
    with _lock("registry"):
        for tag,(record,changed) in built.items():
            if changed:_atomic_json(circuit_path(tag),record)
    rebuild_registry_v2()
    for tag in DEST:validate_v2(json.loads(circuit_path(tag).read_text()))
    print(json.dumps({tag:DEST[tag][1] for tag in DEST},indent=2))


if __name__=="__main__":main()
