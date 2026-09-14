#!/usr/bin/env python3
"""Advance numeric sequence authority with fresh +1 and exact factor evidence."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]; BQ=REPO/"basis_aligned/bilinear_quotient"
sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2

TAG="task.numeric_sequence.continuation"; OLD="numeric_sequence_continuation.v8"; NEW="numeric_sequence_continuation.v9"
CHECKPOINT="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT="numeric_sequence_nonadjacent_ood_v1"
ARTIFACTS={
 "numeric_plus1_rows_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_NONADJACENT_OOD_V1_ROWS.json","rows"),
 "numeric_plus1_builder_v1":("basis_aligned/polynomial_causal/build_numeric_sequence_nonadjacent_ood_v1_rows.py","builder"),
 "numeric_plus1_capability_prereg_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_PREREGISTRATION.md","preregistration"),
 "numeric_plus1_capability_binding_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_BINDING.json","binding"),
 "numeric_plus1_capability_prior_v1":("basis_aligned/bilinear_quotient/circuits/prior_art/numeric_sequence_nonadjacent_ood_capability_v1.json","prior_art"),
 "numeric_plus1_capability_runner_v1":("basis_aligned/bilinear_quotient/ops/run_numeric_sequence_nonadjacent_ood_capability_v1.py","implementation"),
 "numeric_plus1_capability_result_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_RESULT.json","result"),
 "numeric_router_prereg_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_PREREGISTRATION.md","preregistration"),
 "numeric_router_binding_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_BINDING.json","binding"),
 "numeric_router_prior_v1":("basis_aligned/bilinear_quotient/circuits/prior_art/numeric_sequence_payload_router_interaction_v1.json","prior_art"),
 "numeric_router_runner_v1":("basis_aligned/bilinear_quotient/ops/run_numeric_sequence_payload_router_interaction_v1.py","implementation"),
 "numeric_router_result_v1":("basis_aligned/polynomial_causal/NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_RESULT.json","result"),
}


def artifact(path,kind): return {"path":path,"sha256":file_sha256(REPO/path),"kind":kind,"status":"frozen"}
def metric(name,estimate,bar): return {"name":name,"estimate":estimate,"ci95":None,"bar":bar}
def event(record,event_id,test_type,verdict,failure,families,result_id,prereg_id,metrics,notes):
    value={"event_id":event_id,"claim_id":NEW,"test_type":test_type,"stage":"complete","verdict":verdict,
      "failure_kind":failure,"family_ids":families,"site_id":"final_label_l0_value_through_l8h3_h7",
      "split_plan_id":SPLIT,"evaluation_role":"fresh_nonadjacent_plus1_primary","metrics":metrics,
      "prereg_artifact_id":prereg_id,"result_artifact_id":result_id,
      "input_artifact_ids":["numeric_plus1_rows_v1","numeric_plus1_capability_result_v1","spaced_capability_result_v1"],
      "seed":None,"checkpoint_sha256":CHECKPOINT,"supersedes_event_id":None,"replicates_event_id":None,
      "sections":[ARTIFACTS[result_id][0]],"notes":notes}
    value["design_key"]=design_key(record,value); value["execution_key"]=execution_key(record,value); return value


def add_family(claim,family):
    if not any(x["family_id"]==family["family_id"] for x in claim["counterfactual_families"]): claim["counterfactual_families"].append(family)


def main():
    capability=json.loads((HERE/"NUMERIC_SEQUENCE_NONADJACENT_OOD_CAPABILITY_V1_RESULT.json").read_text())
    result=json.loads((HERE/"NUMERIC_SEQUENCE_PAYLOAD_ROUTER_INTERACTION_V1_RESULT.json").read_text())
    if capability["terminal"]!="capability_pass" or result["terminal"]!="shared_final_source_factor_without_relation_selectivity": raise RuntimeError("terminal changed")
    path=circuit_path(TAG); record=json.loads(path.read_text())
    if any(x["claim_id"]==NEW for x in record["claims"]):
        validate_v2(record); rebuild_registry_v2(); print(json.dumps({"status":"already_registered","claim_id":NEW})); return
    if record["claims"][-1]["claim_id"]!=OLD: raise RuntimeError("numeric authority moved")
    for aid,(p,kind) in ARTIFACTS.items(): record["artifacts"][aid]=artifact(p,kind)
    record["split_plans"].append({"split_plan_id":SPLIT,"unit":"fresh paired nonadjacent +1 sequence group crossed with representation and construction",
      "partition_artifact_id":"numeric_plus1_rows_v1","builder_artifact_id":"numeric_plus1_builder_v1","seed":None,
      "groups":{"fresh_series":8,"fresh_instruction":8},"leakage_group_keys":["paired texts","lexical cue","construction","representation","value pair"],
      "sealed_before_outcomes":True,"sealed_at":"2026-09-14T21:54:00Z"})
    claim=deepcopy(record["claims"][-1]); ids=["numeric_sequence.nonadjacent_plus1_capability.held.v1",
      "numeric_sequence.cached_payload_plus1_transfer.held.v1","numeric_sequence.score_router_transfer.null.v1",
      "numeric_sequence.score_cached_interaction.null.v1","numeric_sequence.final_source_factor_selectivity.null.v1"]
    claim.update({"claim_id":NEW,"revision":9,"supersedes":OLD,"status":"weights_translated",
      "evidence_event_ids":[*claim["evidence_event_ids"],*ids],"split_plan_ids":[*claim["split_plan_ids"],SPLIT],
      "next_missing":("The exact L8H3+H7 final-source cached-value term is a shared numeric payload: cached-only and joint replacement "
        "transfer +1 answers across every fresh digit/word construction and direction, while score-only and a local score-by-cached "
        "interaction fail. No arm is selective against capable copy rows. The numeric value carrier is weight-computed, but the native "
        "relation selector/context router remains external. Next compare complete downstream responses to an identical cached-payload "
        "change under +1 versus copy contexts. Do not repeat factor, gain, adjacent-pair, threshold, rank, or quantization screens.")})
    claim["causal_variable"].update({"read":"the final numeric value through an exact shared L8H3+H7 cached-value payload; relation context is external",
      "operation":"install the paired cached-value term to move the candidate numeric state, then let the native context-dependent suffix choose +1 or copy"})
    add_family(claim,{"family_id":"numeric_plus1_digit_nonadjacent_ood","role":"interchange","changes":["digit +1 state and answer"],
      "holds_fixed":["digit representation","three-value +1 relation","prompt construction"],"builder_artifact_id":"numeric_plus1_builder_v1",
      "control_ids":["number-word +1","digit copy"],"split_plan_id":SPLIT,"status":"validated"})
    add_family(claim,{"family_id":"numeric_plus1_word_nonadjacent_ood","role":"interchange","changes":["number-word +1 state and answer"],
      "holds_fixed":["number-word representation","three-value +1 relation","prompt construction"],"builder_artifact_id":"numeric_plus1_builder_v1",
      "control_ids":["digit +1","number-word copy"],"split_plan_id":SPLIT,"status":"validated"})
    for site in claim["candidate_sites"]:
        if site["site_id"]=="final_label_l0_value_through_l8h3_h7": site["ceiling_event_ids"]=[*site["ceiling_event_ids"],ids[1],ids[2],ids[3],ids[4]]
    record["claims"].append(claim)
    target=[v for k,v in result["cell_reports"].items() if k.startswith("target|") and k.endswith("|cached")]
    score=[v for k,v in result["cell_reports"].items() if k.startswith("target|") and k.endswith("|score")]
    controls=[v for k,v in result["cell_reports"].items() if k.startswith("control|")]
    events=[
      event(record,ids[0],"capability","held",None,["numeric_plus1_digit_nonadjacent_ood","numeric_plus1_word_nonadjacent_ood"],"numeric_plus1_capability_result_v1","numeric_plus1_capability_prereg_v1",
        [metric("minimum_native_accuracy",min(x["accuracy"] for x in capability["cell_reports"].values()),">=0.85 each"),metric("minimum_mean_partner_margin",min(x["mean_partner_margin"] for x in capability["cell_reports"].values()),">0 each")],"All four fresh nonadjacent +1 cells are capable before intervention."),
      event(record,ids[1],"cross_family_transfer","held",None,["numeric_plus1_digit_nonadjacent_ood","numeric_plus1_word_nonadjacent_ood"],"numeric_router_result_v1","numeric_router_prereg_v1",
        [metric("minimum_cached_donorward_fraction",min(x["donorward_fraction"] for x in target),">=0.75"),metric("minimum_cached_donor_answer_win_fraction",min(x["donor_answer_win_fraction"] for x in target),">=0.50"),metric("cached_mean_effect_range",[min(x["mean_donor_margin_effect"] for x in target),max(x["mean_donor_margin_effect"] for x in target)],">0 each")],"Cached-only exact replacement transfers +1 for every construction, representation and direction."),
      event(record,ids[2],"composition","null","scientific_null",["numeric_plus1_digit_nonadjacent_ood","numeric_plus1_word_nonadjacent_ood"],"numeric_router_result_v1","numeric_router_prereg_v1",
        [metric("score_mean_effect_range",[min(x["mean_donor_margin_effect"] for x in score),max(x["mean_donor_margin_effect"] for x in score)],"target transfer bars")],"Final-source score substitution alone does not transfer +1 state."),
      event(record,ids[3],"composition","null","scientific_null",["numeric_plus1_digit_nonadjacent_ood","numeric_plus1_word_nonadjacent_ood"],"numeric_router_result_v1","numeric_router_prereg_v1",
        [metric("joint_over_best_single_range",result["joint_over_best_single_range"],">=1.10 every cell")],"Joint score-by-cached substitution does not consistently improve on cached alone."),
      event(record,ids[4],"null_control","null","scientific_null",[],"numeric_router_result_v1","numeric_router_prereg_v1",
        [metric("selective_arm_count",len(result["selective_arms"]),">=1"),metric("minimum_copy_answer_preservation",min(x.get("recipient_answer_preservation",1) for x in controls),">=0.85")],"No target-passing final-source arm is selective against all capable digit/word copy cells."),
    ]
    record["evidence_events"].extend(events); validate_v2(record)
    with _lock("registry"):_atomic_json(path,record)
    rebuild_registry_v2(); validate_v2(json.loads(path.read_text()))
    print(json.dumps({"status":"registered","claim_id":NEW,"events":len(events)},indent=2))


if __name__=="__main__":main()
