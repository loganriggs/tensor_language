#!/usr/bin/env python3
"""Register the capable spaced authority and shared H3+H7 cached payload."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
BQ=REPO/"basis_aligned/bilinear_quotient"; sys.path.insert(0,str(BQ))
from circuit_registry_v2 import _atomic_json,_lock,circuit_path,design_key,execution_key,file_sha256,rebuild_registry_v2,validate_v2

CHECKPOINT="680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
SPLIT="numeric_successor_spaced_ood_v1"
CAPABILITY=HERE/"NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_RESULT.json"
TRANSFER=HERE/"NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_RESULT.json"
ARTIFACTS={
 "spaced_rows_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_OOD_V1_ROWS.json","rows"),
 "spaced_rows_builder_v1":("basis_aligned/polynomial_causal/build_numeric_successor_spaced_ood_v1_rows.py","builder"),
 "spaced_capability_prereg_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_PREREGISTRATION.md","preregistration"),
 "spaced_capability_binding_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_BINDING.json","binding"),
 "spaced_capability_prior_v1":("basis_aligned/bilinear_quotient/circuits/prior_art/numeric_successor_spaced_ood_capability_v1.json","prior_art"),
 "spaced_capability_runner_v1":("basis_aligned/bilinear_quotient/ops/run_numeric_successor_spaced_ood_capability_v1.py","implementation"),
 "spaced_capability_result_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_OOD_CAPABILITY_V1_RESULT.json","result"),
 "spaced_transfer_prereg_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_PREREGISTRATION.md","preregistration"),
 "spaced_transfer_binding_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_BINDING.json","binding"),
 "spaced_transfer_prior_v1":("basis_aligned/bilinear_quotient/circuits/prior_art/numeric_successor_spaced_cached_transfer_v1.json","prior_art"),
 "spaced_transfer_runner_v1":("basis_aligned/bilinear_quotient/ops/run_numeric_successor_spaced_cached_transfer_v1.py","implementation"),
 "spaced_transfer_result_v1":("basis_aligned/polynomial_causal/NUMERIC_SUCCESSOR_SPACED_CACHED_TRANSFER_V1_RESULT.json","result"),
}
DESTINATIONS={
 "task.numbered_list.index_successor":("numbered_list_index_successor.v10","numbered_list_index_successor.v11",11,"weights_translated","numbered_list"),
 "task.numeric_sequence.continuation":("numeric_sequence_continuation.v7","numeric_sequence_continuation.v8",8,"site_live","numeric_sequence"),
}


def artifact(path,kind): return {"path":path,"sha256":file_sha256(REPO/path),"kind":kind,"status":"frozen"}
def metric(name,estimate,bar): return {"name":name,"estimate":estimate,"ci95":None,"bar":bar}


def add_family(claim,family):
    if not any(x["family_id"]==family["family_id"] for x in claim["counterfactual_families"]):
        claim["counterfactual_families"].append(family)


def make_event(record,event_id,claim_id,test_type,verdict,failure,families,result_id,metrics,notes,prereg):
    event={"event_id":event_id,"claim_id":claim_id,"test_type":test_type,"stage":"complete","verdict":verdict,
           "failure_kind":failure,"family_ids":families,"site_id":"final_label_l0_value_through_l8h3_h7",
           "split_plan_id":SPLIT,"evaluation_role":"fresh_spaced_nonadjacent_ood_primary",
           "metrics":metrics,"prereg_artifact_id":prereg,"result_artifact_id":result_id,
           "input_artifact_ids":["spaced_rows_v1","spaced_capability_result_v1"],"seed":None,
           "checkpoint_sha256":CHECKPOINT,"supersedes_event_id":None,"replicates_event_id":None,
           "sections":[ARTIFACTS[result_id][0]],"notes":notes}
    event["design_key"]=design_key(record,event); event["execution_key"]=execution_key(record,event)
    return event


def build(tag,capability,transfer):
    old_id,new_id,revision,status,namespace=DESTINATIONS[tag]
    record=json.loads(circuit_path(tag).read_text())
    if any(c["claim_id"]==new_id for c in record["claims"]): return record,False
    if record["claims"][-1]["claim_id"]!=old_id: raise RuntimeError(f"{tag} authority moved")
    for aid,(path,kind) in ARTIFACTS.items(): record["artifacts"][aid]=artifact(path,kind)
    if not any(x["split_plan_id"]==SPLIT for x in record["split_plans"]):
        record["split_plans"].append({"split_plan_id":SPLIT,
            "unit":"model-free paired numeric endpoint group crossed with prompt construction and role",
            "partition_artifact_id":"spaced_rows_v1","builder_artifact_id":"spaced_rows_builder_v1","seed":None,
            "groups":{"fresh_plain":8,"fresh_headered":8},
            "leakage_group_keys":["paired endpoint texts","construction","lexical triple","nonadjacent value pair"],
            "sealed_before_outcomes":True,"sealed_at":"2026-09-14T21:44:00Z"})
    claim=deepcopy(record["claims"][-1]); claim.update({"claim_id":new_id,"revision":revision,"supersedes":old_id,"status":status})
    claim["split_plan_ids"]=[*claim["split_plan_ids"],SPLIT]
    add_family(claim,{"family_id":"list_step_two_spaced_ood","role":"interchange",
        "changes":["nonadjacent final list label and successor answer"],
        "holds_fixed":["three-line step-two list structure","endpoint separation of five","prompt construction"],
        "builder_artifact_id":"spaced_rows_builder_v1","control_ids":["digit copy","number-word copy"],
        "split_plan_id":SPLIT,"status":"validated"})
    if namespace=="numeric_sequence":
        for family,representation in (("digit_copy_spaced_ood","digits"),("word_copy_spaced_ood","number words")):
            add_family(claim,{"family_id":family,"role":"invariance","changes":["repeated numeric value and paired endpoint"],
                "holds_fixed":[representation,"copy relation","capable native answer"],"builder_artifact_id":"spaced_rows_builder_v1",
                "control_ids":["numbered-list successor","paired alternate representation"],"split_plan_id":SPLIT,"status":"validated"})
    ids=[f"{namespace}.spaced_ood_capability.held.v1",f"{namespace}.spaced_cached_bidirectional.held.v1",
         f"{namespace}.spaced_cached_list_selectivity.null.v1"]
    claim["evidence_event_ids"]=[*claim["evidence_event_ids"],*ids]
    for site in claim["candidate_sites"]:
        if site["site_id"]=="final_label_l0_value_through_l8h3_h7": site["ceiling_event_ids"]=[*site["ceiling_event_ids"],ids[1],ids[2]]
    if namespace=="numbered_list":
        claim["causal_variable"]["read"]="the final visible numeric label through a shared L8H3+H7 cached-value payload, conditioned by list context"
        claim["next_missing"]=("Exact L8H3+H7 final-source cached value now transfers the numbered-list successor bidirectionally on two capable "
            "nonadjacent OOD constructions, but strict list selectivity fails on capable word-copy CE response. The shared numeric/copy "
            "payload is weight-computed; list-specific routing or suffix context remains external. Next test a preregistered payload-by-context "
            "interaction on this frozen authority. Do not repeat capability, adjacent pairing, threshold, gain, or cached-payload transfer screens.")
    else:
        claim["causal_variable"]["read"]="a shared final numeric-token cached-value payload at L8H3+H7; arithmetic-relation state remains unlocalized"
        claim["next_missing"]=("A capable fresh authority shows that the exact L8H3+H7 cached term is a shared numeric/copy payload: it transfers "
            "numbered-list successor bidirectionally and causes a preregistered word-copy CE response while preserving answers. This does not "
            "localize the +1 relation for comma-separated numeric sequences. Next establish capable nonadjacent +1 digit/word rows and separate "
            "shared value payload from relation/router state. Do not repeat the invalid v5 instrument or threshold-rescue copy selectivity.")
    capability_metrics=[metric("minimum_native_accuracy",min(x["accuracy"] for x in capability["cell_reports"].values()),">=0.85 each cell"),
                        metric("observed_forwards",capability["price"]["observed_forwards"],"1")]
    target=[v for k,v in transfer["cell_reports"].items() if "|list_step_two_spaced|" in k]
    failed=[v for k,v in transfer["cell_reports"].items() if not v["passes"] and "control" in k]
    transfer_metrics=[metric("minimum_directional_donorward_fraction",min(x["donorward_fraction"] for x in target),">=0.75"),
                      metric("minimum_directional_donor_answer_win_fraction",min(x["donor_answer_win_fraction"] for x in target),">=0.50"),
                      metric("directional_mean_margin_effect_range",[min(x["mean_donor_margin_effect"] for x in target),max(x["mean_donor_margin_effect"] for x in target)],">0 each"),
                      metric("native_replay_relative_squared_error",transfer["exactness"]["native_replay_relative_squared_error"],"<=1e-10")]
    null_metrics=[metric("minimum_copy_answer_preservation",min(v.get("recipient_answer_preservation",1) for k,v in transfer["cell_reports"].items() if "control" in k),">=0.85"),
                  metric("maximum_copy_absolute_mean_ce_change",max(v.get("absolute_mean_recipient_ce_change",0) for k,v in transfer["cell_reports"].items() if "control" in k),"<=0.10"),
                  metric("failed_control_cell_count",len(failed),"0 required")]
    record["claims"].append(claim)
    events=[
      make_event(record,ids[0],new_id,"ood","held",None,[],"spaced_capability_result_v1",capability_metrics,"All six fresh nonadjacent target/control cells are natively capable before intervention.","spaced_capability_prereg_v1"),
      make_event(record,ids[1],new_id,"cross_family_transfer","held",None,["list_step_two_spaced_ood"],"spaced_transfer_result_v1",transfer_metrics,"Exact H3+H7 cached-value replacement transfers the numbered-list successor in both directions and constructions.","spaced_transfer_prereg_v1"),
      make_event(record,ids[2],new_id,"null_control","null","scientific_null",[],"spaced_transfer_result_v1",null_metrics,"Strict list selectivity fails because one capable word-copy direction exceeds the frozen CE-response bar; no threshold rescue.","spaced_transfer_prereg_v1"),
    ]
    record["evidence_events"].extend(events); validate_v2(record)
    return record,True


def main():
    capability=json.loads(CAPABILITY.read_text()); transfer=json.loads(TRANSFER.read_text())
    if capability["terminal"]!="capability_pass" or transfer["terminal"]!="shared_numeric_copy_payload": raise RuntimeError("terminal changed")
    built={tag:build(tag,capability,transfer) for tag in DESTINATIONS}
    if any(changed for _,changed in built.values()):
        with _lock("registry"):
            for tag,(record,changed) in built.items():
                if changed:_atomic_json(circuit_path(tag),record)
    rebuild_registry_v2()
    for tag in DESTINATIONS: validate_v2(json.loads(circuit_path(tag).read_text()))
    print(json.dumps({tag:DESTINATIONS[tag][1] for tag in DESTINATIONS},indent=2))


if __name__=="__main__":main()
