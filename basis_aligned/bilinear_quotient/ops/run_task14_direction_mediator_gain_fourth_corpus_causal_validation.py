#!/usr/bin/env python3
"""Prospectively validate fixed Task14 mediator gains on the fourth corpus."""

# BQGATE: EXPERIMENT pred_a_authority_capability_seal_and_instrument pred_b_fixed_reader_predicts_program_effect pred_c_mlp15_transfer pred_d_mlp17_and_interaction_transfer pred_e_joint_and_each_template_transfer pred_f_exact_fixed_program_and_price
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_direction_mediator_gain_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_direction_mediator_gain_transfer_native_capability as capability
import run_task14_mlp6_7_direction_cardinality_program_mlp15_vs_mlp17_mediation as split
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as program
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_mlp6_7_direction_mediator_gain_fourth_corpus_transfer_v1.json"
PREDICTIONS=ROOT/"circuits/fast_screens/task14_direction_mediator_gain_fourth_corpus_predictions_v1.json"
OUT=ROOT/"circuits/followups/task14_direction_mediator_gain_fourth_corpus_causal_validation_v1_result.json"
PRIOR_ART_SHA256="ff633ae538629fe0a2cbded343fd0c1efab8511d8d617fee6d77577edb02cd53"
PREDICTION_SHA256="e5ee7b10136793d8b3fdc30415070411f236404248e5f9635dd022364d633ea9"
CAPABILITY_RESULT_SHA256="8e204febc3477ea1a6e8e914356e77ff05ea3eb527d8fb0dbd6e69e247af9733"
CAPABILITY_LICENSE_SHA256="f8dba12da6d6d9668017ae5687af64dd90868e5110c215ec2006d20ad01b2e5a"
SUBSETS=factor_gate.BACKGROUND_SUBSETS
ARMS=split.ARMS
SPEC_CHUNK=split.SPEC_CHUNK
MAX_ERROR=5e-5
BARS={"reader_min_cosine":.90,"reader_max_relative_l2":.45,"reader_min_sign":.85,"m15_min_cosine":.75,"m15_max_relative_l2":.75,"m15_min_sign":.65,"m17_min_cosine":.90,"m17_max_relative_l2":.40,"m17_min_sign":.90,"interaction_min_cosine":.90,"interaction_max_relative_l2":.40,"interaction_min_sign":.90,"joint_min_cosine":.90,"joint_max_relative_l2":.40,"joint_min_sign":.90,"template_joint_min_cosine":.85}
PRED_KEYS=("pred_a_authority_capability_seal_and_instrument","pred_b_fixed_reader_predicts_program_effect","pred_c_mlp15_transfer","pred_d_mlp17_and_interaction_transfer","pred_e_joint_and_each_template_transfer","pred_f_exact_fixed_program_and_price")


class TransferError(ValueError): pass
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price():
    cells=32*len(SUBSETS);return {"physical_model_forwards":1+math.ceil(cells/SPEC_CHUNK),"example_evaluations":32*3+cells*len(ARMS),"causal_installations":cells*7,"mediator_clamps":cells*2*4,"backwards":0,"parameter_updates":0,"maximum_forward_batch":SPEC_CHUNK*len(ARMS)}


def validate_preflight():
    for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),(PREDICTIONS,PREDICTION_SHA256,"sealed predictions"),(program.PROTOTYPES,program.PROTOTYPE_SHA256,"prototypes"),(capability.RESULT,CAPABILITY_RESULT_SHA256,"capability"),(capability.LICENSE,CAPABILITY_LICENSE_SHA256,"license")):
        if _sha(path)!=expected: raise TransferError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,expected_license_sha256=CAPABILITY_LICENSE_SHA256,causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    sealed=json.loads(PREDICTIONS.read_text())
    if sealed.get("terminal")!="sealed_prediction" or sealed.get("fourth_corpus_causal_outcomes_opened") is not False or sealed.get("fourth_corpus_model_forwards")!=0 or not all(sealed.get("predictions",{}).values()): raise TransferError("prediction seal invalid")


def compile_plan():
    validate_preflight();return {"schema":"task14_direction_mediator_gain_fourth_corpus_causal_validation_plan_v1","candidate_id":authority.CAUSAL_CANDIDATE_ID,"split":"FOURTH_CORPUS_COMPLETE_CAUSAL_LATTICE","row_count":32,"background_subsets":list(SUBSETS),"arms":list(ARMS),"prior_art_sha256":PRIOR_ART_SHA256,"prediction_sha256":PREDICTION_SHA256,"capability_license_sha256":CAPABILITY_LICENSE_SHA256,"bars":dict(BARS),"price":derive_price(),"fit_operations":0,"gain_vector_reader_changes":0}


def evaluate(model,torch,F,facade):
    artifact,_=program._load_artifacts();rows=authority.build_rows();count=len(rows);parent=tangent.parent;device=next(model.parameters()).device
    role_tokens,role_finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
    _,captured,projection,role_closure,inputs=parent._decomposed_forward(model,role_tokens,role_finals,torch,F,facade)
    roles={"recipient":tangent._role_slice(captured,0,count),"opposite":tangent._role_slice(captured,count,2*count)};input_roles={"recipient":tangent._role_slice(inputs,0,count),"opposite":tangent._role_slice(inputs,count,2*count)}
    function=tangent._head_function(model,roles["recipient"],roles["opposite"],model.transformer.h[parent.LAYER].attn,projection,torch,F)
    vectors={key:torch.tensor(value["coordinates"],dtype=torch.float32,device=device) for key,value in artifact["prototypes"].items() if ".cardinality_" in key};cells=[]
    with torch.no_grad():
        for subset in SUBSETS:
            base_heads=function(factor_gate._raw_for(input_roles["recipient"],input_roles["opposite"],subset,F)).detach()
            for index,row in enumerate(rows): cells.append((index,subset,base_heads[index],base_heads[index]+vectors[f'{row["direction_id"]}.cardinality_{len(subset)}']))
        margins,closures={},[]
        for start in range(0,len(cells),SPEC_CHUNK):
            chunk=cells[start:start+SPEC_CHUNK];row_indices=[];heads=[];specs=[]
            for index,subset,base_head,program_head in chunk:
                for arm in ARMS: row_indices.append(index);heads.append(program_head if arm.startswith("program") else base_head);specs.append((index,subset,arm))
            index_tensor=torch.tensor(row_indices,dtype=torch.long,device=device);tokens=role_tokens[:count][index_tensor];finals=torch.full_like(index_tensor,parent.SUBJECT_POSITION);handles=split._install_clamps(model,finals,torch)
            try: logits,_,_,closure=parent.downstream._decomposed_forward(model,tokens,finals,torch,F,facade,replacement_heads=torch.stack(heads),native_reinstall_mask=torch.zeros(len(specs),dtype=torch.bool,device=device))
            finally:
                for handle in handles: handle.remove()
            closures.append(closure)
            for local,(index,subset,arm) in enumerate(specs):
                endpoint=rows[index]["endpoints"]["opposite_same_lemma"];margins[(index,subset,arm)]=float(logits[local,parent.SUBJECT_POSITION,endpoint["answer_id"]]-logits[local,parent.SUBJECT_POSITION,endpoint["foil_id"]])
    evidence=[]
    for index,row in enumerate(rows):
        for subset in SUBSETS:
            values={arm:margins[(index,subset,arm)] for arm in ARMS};q={name:values[f"program_{name}"]-values[f"base_{name}"] for name in ("empty","15","17","both")};m15,m17,mboth=q["empty"]-q["15"],q["empty"]-q["17"],q["empty"]-q["both"]
            evidence.append({"row_id":row["row_id"],"direction":row["direction_id"],"template":row["template_id"],"background":subset,"cardinality":len(subset),**values,**{f"q_{k}":v for k,v in q.items()},"m15":m15,"m17":m17,"m_both":mboth,"interaction":mboth-m15-m17})
    exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],"role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],"downstream_state_closure_max_absolute_error":max(x["state_sum_max_absolute_error"] for x in closures),"downstream_normalized_closure_max_absolute_error":max(x["normalized_state_max_absolute_error"] for x in closures)}
    return evidence,exactness


def _stats(rows,actual,predicted):
    a=[x[actual] for x in rows];p=[x[predicted] for x in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(a)}


def _passes(s,min_cos,max_err,min_sign): return s["cosine"]>=min_cos and s["relative_l2_error"]<=max_err and s["sign_agreement"]>=min_sign


def score(evidence,exactness):
    sealed={(x["row_id"],x["background"]):x for x in json.loads(PREDICTIONS.read_text())["evidence"]};joined=[]
    for x in evidence:
        p=sealed.get((x["row_id"],x["background"]));
        if p is None: raise TransferError("missing sealed prediction")
        joined.append({**x,**{k:p[k] for k in ("sealed_reader_q","sealed_m15","sealed_m17","sealed_interaction","sealed_joint_mediation")}})
    base_replay=max(abs(x["base_empty"]-x[f"base_{site}"]) for x in joined for site in ("15","17","both"));reader=_stats(joined,"q_empty","sealed_reader_q");m15=_stats(joined,"m15","sealed_m15");m17=_stats(joined,"m17","sealed_m17");interaction=_stats(joined,"interaction","sealed_interaction");joint=_stats(joined,"m_both","sealed_joint_mediation");templates={t:_stats([x for x in joined if x["template"]==t],"m_both","sealed_joint_mediation") for t in ("under_beyond","beyond_under")}
    instrument=len(joined)==512 and len({(x["row_id"],x["background"]) for x in joined})==512 and base_replay<=MAX_ERROR and all(x<=MAX_ERROR for x in exactness.values())
    pred_b=_passes(reader,BARS["reader_min_cosine"],BARS["reader_max_relative_l2"],BARS["reader_min_sign"]);pred_c=_passes(m15,BARS["m15_min_cosine"],BARS["m15_max_relative_l2"],BARS["m15_min_sign"]);pred_d=_passes(m17,BARS["m17_min_cosine"],BARS["m17_max_relative_l2"],BARS["m17_min_sign"]) and _passes(interaction,BARS["interaction_min_cosine"],BARS["interaction_max_relative_l2"],BARS["interaction_min_sign"]);pred_e=_passes(joint,BARS["joint_min_cosine"],BARS["joint_max_relative_l2"],BARS["joint_min_sign"]) and all(x["cosine"]>=BARS["template_joint_min_cosine"] for x in templates.values());price=derive_price();pred_f=price["physical_model_forwards"]<=17 and price["example_evaluations"]<=4192
    predictions=dict(zip(PRED_KEYS,(instrument,instrument and pred_b,instrument and pred_c,instrument and pred_d,instrument and pred_e,pred_f)));terminal="invalid" if not(predictions[PRED_KEYS[0]] and predictions[PRED_KEYS[5]]) else "prospective_program_screen" if all(predictions.values()) else "null" if not predictions[PRED_KEYS[4]] else "inconclusive"
    return {**exactness,"base_replay_max_absolute_error":base_replay,"reader_program_effect":reader,"mlp15":m15,"mlp17":m17,"interaction":interaction,"joint_mediation":joint,"joint_by_template":templates,"predictions":predictions,"terminal":terminal,"joined_evidence":joined}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);plan=compile_plan()
    if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True));return
    if OUT.exists(): raise TransferError(f"refusing overwrite {OUT}")
    torch,F,facade=tangent.parent.factors._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True);evidence,exactness=evaluate(model,torch,F,facade);scored=score(evidence,exactness);payload=managed.atomic_create_json(OUT,{"schema":"task14_direction_mediator_gain_fourth_corpus_causal_validation_result_v1","candidate_id":authority.CAUSAL_CANDIDATE_ID,"terminal":scored["terminal"],"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored})
    print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__":main()
