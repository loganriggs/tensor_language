#!/usr/bin/env python3
"""Seal full-lattice predictions from one central downstream reader per row."""

# BQGATE: EXPERIMENT pred_a_capability_and_parent_license pred_b_gradient_instrument pred_c_five_hundred_twelve_predictions_sealed
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate
import run_task14_prospective_jvp_amplitude_native_capability as capability


ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_prospective_mlp6_7_single_reader_full_lattice_v1.json"
PARENT_RESULT=ROOT/"circuits/fast_screens/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_result.json"
OUT=ROOT/"circuits/fast_screens/task14_prospective_mlp6_7_single_reader_full_lattice_v1_predictions.json"
PRIOR_ART_SHA256="47716d90ac91975aeb4e0d20d3c6e837c663c0a43ba3161bae07f1054e82ea75"
PARENT_RESULT_SHA256="714f61a8d3911564160236276c108900d341037b1046851fa19be00d5035463d"
CAPABILITY_RESULT_SHA256="9ee68c9297995cc5cf1f6a7c29759c7199b258ec35974fdf5c4000d3e5085749"
CAPABILITY_LICENSE_SHA256="27acd0cb5e7459630f89188abd2160622e07967ba0ee9194bf26708801fde33c"
SUBSETS=gate.BACKGROUND_SUBSETS
POINTS=("native_base","central",)
PRED_KEYS=("pred_a_capability_and_parent_license","pred_b_gradient_instrument",
           "pred_c_five_hundred_twelve_predictions_sealed",)
MAXIMUM_NUMERICAL_ABSOLUTE_ERROR=5e-5


class SingleReaderPredictionError(ValueError): pass


def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=32):
    return {"physical_model_forwards":3,
        "example_evaluations":row_count*len(authority.ROLES)+row_count*len(POINTS),
        "backwards":2,"causal_interventions":0,"sealed_predictions":row_count*len(SUBSETS),
        "parameter_updates":0}


def validate_preflight():
    for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),
        (PARENT_RESULT,PARENT_RESULT_SHA256,"parent result"),
        (capability.RESULT,CAPABILITY_RESULT_SHA256,"capability result"),
        (capability.LICENSE,CAPABILITY_LICENSE_SHA256,"capability license")):
        if _sha256(path)!=expected: raise SingleReaderPredictionError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,
        expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    parent=json.loads(PARENT_RESULT.read_text())
    if parent.get("terminal")!="valid_causal_screen" or not all(parent.get("score",{}).get("predictions",{}).values()):
        raise SingleReaderPredictionError("prospective endpoint parent does not license lattice test")
    if derive_price()!={"physical_model_forwards":3,"example_evaluations":160,
        "backwards":2,"causal_interventions":0,"sealed_predictions":512,"parameter_updates":0}:
        raise SingleReaderPredictionError("derived price changed")


def compile_plan():
    validate_preflight()
    return {"schema":"task14_prospective_mlp6_7_single_reader_lattice_prediction_plan_v1",
        "candidate_id":"subject_verb.number_agreement.prospective_mlp6_7_single_reader_full_lattice_v1",
        "split":"SEALED_BEFORE_INTERMEDIATE_CAUSAL_LATTICE","row_count":32,
        "background_subsets":list(SUBSETS),"linearization_points":list(POINTS),
        "center":"arithmetic mean of all 32 base/exact background heads per row",
        "prior_art_sha256":PRIOR_ART_SHA256,"parent_result_sha256":PARENT_RESULT_SHA256,
        "capability_license_sha256":CAPABILITY_LICENSE_SHA256,
        "causal_outcomes_opened":"ENDPOINTS_ONLY; 448 INTERMEDIATE TARGETS CLOSED",
        "predictions":dict(zip(PRED_KEYS,("capability and prospective endpoint parent validate",
            "all closures <=5e-5 and both 32-row gradient batches finite/nonzero",
            "exactly 512 unique central and native-base predictions atomically sealed"))),
        "price":derive_price()}


def _signed_margins(logits,rows,torch):
    return torch.stack([logits[i,tangent.parent.SUBJECT_POSITION,
        row["endpoints"]["opposite_same_lemma"]["answer_id"]]-logits[i,tangent.parent.SUBJECT_POSITION,
        row["endpoints"]["opposite_same_lemma"]["foil_id"]] for i,row in enumerate(rows)])


def evaluate(model,torch,F,facade):
    rows=authority.build_rows(); n=len(rows); parent=tangent.parent; device=next(model.parameters()).device
    tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
    _,captured,projection,role_closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade)
    roles={"recipient":tangent._role_slice(captured,0,n),"opposite":tangent._role_slice(captured,n,2*n)}
    input_roles={"recipient":tangent._role_slice(inputs,0,n),"opposite":tangent._role_slice(inputs,n,2*n)}
    function=tangent._head_function(model,roles["recipient"],roles["opposite"],
        model.transformer.h[parent.LAYER].attn,projection,torch,F)
    bases,exacts={},{}
    with torch.no_grad():
        for subset in SUBSETS:
            bases[subset]=function(gate._raw_for(input_roles["recipient"],input_roles["opposite"],subset,F)).detach()
            exacts[subset]=function(gate._raw_for(input_roles["recipient"],input_roles["opposite"],subset+"YZ",F)).detach()
    stacked=torch.stack([value for subset in SUBSETS for value in (bases[subset],exacts[subset])])
    central=stacked.mean(dim=0); native_base=bases[""]; gradients={}; closures={}; gradient_stats={}
    for point,center in (("native_base",native_base),("central",central)):
        replacement=center.detach().clone().requires_grad_(True)
        logits,_,_,closure=parent.downstream._decomposed_forward(model,tokens[:n],
            torch.full((n,),parent.SUBJECT_POSITION,dtype=torch.long,device=device),torch,F,facade,
            replacement_heads=replacement,native_reinstall_mask=torch.zeros(n,dtype=torch.bool,device=device))
        grad=torch.autograd.grad(_signed_margins(logits,rows,torch).sum(),replacement)[0]
        gradients[point]=grad.detach(); closures[point]=closure
        gradient_stats[point]={"finite":bool(torch.isfinite(grad).all()),
            "l2_norm":float(torch.linalg.vector_norm(grad)),
            "nonzero_row_count":int((torch.linalg.vector_norm(grad,dim=-1)>0).sum())}
    evidence=[]
    for row_index,row in enumerate(rows):
        for subset in SUBSETS:
            delta=exacts[subset][row_index]-bases[subset][row_index]
            evidence.append({"row_id":row["row_id"],"direction":row["direction_id"],
                "template":row["template_id"],"background":subset,"cardinality":len(subset),
                "native_base_reader_q":float(torch.dot(gradients["native_base"][row_index],delta)),
                "central_reader_q":float(torch.dot(gradients["central"][row_index],delta)),
                "head_delta_l2_norm":float(torch.linalg.vector_norm(delta))})
    exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error":max(x["state_sum_max_absolute_error"] for x in closures.values()),
        "downstream_normalized_closure_max_absolute_error":max(x["normalized_state_max_absolute_error"] for x in closures.values())}
    instrument=all(x<=MAXIMUM_NUMERICAL_ABSOLUTE_ERROR for x in exactness.values()) and all(
        x["finite"] and x["l2_norm"]>0 and x["nonzero_row_count"]==32 for x in gradient_stats.values())
    unique={(x["row_id"],x["background"]) for x in evidence}
    predictions=dict(zip(PRED_KEYS,(True,bool(instrument),bool(len(evidence)==512 and len(unique)==512))))
    return evidence,exactness,gradient_stats,predictions


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv); plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(plan,sort_keys=True)); return
    if OUT.exists(): raise SingleReaderPredictionError(f"refusing to overwrite {OUT}")
    torch,F,facade=tangent.parent.factors._dependencies()
    model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    evidence,exactness,gradient_stats,predictions=evaluate(model,torch,F,facade)
    terminal="sealed_prediction" if all(predictions.values()) else "invalid"
    result={"schema":"task14_prospective_mlp6_7_single_reader_lattice_predictions_v1",
        "candidate_id":plan["candidate_id"],"terminal":terminal,
        "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"exactness":exactness,
        "gradient_stats":gradient_stats,"predictions":predictions,"evidence":evidence,
        "intermediate_causal_outcomes_opened":False}
    payload=managed.atomic_create_json(OUT,result)
    print(json.dumps({"terminal":terminal,"predictions":predictions,
        "result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
