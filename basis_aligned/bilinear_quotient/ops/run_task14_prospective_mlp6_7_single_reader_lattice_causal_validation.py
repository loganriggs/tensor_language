#!/usr/bin/env python3
"""Validate sealed one-reader predictions on the complete E/A/U/W causal lattice."""

# BQGATE: EXPERIMENT pred_a_instrument_and_temporal_seal pred_b_single_reader_full_lattice pred_c_unopened_intermediate_composition pred_d_direction_template_stability pred_e_cardinality_stability pred_f_center_beats_native_base_reader
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
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
PREDICTION=ROOT/"circuits/fast_screens/task14_prospective_mlp6_7_single_reader_full_lattice_v1_predictions.json"
OUT=ROOT/"circuits/fast_screens/task14_prospective_mlp6_7_single_reader_full_lattice_v1_result.json"
PRIOR_ART_SHA256="47716d90ac91975aeb4e0d20d3c6e837c663c0a43ba3161bae07f1054e82ea75"
PREDICTION_SHA256="a0536512da57d43bb13dd8ab0fa08e8e87dc7aecc1fbbc80436b43584671ae31"
CAPABILITY_RESULT_SHA256="9ee68c9297995cc5cf1f6a7c29759c7199b258ec35974fdf5c4000d3e5085749"
CAPABILITY_LICENSE_SHA256="27acd0cb5e7459630f89188abd2160622e07967ba0ee9194bf26708801fde33c"
SUBSETS=gate.BACKGROUND_SUBSETS; METHODS=("base","exact"); PATCH_CHUNK_ROWS=256
BARS={"maximum_numerical_absolute_error":5e-5,"minimum_overall_cosine":.995,
    "maximum_overall_relative_l2_error":.05,"minimum_overall_sign_agreement":.95,
    "minimum_intermediate_cosine":.99,"maximum_intermediate_relative_l2_error":.08,
    "minimum_intermediate_sign_agreement":.90,"minimum_cell_cosine":.98,
    "maximum_cell_relative_l2_error":.10,"minimum_cardinality_cosine":.98,
    "maximum_cardinality_relative_l2_error":.10,"minimum_center_error_reduction":.30}
PRED_KEYS=("pred_a_instrument_and_temporal_seal","pred_b_single_reader_full_lattice",
    "pred_c_unopened_intermediate_composition","pred_d_direction_template_stability",
    "pred_e_cardinality_stability","pred_f_center_beats_native_base_reader",)


class SingleReaderValidationError(ValueError): pass
def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=32):
    installs=row_count*len(SUBSETS)*len(METHODS); chunks=math.ceil(installs/PATCH_CHUNK_ROWS)
    return {"physical_model_forwards":1+chunks,
        "example_evaluations":row_count*len(authority.ROLES)+installs,
        "causal_interventions":installs,"backwards":0,"parameter_updates":0,
        "maximum_patch_chunk_rows":PATCH_CHUNK_ROWS,"patch_chunks":chunks}


def _load_prediction():
    if _sha256(PREDICTION)!=PREDICTION_SHA256: raise SingleReaderValidationError("prediction changed")
    value=json.loads(PREDICTION.read_text())
    if value.get("terminal")!="sealed_prediction" or value.get("intermediate_causal_outcomes_opened") is not False \
            or not all(value.get("predictions",{}).values()):
        raise SingleReaderValidationError("prediction is not a valid seal")
    return value


def validate_preflight():
    for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),
        (capability.RESULT,CAPABILITY_RESULT_SHA256,"capability result"),
        (capability.LICENSE,CAPABILITY_LICENSE_SHA256,"capability license")):
        if _sha256(path)!=expected: raise SingleReaderValidationError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,
        expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    _load_prediction()
    expected={"physical_model_forwards":5,"example_evaluations":1120,
        "causal_interventions":1024,"backwards":0,"parameter_updates":0,
        "maximum_patch_chunk_rows":256,"patch_chunks":4}
    if derive_price()!=expected: raise SingleReaderValidationError("derived price changed")


def compile_plan():
    prediction=_load_prediction(); validate_preflight()
    return {"schema":"task14_prospective_mlp6_7_single_reader_lattice_validation_plan_v1",
        "candidate_id":"subject_verb.number_agreement.prospective_mlp6_7_single_reader_full_lattice_v1",
        "split":"PROSPECTIVE_INTERMEDIATE_CAUSAL_LATTICE","row_count":32,
        "background_subsets":list(SUBSETS),"methods":list(METHODS),
        "sealed_prediction_sha256":PREDICTION_SHA256,
        "sealed_prediction_created_utc":prediction["created_utc"],
        "prior_art_sha256":PRIOR_ART_SHA256,"capability_license_sha256":CAPABILITY_LICENSE_SHA256,
        "literal_scorer":"no fitted scale, offset, target endpoint, or group mean",
        "bars":dict(BARS),"predictions":dict(zip(PRED_KEYS,(
            "hashes, temporal seal, complete lattice, and closures pass",
            "all 512 central-reader amplitudes meet overall cosine/error/sign bars",
            "448 unopened intermediate amplitudes meet cosine/error/sign bars",
            "each direction-template cell meets cosine/error bars",
            "each background cardinality meets cosine/error bars",
            "central reader reduces error at least 30% versus native-base reader"))),
        "price":derive_price()}


def _compile_patch(tokens,heads,rows,torch):
    indices=[]; replacements=[]; specs=[]
    for row_index,row in enumerate(rows):
        for subset in SUBSETS:
            for method in METHODS:
                indices.append(row_index); replacements.append(heads[(subset,method)][row_index])
                specs.append((row_index,subset,method,row["direction_id"],row["template_id"]))
    index=torch.tensor(indices,dtype=torch.long,device=tokens.device)
    return {"tokens":tokens[:len(rows)][index],"finals":torch.full_like(index,tangent.parent.SUBJECT_POSITION),
        "replacement_heads":torch.stack(replacements),
        "native_reinstall_mask":torch.zeros(len(specs),dtype=torch.bool,device=tokens.device),"specs":specs}


def evaluate(model,torch,F,facade):
    rows=authority.build_rows(); n=len(rows); parent=tangent.parent; device=next(model.parameters()).device
    tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
    _,captured,projection,role_closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade)
    roles={"recipient":tangent._role_slice(captured,0,n),"opposite":tangent._role_slice(captured,n,2*n)}
    input_roles={"recipient":tangent._role_slice(inputs,0,n),"opposite":tangent._role_slice(inputs,n,2*n)}
    function=tangent._head_function(model,roles["recipient"],roles["opposite"],
        model.transformer.h[parent.LAYER].attn,projection,torch,F)
    heads={}
    with torch.no_grad():
        for subset in SUBSETS:
            heads[(subset,"base")]=function(gate._raw_for(input_roles["recipient"],input_roles["opposite"],subset,F)).detach()
            heads[(subset,"exact")]=function(gate._raw_for(input_roles["recipient"],input_roles["opposite"],subset+"YZ",F)).detach()
        patch=_compile_patch(tokens,heads,rows,torch); margin={}; closures=[]
        for start in range(0,len(patch["specs"]),PATCH_CHUNK_ROWS):
            stop=min(start+PATCH_CHUNK_ROWS,len(patch["specs"]))
            logits,_,_,closure=parent.downstream._decomposed_forward(model,patch["tokens"][start:stop],
                patch["finals"][start:stop],torch,F,facade,
                replacement_heads=patch["replacement_heads"][start:stop],
                native_reinstall_mask=patch["native_reinstall_mask"][start:stop])
            closures.append(closure)
            for local,spec in enumerate(patch["specs"][start:stop]):
                row_index,subset,method,_,_=spec; endpoint=rows[row_index]["endpoints"]["opposite_same_lemma"]
                margin[(row_index,subset,method)]=float(logits[local,parent.SUBJECT_POSITION,endpoint["answer_id"]]
                    -logits[local,parent.SUBJECT_POSITION,endpoint["foil_id"]])
    evidence=[{"row_id":row["row_id"],"direction":row["direction_id"],"template":row["template_id"],
        "background":subset,"cardinality":len(subset),
        "actual_q":margin[(i,subset,"exact")]-margin[(i,subset,"base")]}
        for i,row in enumerate(rows) for subset in SUBSETS]
    exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error":max(x["state_sum_max_absolute_error"] for x in closures),
        "downstream_normalized_closure_max_absolute_error":max(x["normalized_state_max_absolute_error"] for x in closures)}
    return evidence,exactness


def _stats(items,field):
    a=[x["actual_q"] for x in items]; p=[x[field] for x in items]
    dot=sum(x*y for x,y in zip(a,p)); an=math.sqrt(sum(x*x for x in a)); pn=math.sqrt(sum(x*x for x in p))
    return {"count":len(items),"cosine":dot/max(an*pn,1e-30),
        "relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),
        "sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(items)}


def score(causal,exactness,bars=BARS):
    sealed=_load_prediction(); pred={(x["row_id"],x["background"]):x for x in sealed["evidence"]}
    items=[]
    for actual in causal:
        p=pred.get((actual["row_id"],actual["background"]))
        if p is None: raise SingleReaderValidationError("missing sealed prediction")
        items.append({**actual,"central_reader_q":p["central_reader_q"],
            "native_base_reader_q":p["native_base_reader_q"]})
    overall={name:_stats(items,f"{name}_reader_q") for name in ("native_base","central")}
    intermediate=_stats([x for x in items if x["background"] not in {"","EAUW"}],"central_reader_q")
    cells={f"{d}__{t}":_stats([x for x in items if x["direction"]==d and x["template"]==t],"central_reader_q")
        for d in ("plural_to_singular","singular_to_plural") for t in ("above_below","below_above")}
    cardinality={str(k):_stats([x for x in items if x["cardinality"]==k],"central_reader_q") for k in range(5)}
    central=overall["central"]; reduction=1-central["relative_l2_error"]/max(overall["native_base"]["relative_l2_error"],1e-30)
    instrument=len(items)==512 and len({(x["row_id"],x["background"]) for x in items})==512 and all(
        x<=bars["maximum_numerical_absolute_error"] for x in exactness.values())
    b=central["cosine"]>=bars["minimum_overall_cosine"] and central["relative_l2_error"]<=bars["maximum_overall_relative_l2_error"] and central["sign_agreement"]>=bars["minimum_overall_sign_agreement"]
    c=intermediate["cosine"]>=bars["minimum_intermediate_cosine"] and intermediate["relative_l2_error"]<=bars["maximum_intermediate_relative_l2_error"] and intermediate["sign_agreement"]>=bars["minimum_intermediate_sign_agreement"]
    d=all(x["cosine"]>=bars["minimum_cell_cosine"] and x["relative_l2_error"]<=bars["maximum_cell_relative_l2_error"] for x in cells.values())
    e=all(x["cosine"]>=bars["minimum_cardinality_cosine"] and x["relative_l2_error"]<=bars["maximum_cardinality_relative_l2_error"] for x in cardinality.values())
    predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and b),bool(instrument and c),
        bool(instrument and d),bool(instrument and e),bool(instrument and reduction>=bars["minimum_center_error_reduction"]))))
    return {**exactness,"overall":overall,"intermediate_only":intermediate,
        "by_direction_template":cells,"by_cardinality":cardinality,
        "center_error_reduction_over_native_base":reduction,"predictions":predictions,"joined_evidence":items}


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv); plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":
        print(json.dumps(plan,sort_keys=True)); return
    if OUT.exists(): raise SingleReaderValidationError(f"refusing to overwrite {OUT}")
    torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
    causal,exactness=evaluate(model,torch,F,facade); scored=score(causal,exactness)
    terminal="valid_causal_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
    result={"schema":"task14_prospective_mlp6_7_single_reader_lattice_validation_result_v1",
        "candidate_id":plan["candidate_id"],"terminal":terminal,
        "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,
        "checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,
        "causal_evidence":causal,"sealed_prediction_sha256":PREDICTION_SHA256}
    payload=managed.atomic_create_json(OUT,result)
    print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],
        "result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
