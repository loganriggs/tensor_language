#!/usr/bin/env python3
"""Transfer downstream margin readers leave-one-row-out within answer direction."""

# BQGATE: EXPERIMENT pred_a_gradient_instrument pred_b_direction_reader_transfers pred_c_reader_transfers_each_template pred_d_reader_transfers_intermediate_composition pred_e_direction_grouping_beats_global pred_f_reader_geometry_is_stable
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_mlp6_7_direction_shared_reader_transfer_v1.json"
PARENT_RESULT=ROOT/"circuits/fast_screens/task14_prospective_mlp6_7_single_reader_full_lattice_v1_result.json"
OUT=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_shared_reader_transfer_v1_result.json"
PRIOR_ART_SHA256="5a5a70bab7fe286707b29e1e47a3f044d72b4282a81802ae7f685a24e40d011d"
PARENT_RESULT_SHA256="3185f6632b12109236ac127eb9eb297eea8185949d6e4b396b332e530ffda107"
SUBSETS=gate.BACKGROUND_SUBSETS
BARS={"maximum_numerical_absolute_error":5e-5,"minimum_overall_cosine":.95,
 "maximum_overall_relative_l2_error":.35,"minimum_overall_sign_agreement":.90,
 "minimum_template_cosine":.90,"maximum_template_relative_l2_error":.45,
 "minimum_template_sign_agreement":.85,"minimum_intermediate_cosine":.95,
 "maximum_intermediate_relative_l2_error":.35,"minimum_intermediate_sign_agreement":.90,
 "minimum_direction_sse_reduction_over_global":.20,"minimum_gradient_cosine":.80}
PRED_KEYS=("pred_a_gradient_instrument","pred_b_direction_reader_transfers",
 "pred_c_reader_transfers_each_template","pred_d_reader_transfers_intermediate_composition",
 "pred_e_direction_grouping_beats_global","pred_f_reader_geometry_is_stable",)

class SharedReaderError(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price(): return {"physical_model_forwards":2,"example_evaluations":128,
 "backwards":1,"causal_interventions":0,"parameter_updates":0}

def validate_preflight():
 if _sha256(PRIOR_ART)!=PRIOR_ART_SHA256: raise SharedReaderError("prior art changed")
 if _sha256(PARENT_RESULT)!=PARENT_RESULT_SHA256: raise SharedReaderError("parent result changed")
 parent=json.loads(PARENT_RESULT.read_text())
 if parent.get("terminal")!="valid_causal_screen" or not all(parent.get("score",{}).get("predictions",{}).values()):
  raise SharedReaderError("parent does not license reader sharing")
 if derive_price()!={"physical_model_forwards":2,"example_evaluations":128,
  "backwards":1,"causal_interventions":0,"parameter_updates":0}: raise SharedReaderError("price changed")

def compile_plan():
 validate_preflight()
 return {"schema":"task14_mlp6_7_direction_shared_reader_transfer_plan_v1",
  "candidate_id":"subject_verb.number_agreement.mlp6_7_direction_shared_reader_transfer_v1",
  "data_status":"RETROSPECTIVE_FROZEN_CAUSAL_LATTICE_NEW_READER_GRADIENTS",
  "row_count":32,"background_subsets":list(SUBSETS),"prior_art_sha256":PRIOR_ART_SHA256,
  "parent_result_sha256":PARENT_RESULT_SHA256,
  "reader_rule":"leave-one-row-out mean central gradient within answer direction",
  "control":"leave-one-row-out global mean central gradient",
  "predictions":dict(zip(PRED_KEYS,("closures and gradients valid","direction-LOO predicts all amplitudes",
   "direction-LOO transfers in each syntax","direction-LOO predicts intermediate compositions",
   "direction grouping beats global reader","row gradients align with direction-LOO reader"))),"bars":dict(BARS),"price":derive_price()}

def _margins(logits,rows,torch):
 return torch.stack([logits[i,tangent.parent.SUBJECT_POSITION,r["endpoints"]["opposite_same_lemma"]["answer_id"]]
  -logits[i,tangent.parent.SUBJECT_POSITION,r["endpoints"]["opposite_same_lemma"]["foil_id"]] for i,r in enumerate(rows)])

def evaluate(model,torch,F,facade):
 rows=authority.build_rows(); n=len(rows); parent=tangent.parent; device=next(model.parameters()).device
 tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
 _,captured,projection,role_closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade)
 roles={"recipient":tangent._role_slice(captured,0,n),"opposite":tangent._role_slice(captured,n,2*n)}
 ir={"recipient":tangent._role_slice(inputs,0,n),"opposite":tangent._role_slice(inputs,n,2*n)}
 fn=tangent._head_function(model,roles["recipient"],roles["opposite"],model.transformer.h[parent.LAYER].attn,projection,torch,F)
 with torch.no_grad():
  bases={s:fn(gate._raw_for(ir["recipient"],ir["opposite"],s,F)).detach() for s in SUBSETS}
  exacts={s:fn(gate._raw_for(ir["recipient"],ir["opposite"],s+"YZ",F)).detach() for s in SUBSETS}
  center=torch.stack([v for s in SUBSETS for v in (bases[s],exacts[s])]).mean(0)
 replacement=center.detach().clone().requires_grad_(True)
 logits,_,_,closure=parent.downstream._decomposed_forward(model,tokens[:n],torch.full((n,),parent.SUBJECT_POSITION,
  dtype=torch.long,device=device),torch,F,facade,replacement_heads=replacement,
  native_reinstall_mask=torch.zeros(n,dtype=torch.bool,device=device))
 gradients=torch.autograd.grad(_margins(logits,rows,torch).sum(),replacement)[0].detach()
 evidence=[]; min_cos=1.0
 for i,row in enumerate(rows):
  same=[j for j,r in enumerate(rows) if j!=i and r["direction_id"]==row["direction_id"]]
  others=[j for j in range(n) if j!=i]
  dg=gradients[same].mean(0); gg=gradients[others].mean(0)
  cos=float(torch.dot(gradients[i],dg)/(gradients[i].norm()*dg.norm()).clamp_min(1e-30)); min_cos=min(min_cos,cos)
  for s in SUBSETS:
   delta=exacts[s][i]-bases[s][i]
   evidence.append({"row_id":row["row_id"],"direction":row["direction_id"],"template":row["template_id"],
    "background":s,"cardinality":len(s),"direction_loo_q":float(torch.dot(dg,delta)),
    "global_loo_q":float(torch.dot(gg,delta)),"row_reader_q":float(torch.dot(gradients[i],delta))})
 exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],
  "role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],
  "downstream_state_closure_max_absolute_error":closure["state_sum_max_absolute_error"],
  "downstream_normalized_closure_max_absolute_error":closure["normalized_state_max_absolute_error"]}
 gs={"finite":bool(torch.isfinite(gradients).all()),"l2_norm":float(gradients.norm()),
  "nonzero_row_count":int((gradients.norm(dim=-1)>0).sum()),"minimum_direction_loo_cosine":min_cos}
 return evidence,exactness,gs

def _actual():
 p=json.loads(PARENT_RESULT.read_text()); return {(x["row_id"],x["background"]):x["actual_q"] for x in p["score"]["joined_evidence"]}
def _stats(items,field):
 a=[x["actual_q"] for x in items]; p=[x[field] for x in items]; dot=sum(x*y for x,y in zip(a,p)); an=math.sqrt(sum(x*x for x in a)); pn=math.sqrt(sum(x*x for x in p))
 return {"count":len(items),"cosine":dot/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),
  "sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(items),"sse":sum((x-y)**2 for x,y in zip(a,p))}

def score(evidence,exactness,gs,bars=BARS):
 actual=_actual(); items=[{**x,"actual_q":actual[(x["row_id"],x["background"])]} for x in evidence]
 overall={k:_stats(items,k+"_q") for k in ("direction_loo","global_loo","row_reader")}
 templates={t:_stats([x for x in items if x["template"]==t],"direction_loo_q") for t in ("above_below","below_above")}
 inter=_stats([x for x in items if x["background"] not in {"","EAUW"}],"direction_loo_q")
 reduction=1-overall["direction_loo"]["sse"]/max(overall["global_loo"]["sse"],1e-30); d=overall["direction_loo"]
 instrument=all(x<=bars["maximum_numerical_absolute_error"] for x in exactness.values()) and gs["finite"] and gs["nonzero_row_count"]==32
 b=d["cosine"]>=bars["minimum_overall_cosine"] and d["relative_l2_error"]<=bars["maximum_overall_relative_l2_error"] and d["sign_agreement"]>=bars["minimum_overall_sign_agreement"]
 c=all(x["cosine"]>=bars["minimum_template_cosine"] and x["relative_l2_error"]<=bars["maximum_template_relative_l2_error"] and x["sign_agreement"]>=bars["minimum_template_sign_agreement"] for x in templates.values())
 di=inter["cosine"]>=bars["minimum_intermediate_cosine"] and inter["relative_l2_error"]<=bars["maximum_intermediate_relative_l2_error"] and inter["sign_agreement"]>=bars["minimum_intermediate_sign_agreement"]
 predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and b),bool(instrument and c),bool(instrument and di),
  bool(instrument and reduction>=bars["minimum_direction_sse_reduction_over_global"]),bool(instrument and gs["minimum_direction_loo_cosine"]>=bars["minimum_gradient_cosine"]))))
 return {**exactness,"gradient_stats":gs,"overall":overall,"by_template":templates,"intermediate_only":inter,
  "direction_sse_reduction_over_global":reduction,"predictions":predictions}

def main(argv=None):
 parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise SharedReaderError(f"refusing to overwrite {OUT}")
 torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 evidence,exactness,gs=evaluate(model,torch,F,facade); scored=score(evidence,exactness,gs)
 terminal="valid_diagnostic" if scored["predictions"][PRED_KEYS[0]] else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_direction_shared_reader_transfer_result_v1",
  "candidate_id":plan["candidate_id"],"terminal":terminal,"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,
  "score":scored,"reader_evidence":evidence})
 print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
