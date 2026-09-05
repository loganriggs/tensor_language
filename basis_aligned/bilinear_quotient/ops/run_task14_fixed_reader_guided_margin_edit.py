#!/usr/bin/env python3
"""Use fixed-reader predictions to choose gains for an absolute signed margin edit."""

# BQGATE: EXPERIMENT pred_a_instrument_and_frozen_support pred_b_reader_guided_edit_hits_target pred_c_both_directions_hit_target pred_d_both_templates_hit_target pred_e_reader_guidance_beats_fixed_gain pred_f_no_target_outcome_calibration
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics
from pathlib import Path

import circuit_fast_screen_candidate_task14_fixed_reader_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_fixed_reader_guided_margin_edit_v1.json"
PREDICTION=ROOT/"circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_predictions.json"
PARENT_RESULT=ROOT/"circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json"
OUT=ROOT/"circuits/fast_screens/task14_fixed_reader_guided_margin_edit_v1_result.json"
PRIOR_ART_SHA256="bf8a13a82d4efc405f6bcd0927faf4dea92d23f3777e6063a21e17e7ab8d4170"
PREDICTION_SHA256="1d6a5ce082efb6f59b492b5d80c690979e1d3df1bd532d73edf49caaefc8cc81"
PARENT_RESULT_SHA256="e53160cff6407c27dfd3a0e6b15740984db0d8b4468fcb96a10d32cd4a5f13b9"
SUBSETS=gate.BACKGROUND_SUBSETS; METHODS=("base","guided","half"); SUPPORT_FLOOR=.05; TARGET_MAGNITUDE=.04; PATCH_CHUNK_ROWS=256
BARS={"maximum_numerical_absolute_error":5e-5,"minimum_overall_sign_agreement":.98,
 "maximum_median_absolute_target_error":.015,"maximum_p90_absolute_target_error":.03,
 "minimum_cell_sign_agreement":.95,"maximum_cell_median_absolute_target_error":.02,
 "minimum_mae_reduction_over_half_gain":.30}
PRED_KEYS=("pred_a_instrument_and_frozen_support","pred_b_reader_guided_edit_hits_target",
 "pred_c_both_directions_hit_target","pred_d_both_templates_hit_target",
 "pred_e_reader_guidance_beats_fixed_gain","pred_f_no_target_outcome_calibration",)
class GuidedEditError(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _prediction_rows():
 p=json.loads(PREDICTION.read_text()); return [x for x in p["evidence"] if abs(x["fixed_reader_q"])>=SUPPORT_FLOOR]
def derive_price():
 n=len(_prediction_rows()); installs=n*len(METHODS); chunks=math.ceil(installs/PATCH_CHUNK_ROWS)
 return {"physical_model_forwards":1+chunks,"example_evaluations":96+installs,"causal_installations":installs,
  "maximum_patch_chunk_rows":PATCH_CHUNK_ROWS,"patch_chunks":chunks,"backwards":0,"parameter_updates":0}
def validate_preflight():
 for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),(PREDICTION,PREDICTION_SHA256,"prediction"),(PARENT_RESULT,PARENT_RESULT_SHA256,"parent")):
  if _sha256(path)!=expected: raise GuidedEditError(f"{label} changed")
 p=json.loads(PREDICTION.read_text())
 if p.get("terminal")!="sealed_prediction" or p.get("causal_outcomes_opened") is not False: raise GuidedEditError("prediction not sealed")
 support=_prediction_rows(); counts={d:sum(x["direction"]==d for x in support) for d in ("plural_to_singular","singular_to_plural")}
 templates={t:sum(x["template"]==t for x in support) for t in ("above_inside","inside_above")}
 gains=[TARGET_MAGNITUDE/abs(x["fixed_reader_q"]) for x in support]
 if len(support)!=278 or counts!={"plural_to_singular":156,"singular_to_plural":122} or templates!={"above_inside":126,"inside_above":152}: raise GuidedEditError("frozen support changed")
 if not all(0<g<=.8 for g in gains): raise GuidedEditError("gain range changed")
 if derive_price()!={"physical_model_forwards":5,"example_evaluations":930,"causal_installations":834,
  "maximum_patch_chunk_rows":256,"patch_chunks":4,"backwards":0,"parameter_updates":0}: raise GuidedEditError("price changed")
def compile_plan():
 validate_preflight(); gains=[TARGET_MAGNITUDE/abs(x["fixed_reader_q"]) for x in _prediction_rows()]
 return {"schema":"task14_fixed_reader_guided_margin_edit_plan_v1","candidate_id":"subject_verb.number_agreement.fixed_reader_guided_margin_edit_v1",
  "data_status":"NEW_FRACTIONAL_GAIN_INTERVENTIONS","support_count":278,"support_floor":SUPPORT_FLOOR,
  "target_magnitude":TARGET_MAGNITUDE,"gain_range":[min(gains),max(gains)],"methods":list(METHODS),
  "prior_art_sha256":PRIOR_ART_SHA256,"sealed_prediction_sha256":PREDICTION_SHA256,"parent_result_sha256":PARENT_RESULT_SHA256,
  "gain_rule":"desired=0.04*sign(q_hat); alpha=desired/q_hat","predictions":dict(zip(PRED_KEYS,
   ("hashes, support, gains, closures pass","guided edits hit overall target bars","both directions hit target",
    "both templates hit target","guided beats fixed alpha=.5","no fractional outcome or fitted calibration enters gain"))),
  "bars":dict(BARS),"price":derive_price()}
def _compile(tokens,heads,rows,torch):
 row_index={r["row_id"]:i for i,r in enumerate(rows)}; idx=[]; repl=[]; specs=[]
 for item in _prediction_rows():
  i=row_index[item["row_id"]]; s=item["background"]; desired=TARGET_MAGNITUDE if item["fixed_reader_q"]>0 else -TARGET_MAGNITUDE; alpha=desired/item["fixed_reader_q"]
  for method in METHODS:
   gain_value={"base":0.0,"guided":alpha,"half":.5}[method]; idx.append(i)
   repl.append(heads[(s,"base")][i]+gain_value*(heads[(s,"exact")][i]-heads[(s,"base")][i]))
   specs.append((i,s,method,item["direction"],item["template"],desired,alpha))
 index=torch.tensor(idx,dtype=torch.long,device=tokens.device); return {"tokens":tokens[:len(rows)][index],
  "finals":torch.full_like(index,tangent.parent.SUBJECT_POSITION),"replacement_heads":torch.stack(repl),
  "native_reinstall_mask":torch.zeros(len(specs),dtype=torch.bool,device=tokens.device),"specs":specs}
def evaluate(model,torch,F,facade):
 rows=authority.build_rows(); n=len(rows); parent=tangent.parent; device=next(model.parameters()).device
 tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
 _,captured,projection,role_closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade)
 roles={"recipient":tangent._role_slice(captured,0,n),"opposite":tangent._role_slice(captured,n,2*n)}; ir={"recipient":tangent._role_slice(inputs,0,n),"opposite":tangent._role_slice(inputs,n,2*n)}
 fn=tangent._head_function(model,roles["recipient"],roles["opposite"],model.transformer.h[parent.LAYER].attn,projection,torch,F)
 with torch.no_grad():
  needed={x["background"] for x in _prediction_rows()}; heads={}
  for s in needed:
   heads[(s,"base")]=fn(gate._raw_for(ir["recipient"],ir["opposite"],s,F)).detach(); heads[(s,"exact")]=fn(gate._raw_for(ir["recipient"],ir["opposite"],s+"YZ",F)).detach()
  patch=_compile(tokens,heads,rows,torch); margins={}; closures=[]
  for start in range(0,len(patch["specs"]),PATCH_CHUNK_ROWS):
   stop=min(start+PATCH_CHUNK_ROWS,len(patch["specs"])); logits,_,_,closure=parent.downstream._decomposed_forward(model,patch["tokens"][start:stop],patch["finals"][start:stop],torch,F,facade,
    replacement_heads=patch["replacement_heads"][start:stop],native_reinstall_mask=patch["native_reinstall_mask"][start:stop]); closures.append(closure)
   for local,spec in enumerate(patch["specs"][start:stop]):
    i,s,method,*_=spec; e=rows[i]["endpoints"]["opposite_same_lemma"]
    margins[(i,s,method)]=float(logits[local,parent.SUBJECT_POSITION,e["answer_id"]]-logits[local,parent.SUBJECT_POSITION,e["foil_id"]])
 evidence=[]
 for i,s,method,d,t,desired,alpha in patch["specs"]:
  if method!="guided": continue
  base=margins[(i,s,"base")]; evidence.append({"row_id":rows[i]["row_id"],"direction":d,"template":t,"background":s,
   "desired_q":desired,"alpha":alpha,"guided_q":margins[(i,s,"guided")]-base,"half_q":margins[(i,s,"half")]-base})
 exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],"role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],
  "downstream_state_closure_max_absolute_error":max(x["state_sum_max_absolute_error"] for x in closures),"downstream_normalized_closure_max_absolute_error":max(x["normalized_state_max_absolute_error"] for x in closures)}
 return evidence,exactness
def _metrics(items,field):
 errors=[abs(x[field]-x["desired_q"]) for x in items]; ordered=sorted(errors); p90=ordered[math.ceil(.9*len(ordered))-1]
 return {"count":len(items),"sign_agreement":sum((x[field]>0)==(x["desired_q"]>0) for x in items)/len(items),
  "mean_absolute_target_error":statistics.fmean(errors),"median_absolute_target_error":statistics.median(errors),"p90_absolute_target_error":p90}
def score(evidence,exactness,bars=BARS):
 guided=_metrics(evidence,"guided_q"); half=_metrics(evidence,"half_q"); dirs={d:_metrics([x for x in evidence if x["direction"]==d],"guided_q") for d in ("plural_to_singular","singular_to_plural")}; temps={t:_metrics([x for x in evidence if x["template"]==t],"guided_q") for t in ("above_inside","inside_above")}
 reduction=1-guided["mean_absolute_target_error"]/max(half["mean_absolute_target_error"],1e-30); gains=[x["alpha"] for x in evidence]
 instrument=len(evidence)==278 and len({(x["row_id"],x["background"]) for x in evidence})==278 and all(x<=bars["maximum_numerical_absolute_error"] for x in exactness.values()) and all(0<x<=.8 for x in gains)
 b=guided["sign_agreement"]>=bars["minimum_overall_sign_agreement"] and guided["median_absolute_target_error"]<=bars["maximum_median_absolute_target_error"] and guided["p90_absolute_target_error"]<=bars["maximum_p90_absolute_target_error"]
 cell=lambda x:x["sign_agreement"]>=bars["minimum_cell_sign_agreement"] and x["median_absolute_target_error"]<=bars["maximum_cell_median_absolute_target_error"]
 predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and b),bool(instrument and all(map(cell,dirs.values()))),bool(instrument and all(map(cell,temps.values()))),bool(instrument and reduction>=bars["minimum_mae_reduction_over_half_gain"]),True)))
 return {**exactness,"guided":guided,"fixed_half_control":half,"by_direction":dirs,"by_template":temps,"mae_reduction_over_half_gain":reduction,"gain_range":[min(gains),max(gains)],"predictions":predictions}
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise GuidedEditError(f"refusing overwrite {OUT}")
 torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 evidence,exactness=evaluate(model,torch,F,facade); scored=score(evidence,exactness); terminal="valid_causal_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_fixed_reader_guided_margin_edit_result_v1","candidate_id":plan["candidate_id"],"terminal":terminal,"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":evidence})
 print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
