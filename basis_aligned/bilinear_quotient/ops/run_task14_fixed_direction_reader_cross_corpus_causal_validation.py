#!/usr/bin/env python3
"""Score sealed fixed-reader predictions against the new-corpus causal lattice."""

# BQGATE: EXPERIMENT pred_a_instrument_and_temporal_seal pred_b_fixed_readers_transfer pred_c_intermediate_composition_transfers pred_d_each_new_template_transfers pred_e_direction_choice_is_necessary pred_f_no_target_tail_or_calibration
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path

import circuit_fast_screen_candidate_task14_fixed_reader_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fixed_reader_transfer_native_capability as capability
import run_task14_prospective_mlp6_7_single_reader_lattice_causal_validation as lattice

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_fixed_direction_reader_cross_corpus_transfer_v1.json"
PREDICTION=ROOT/"circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_predictions.json"
OUT=ROOT/"circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_result.json"
PRIOR_ART_SHA256="7d9fbc286d00d8ca9c5dca4000a649dfade22be14186c24801da98b1b2d1377b"
PREDICTION_SHA256="1d6a5ce082efb6f59b492b5d80c690979e1d3df1bd532d73edf49caaefc8cc81"
CAPABILITY_RESULT_SHA256="b3fdf19c8b9433d954ae2907472b88735769fa04217a7c326428c0ab990c04ef"
CAPABILITY_LICENSE_SHA256="a8b4c7c78456dac6cb3d5631a3e2ba9b548906a314ecb1e76b4815f3f800f783"
SUBSETS=lattice.SUBSETS
BARS={"maximum_numerical_absolute_error":5e-5,"minimum_overall_cosine":.90,
 "maximum_overall_relative_l2_error":.45,"minimum_overall_sign_agreement":.85,
 "minimum_intermediate_cosine":.90,"maximum_intermediate_relative_l2_error":.45,
 "minimum_intermediate_sign_agreement":.85,"minimum_template_cosine":.80,
 "maximum_template_relative_l2_error":.60,"minimum_template_sign_agreement":.75,
 "minimum_sse_reduction_over_swapped":.50}
PRED_KEYS=("pred_a_instrument_and_temporal_seal","pred_b_fixed_readers_transfer",
 "pred_c_intermediate_composition_transfers","pred_d_each_new_template_transfers",
 "pred_e_direction_choice_is_necessary","pred_f_no_target_tail_or_calibration",)
class FixedReaderValidationError(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price(): return {"physical_model_forwards":5,"example_evaluations":1120,
 "causal_interventions":1024,"backwards":0,"parameter_updates":0,"maximum_patch_chunk_rows":256,"patch_chunks":4}
def _load_prediction():
 if _sha256(PREDICTION)!=PREDICTION_SHA256: raise FixedReaderValidationError("prediction changed")
 p=json.loads(PREDICTION.read_text())
 if p.get("terminal")!="sealed_prediction" or p.get("causal_outcomes_opened") is not False or p.get("target_tail_backwards")!=0 or not all(p.get("predictions",{}).values()): raise FixedReaderValidationError("invalid prediction seal")
 return p
def validate_preflight():
 for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),(capability.RESULT,CAPABILITY_RESULT_SHA256,"capability result"),(capability.LICENSE,CAPABILITY_LICENSE_SHA256,"capability license")):
  if _sha256(path)!=expected: raise FixedReaderValidationError(f"{label} changed")
 licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,
  expected_license_sha256=CAPABILITY_LICENSE_SHA256,causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
 _load_prediction()
 if derive_price()!={"physical_model_forwards":5,"example_evaluations":1120,"causal_interventions":1024,
  "backwards":0,"parameter_updates":0,"maximum_patch_chunk_rows":256,"patch_chunks":4}: raise FixedReaderValidationError("price changed")
def compile_plan():
 p=_load_prediction(); validate_preflight(); return {"schema":"task14_fixed_direction_reader_cross_corpus_validation_plan_v1",
  "candidate_id":authority.CAUSAL_CANDIDATE_ID,"split":"NEW_CORPUS_COMPLETE_CAUSAL_LATTICE","row_count":32,
  "background_subsets":list(SUBSETS),"sealed_prediction_sha256":PREDICTION_SHA256,
  "sealed_prediction_created_utc":p["created_utc"],"prior_art_sha256":PRIOR_ART_SHA256,
  "capability_license_sha256":CAPABILITY_LICENSE_SHA256,"literal_scorer":"no scale, offset, endpoint, or causal-q fit",
  "predictions":dict(zip(PRED_KEYS,("hashes, seal, lattice, closures pass","fixed readers meet overall bars",
   "fixed readers meet intermediate bars","each new template meets bars","correct readers beat swapped readers",
   "sealed predictor used zero target-tail backward and no calibration"))),"bars":dict(BARS),"price":derive_price()}
def evaluate(model,torch,F,facade):
 old=lattice.authority
 try:
  lattice.authority=authority; return lattice.evaluate(model,torch,F,facade)
 finally: lattice.authority=old
def _stats(items,field):
 a=[x["actual_q"] for x in items]; p=[x[field] for x in items]; dot=sum(x*y for x,y in zip(a,p)); an=math.sqrt(sum(x*x for x in a)); pn=math.sqrt(sum(x*x for x in p))
 return {"count":len(items),"cosine":dot/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),
  "sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(items),"sse":sum((x-y)**2 for x,y in zip(a,p))}
def score(causal,exactness,bars=BARS):
 p=_load_prediction(); pred={(x["row_id"],x["background"]):x for x in p["evidence"]}; items=[]
 for x in causal:
  q=pred.get((x["row_id"],x["background"]))
  if q is None: raise FixedReaderValidationError("missing sealed value")
  items.append({**x,"fixed_reader_q":q["fixed_reader_q"],"swapped_reader_q":q["swapped_reader_q"]})
 overall={k:_stats(items,k+"_q") for k in ("fixed_reader","swapped_reader")}
 inter=_stats([x for x in items if x["background"] not in {"","EAUW"}],"fixed_reader_q")
 templates={t:_stats([x for x in items if x["template"]==t],"fixed_reader_q") for t in ("inside_above","above_inside")}
 reduction=1-overall["fixed_reader"]["sse"]/max(overall["swapped_reader"]["sse"],1e-30); f=overall["fixed_reader"]
 instrument=len(items)==512 and len({(x["row_id"],x["background"]) for x in items})==512 and all(x<=bars["maximum_numerical_absolute_error"] for x in exactness.values())
 b=f["cosine"]>=bars["minimum_overall_cosine"] and f["relative_l2_error"]<=bars["maximum_overall_relative_l2_error"] and f["sign_agreement"]>=bars["minimum_overall_sign_agreement"]
 c=inter["cosine"]>=bars["minimum_intermediate_cosine"] and inter["relative_l2_error"]<=bars["maximum_intermediate_relative_l2_error"] and inter["sign_agreement"]>=bars["minimum_intermediate_sign_agreement"]
 d=all(x["cosine"]>=bars["minimum_template_cosine"] and x["relative_l2_error"]<=bars["maximum_template_relative_l2_error"] and x["sign_agreement"]>=bars["minimum_template_sign_agreement"] for x in templates.values())
 predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and b),bool(instrument and c),bool(instrument and d),
  bool(instrument and reduction>=bars["minimum_sse_reduction_over_swapped"]),bool(instrument and p["target_tail_backwards"]==0 and p["plan"]["target_tail_forwards"]==0))))
 return {**exactness,"overall":overall,"intermediate_only":inter,"by_template":templates,
  "sse_reduction_over_swapped_reader":reduction,"predictions":predictions,"joined_evidence":items}
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise FixedReaderValidationError(f"refusing overwrite {OUT}")
 torch,F,facade=lattice.tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 causal,exactness=evaluate(model,torch,F,facade); scored=score(causal,exactness); terminal="valid_causal_screen" if scored["predictions"][PRED_KEYS[0]] else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_fixed_direction_reader_cross_corpus_validation_result_v1",
  "candidate_id":authority.CAUSAL_CANDIDATE_ID,"terminal":terminal,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"causal_evidence":causal,"sealed_prediction_sha256":PREDICTION_SHA256})
 print(json.dumps({"terminal":terminal,"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
