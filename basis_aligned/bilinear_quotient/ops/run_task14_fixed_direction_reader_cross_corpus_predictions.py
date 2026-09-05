#!/usr/bin/env python3
"""Seal new-corpus full-lattice predictions using two already-frozen readers."""

# BQGATE: EXPERIMENT pred_a_capability_and_reader_artifact pred_b_source_instrument pred_c_five_hundred_twelve_predictions_sealed
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path

import circuit_fast_screen_candidate_task14_fixed_reader_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fixed_reader_transfer_native_capability as capability
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_fixed_direction_reader_cross_corpus_transfer_v1.json"
READER=ROOT/"circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json"
OUT=ROOT/"circuits/fast_screens/task14_fixed_direction_reader_cross_corpus_transfer_v1_predictions.json"
PRIOR_ART_SHA256="7d9fbc286d00d8ca9c5dca4000a649dfade22be14186c24801da98b1b2d1377b"
READER_SHA256="9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57"
CAPABILITY_RESULT_SHA256="b3fdf19c8b9433d954ae2907472b88735769fa04217a7c326428c0ab990c04ef"
CAPABILITY_LICENSE_SHA256="a8b4c7c78456dac6cb3d5631a3e2ba9b548906a314ecb1e76b4815f3f800f783"
SUBSETS=gate.BACKGROUND_SUBSETS; MAXIMUM_ERROR=5e-5
PRED_KEYS=("pred_a_capability_and_reader_artifact","pred_b_source_instrument",
 "pred_c_five_hundred_twelve_predictions_sealed",)
class FixedReaderPredictionError(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price(): return {"physical_model_forwards":1,"example_evaluations":96,"backwards":0,
 "causal_interventions":0,"sealed_predictions":512,"parameter_updates":0}

def _load_reader():
 if _sha256(READER)!=READER_SHA256: raise FixedReaderPredictionError("reader artifact changed")
 r=json.loads(READER.read_text())
 if r.get("terminal")!="reader_artifact" or not r.get("predictions",{}).get("pred_b_two_readers_exported"):
  raise FixedReaderPredictionError("reader artifact invalid")
 return r
def validate_preflight():
 for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),(capability.RESULT,CAPABILITY_RESULT_SHA256,"capability result"),(capability.LICENSE,CAPABILITY_LICENSE_SHA256,"capability license")):
  if _sha256(path)!=expected: raise FixedReaderPredictionError(f"{label} changed")
 licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,
  expected_license_sha256=CAPABILITY_LICENSE_SHA256,causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
 _load_reader()
 if derive_price()!={"physical_model_forwards":1,"example_evaluations":96,"backwards":0,
  "causal_interventions":0,"sealed_predictions":512,"parameter_updates":0}: raise FixedReaderPredictionError("price changed")
def compile_plan():
 validate_preflight(); return {"schema":"task14_fixed_direction_reader_cross_corpus_prediction_plan_v1",
  "candidate_id":authority.CAUSAL_CANDIDATE_ID,"split":"NEW_CORPUS_PREDICTIONS_BEFORE_ANY_CAUSAL_OUTCOME",
  "row_count":32,"background_subsets":list(SUBSETS),"reader_artifact_sha256":READER_SHA256,
  "capability_license_sha256":CAPABILITY_LICENSE_SHA256,"prior_art_sha256":PRIOR_ART_SHA256,
  "target_tail_forwards":0,"target_tail_backwards":0,"causal_outcomes_opened":False,
  "predictions":dict(zip(PRED_KEYS,("capability and fixed-reader artifact validate",
   "source decomposition closures <=5e-5","exactly 512 correct-reader and swapped-reader predictions sealed"))),"price":derive_price()}

def evaluate(model,torch,F,facade):
 rows=authority.build_rows(); n=len(rows); parent=tangent.parent; device=next(model.parameters()).device
 tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device)
 _,captured,projection,closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade)
 roles={"recipient":tangent._role_slice(captured,0,n),"opposite":tangent._role_slice(captured,n,2*n)}
 ir={"recipient":tangent._role_slice(inputs,0,n),"opposite":tangent._role_slice(inputs,n,2*n)}
 fn=tangent._head_function(model,roles["recipient"],roles["opposite"],model.transformer.h[parent.LAYER].attn,projection,torch,F)
 artifact=_load_reader(); readers={d:torch.tensor(v["coordinates"],dtype=torch.float32,device=device) for d,v in artifact["readers"].items()}
 evidence=[]
 with torch.no_grad():
  bases={s:fn(gate._raw_for(ir["recipient"],ir["opposite"],s,F)).detach() for s in SUBSETS}
  exacts={s:fn(gate._raw_for(ir["recipient"],ir["opposite"],s+"YZ",F)).detach() for s in SUBSETS}
  for i,row in enumerate(rows):
   direction=row["direction_id"]; opposite="singular_to_plural" if direction=="plural_to_singular" else "plural_to_singular"
   for s in SUBSETS:
    delta=exacts[s][i]-bases[s][i]
    evidence.append({"row_id":row["row_id"],"direction":direction,"template":row["template_id"],
     "background":s,"cardinality":len(s),"fixed_reader_q":float(torch.dot(readers[direction],delta)),
     "swapped_reader_q":float(torch.dot(readers[opposite],delta)),"head_delta_l2_norm":float(delta.norm())})
 exactness={"source_state_closure_max_absolute_error":closure["input_state_closure_max_absolute_error"],
  "source_normalized_closure_max_absolute_error":closure["input_normalized_closure_max_absolute_error"]}
 instrument=all(x<=MAXIMUM_ERROR for x in exactness.values()); unique={(x["row_id"],x["background"]) for x in evidence}
 predictions=dict(zip(PRED_KEYS,(True,bool(instrument),bool(len(evidence)==512 and len(unique)==512))))
 return evidence,exactness,predictions
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--dry-run",action="store_true"); args=p.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise FixedReaderPredictionError(f"refusing overwrite {OUT}")
 torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 evidence,exactness,predictions=evaluate(model,torch,F,facade); terminal="sealed_prediction" if all(predictions.values()) else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_fixed_direction_reader_cross_corpus_predictions_v1",
  "candidate_id":authority.CAUSAL_CANDIDATE_ID,"terminal":terminal,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
  "plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"exactness":exactness,
  "predictions":predictions,"evidence":evidence,"causal_outcomes_opened":False,"target_tail_backwards":0})
 print(json.dumps({"terminal":terminal,"predictions":predictions,"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
