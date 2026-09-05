#!/usr/bin/env python3
"""Width-corrected export of two projected-write direction readers."""

# BQGATE: EXPERIMENT pred_a_gradient_instrument pred_b_two_readers_exported pred_c_direction_readers_are_distinct
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_fixed_direction_reader_artifact as v1

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_mlp6_7_fixed_direction_reader_artifact_v2.json"
OUT=ROOT/"circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json"
PRIOR_ART_SHA256="38affa2a61fb9c3dd2b4ad8fa8a26d6ae6fe1391e06c12f33322ae04b8217985"
INVALID_V1_SHA256="a20bfcdc9bcb274e500e00585735d1fe69afd5e1c3185c352ff25b788d85b6b9"
READER_WIDTH=1152
PRED_KEYS=("pred_a_gradient_instrument","pred_b_two_readers_exported","pred_c_direction_readers_are_distinct",)
class FixedReaderV2Error(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price(): return {"physical_model_forwards":2,"example_evaluations":128,"backwards":1,
 "causal_interventions":0,"parameter_updates":0,"stored_scalars":2304}

def validate_preflight():
 if _sha256(PRIOR_ART)!=PRIOR_ART_SHA256: raise FixedReaderV2Error("v2 prior art changed")
 if _sha256(v1.OUT)!=INVALID_V1_SHA256: raise FixedReaderV2Error("invalid v1 receipt changed")
 v1.validate_preflight()
 if derive_price()!={"physical_model_forwards":2,"example_evaluations":128,"backwards":1,
  "causal_interventions":0,"parameter_updates":0,"stored_scalars":2304}: raise FixedReaderV2Error("price changed")

def compile_plan():
 validate_preflight(); return {"schema":"task14_mlp6_7_fixed_direction_reader_artifact_plan_v2",
  "candidate_id":"subject_verb.number_agreement.mlp6_7_fixed_direction_reader_artifact_v2",
  "repair":"reader width 128->1152 only","reader_width":READER_WIDTH,
  "directions":["plural_to_singular","singular_to_plural"],"prior_art_sha256":PRIOR_ART_SHA256,
  "invalid_v1_sha256":INVALID_V1_SHA256,"predictions":dict(zip(PRED_KEYS,
   ("closures and 32 gradients valid","exactly two finite 1152-coordinate vectors exported",
    "signed direction-reader cosine <= -0.50, unchanged and expected to miss"))),"price":derive_price()}

def rescore(readers,exactness,geometry):
 instrument=all(x<=v1.MAXIMUM_ERROR for x in exactness.values()) and geometry["all_gradient_l2_norm"]>0
 exported=len(readers)==2 and all(len(x["coordinates"])==READER_WIDTH and x["l2_norm"]>0 for x in readers.values())
 return dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and exported),
  bool(instrument and geometry["inter_direction_cosine"]<=-.50))))

def main(argv=None):
 parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise FixedReaderV2Error(f"refusing to overwrite {OUT}")
 torch,F,facade=v1.tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 readers,exactness,geometry,_=v1.evaluate(model,torch,F,facade); predictions=rescore(readers,exactness,geometry)
 terminal="reader_artifact" if predictions[PRED_KEYS[0]] and predictions[PRED_KEYS[1]] else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_fixed_direction_reader_artifact_result_v2",
  "candidate_id":plan["candidate_id"],"terminal":terminal,"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,
  "exactness":exactness,"reader_geometry":geometry,"predictions":predictions,"readers":readers,"causal_outcomes_opened":False})
 print(json.dumps({"terminal":terminal,"predictions":predictions,"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
