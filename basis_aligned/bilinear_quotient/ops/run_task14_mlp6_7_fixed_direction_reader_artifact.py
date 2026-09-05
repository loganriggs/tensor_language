#!/usr/bin/env python3
"""Freeze two direction-conditioned downstream reader vectors for prospective transfer."""

# BQGATE: EXPERIMENT pred_a_gradient_instrument pred_b_two_readers_exported pred_c_direction_readers_are_distinct
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate

ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_mlp6_7_fixed_direction_reader_artifact_v1.json"
PARENT_RESULT=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_shared_reader_transfer_v1_result.json"
OUT=ROOT/"circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v1_result.json"
PRIOR_ART_SHA256="990bd1c907e83986977ada647f2f3a5d26e95d85ff46da1d6edabf8bf725633d"
PARENT_RESULT_SHA256="c45ed6a7874a778ce7b29627b8a7ecd5562dc7e7574254bf91bcc710d18a4d46"
SUBSETS=gate.BACKGROUND_SUBSETS; MAXIMUM_ERROR=5e-5
PRED_KEYS=("pred_a_gradient_instrument","pred_b_two_readers_exported","pred_c_direction_readers_are_distinct",)
class FixedReaderError(ValueError): pass
def _sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price(): return {"physical_model_forwards":2,"example_evaluations":128,"backwards":1,
 "causal_interventions":0,"parameter_updates":0,"stored_scalars":256}

def validate_preflight():
 if _sha256(PRIOR_ART)!=PRIOR_ART_SHA256: raise FixedReaderError("prior art changed")
 if _sha256(PARENT_RESULT)!=PARENT_RESULT_SHA256: raise FixedReaderError("parent changed")
 p=json.loads(PARENT_RESULT.read_text())
 if p.get("terminal")!="valid_diagnostic" or p.get("score",{}).get("predictions",{}).get("pred_b_direction_reader_transfers") is not True:
  raise FixedReaderError("parent does not license reader export")
 if derive_price()!={"physical_model_forwards":2,"example_evaluations":128,"backwards":1,
  "causal_interventions":0,"parameter_updates":0,"stored_scalars":256}: raise FixedReaderError("price changed")

def compile_plan():
 validate_preflight(); return {"schema":"task14_mlp6_7_fixed_direction_reader_artifact_plan_v1",
  "candidate_id":"subject_verb.number_agreement.mlp6_7_fixed_direction_reader_artifact_v1",
  "row_count":32,"reader_width":128,"directions":["plural_to_singular","singular_to_plural"],
  "source":"arithmetic mean of target-free row-central answer-margin gradients",
  "prior_art_sha256":PRIOR_ART_SHA256,"parent_result_sha256":PARENT_RESULT_SHA256,
  "predictions":dict(zip(PRED_KEYS,("closures and 32 gradients valid","exactly two finite 128-coordinate vectors exported",
   "signed direction-reader cosine <= -0.50"))),"price":derive_price()}

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
  heads=[fn(gate._raw_for(ir["recipient"],ir["opposite"],s+suffix,F)).detach()
   for s in SUBSETS for suffix in ("","YZ")]
  center=torch.stack(heads).mean(0)
 replacement=center.detach().clone().requires_grad_(True)
 logits,_,_,closure=parent.downstream._decomposed_forward(model,tokens[:n],torch.full((n,),parent.SUBJECT_POSITION,
  dtype=torch.long,device=device),torch,F,facade,replacement_heads=replacement,
  native_reinstall_mask=torch.zeros(n,dtype=torch.bool,device=device))
 gradients=torch.autograd.grad(_margins(logits,rows,torch).sum(),replacement)[0].detach().float()
 readers={d:gradients[[i for i,r in enumerate(rows) if r["direction_id"]==d]].mean(0)
  for d in ("plural_to_singular","singular_to_plural")}
 a,b=readers.values(); cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30))
 exactness={"role_state_closure_max_absolute_error":role_closure["input_state_closure_max_absolute_error"],
  "role_normalized_closure_max_absolute_error":role_closure["input_normalized_closure_max_absolute_error"],
  "downstream_state_closure_max_absolute_error":closure["state_sum_max_absolute_error"],
  "downstream_normalized_closure_max_absolute_error":closure["normalized_state_max_absolute_error"]}
 instrument=all(x<=MAXIMUM_ERROR for x in exactness.values()) and bool(torch.isfinite(gradients).all()) and int((gradients.norm(dim=-1)>0).sum())==32
 exported={d:{"coordinates":v.cpu().tolist(),"l2_norm":float(v.norm())} for d,v in readers.items()}
 predictions=dict(zip(PRED_KEYS,(bool(instrument),bool(instrument and len(exported)==2 and all(len(v["coordinates"])==128 for v in exported.values())),
  bool(instrument and cosine<=-.50))))
 return exported,exactness,{"all_gradient_l2_norm":float(gradients.norm()),"inter_direction_cosine":cosine},predictions

def main(argv=None):
 parser=argparse.ArgumentParser(); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv); plan=compile_plan()
 if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True)); return
 if OUT.exists(): raise FixedReaderError(f"refusing to overwrite {OUT}")
 torch,F,facade=tangent.parent.factors._dependencies(); model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 readers,exactness,geometry,predictions=evaluate(model,torch,F,facade); terminal="reader_artifact" if predictions[PRED_KEYS[0]] and predictions[PRED_KEYS[1]] else "invalid"
 payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_fixed_direction_reader_artifact_result_v1",
  "candidate_id":plan["candidate_id"],"terminal":terminal,"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,
  "exactness":exactness,"reader_geometry":geometry,"predictions":predictions,"readers":readers,"causal_outcomes_opened":False})
 print(json.dumps({"terminal":terminal,"predictions":predictions,"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
