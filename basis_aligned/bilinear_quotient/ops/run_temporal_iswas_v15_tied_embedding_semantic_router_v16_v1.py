#!/usr/bin/env python3
"""Frozen tied-embedding semantic router from v15 cues to sealed v16."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_alignment_direction_invariance_finiteness_and_price pred_b_v15_crossfit_router_is_selective pred_c_v16_semantic_router_generalizes pred_d_both_v16_target_constructions_are_covered pred_e_frozen_folds_agree_on_v16
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
from circuit_fast_screen_managed_runner import atomic_create_json
import fastload
import tied_embedding_semantic_router_contract as semantic
import unordered_token_pair_router_contract as tokens

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/temporal_iswas_v15_tied_embedding_semantic_router_v16_v1.json"
LOOKUP_OOD=ROOT/"circuits/followups/temporal_iswas_v15_token_router_v16_signature_coverage_audit_v1_result.json"
GAIN=ROOT/"circuits/followups/temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1_result.json"
V16_CAPABILITY=ROOT/"circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
OUT=ROOT/"circuits/followups/temporal_iswas_v15_tied_embedding_semantic_router_v16_v1_result.json"
CANDIDATE_ID="temporal_auxiliary.iswas_v15_tied_embedding_semantic_router_v16_v1"
EXPECTED={
 "prior":"89b493dfe90561fdd2bb36949bf496ed8819487022364e112f9cc15b8a664dc9",
 "lookup_ood":"92abdb1d7f17ee253288c5dd92cc2369dd3950e21a582b00c95d0b04816a8054",
 "gain":"d1e54e50d09a0767be525382cc6f825e5ff18596ae1e66fc817006c79a0bc31f",
 "v16_capability":"a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
 "v15":"7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
 "v16":"5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
 "semantic":"fa3080d692416f7bbe8498ebba9461002c402f4f556a868fd78f1e24d6aaad9b",
 "tokens":"061f471d690c7549794af203e0d4ceefbbaa9f57fe747611388dcd676e85dbb9",
 "fastload":"5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
 "checkpoint":"680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
FILES={"prior":PRIOR,"lookup_ood":LOOKUP_OOD,"gain":GAIN,"v16_capability":V16_CAPABILITY,
 "v15":ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
 "v16":ROOT/"ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py",
 "semantic":ROOT/"ops/tied_embedding_semantic_router_contract.py","tokens":ROOT/"ops/unordered_token_pair_router_contract.py",
 "fastload":ROOT/"ops/fastload.py"}
PRICE={"checkpoint_loads":1,"model_forwards":0,"transformer_backwards":0,"model_updates":0,"example_evaluations":256,"fit_parameters":6912}
LABEL={"A1":0,"A2":1,"P":2,"C":2}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(value):
 if isinstance(value,dict):return all(finite(item) for item in value.values())
 if isinstance(value,(list,tuple)):return all(finite(item) for item in value)
 return not isinstance(value,float) or math.isfinite(value)
def aligned(row):return len(row["base_ids"])==len(row["donor_ids"]) and row["base_semantic_position"]==row["donor_semantic_position"]
def signature(row):return tokens.signature(row["base_ids"],row["donor_ids"],row["base_semantic_position"])
def report(rows,predictions):
 confusion=[[sum(LABEL[row["transform_id"]]==actual and int(pred)==guess for row,pred in zip(rows,predictions)) for guess in range(3)] for actual in range(3)]
 recalls=[confusion[i][i]/max(1,sum(confusion[i])) for i in range(3)]
 return {"classes":["A1","A2","off"],"confusion":confusion,"recalls":recalls,"macro_accuracy":sum(recalls)/3,
         "control_predicted_target_count":sum(LABEL[row["transform_id"]]==2 and int(pred)!=2 for row,pred in zip(rows,predictions))}

def main():
 observed={n:sha(p) for n,p in FILES.items()};lookup=json.loads(LOOKUP_OOD.read_text());gain=json.loads(GAIN.read_text());cap=json.loads(V16_CAPABILITY.read_text())
 config,checkpoint,_=fastload._paths();authority=observed=={n:EXPECTED[n] for n in observed} and sha(checkpoint)==EXPECTED["checkpoint"] and lookup.get("terminal")=="unseen_signature_default_off" and gain.get("terminal")=="gain_calibrated_swap_null" and cap.get("terminal")=="manifest"
 dry={"candidate_id":CANDIDATE_ID,"dryrun":True,"authority_ok":authority,"gpu_accessed":False,"model_loaded":False,"queue_touched":False,"price":PRICE}
 if not authority:raise RuntimeError(f"authority changed: {observed}")
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(dry,sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 start=time.perf_counter();model=fastload.load_model_fast().eval();import torch
 expected_config={"vocab_size":50304,"n_layer":18,"n_head":9,"n_embd":1152,"squared_mlp":False,"bilinear":True,"expansion_factor":4,"gated":False,"squared_attn":True,"bilinear_attn":True}
 rows15=v15.build_rows();rows16=v16.build_rows();score16=[r for r in rows16 if r["transform_id"] in ("A1","A2","P")];excluded16=[r for r in rows16 if r["transform_id"]=="C"]
 aligned15=all(aligned(r) for r in rows15);aligned16=all(aligned(r) for r in score16);c_excluded=all(not aligned(r) for r in excluded16)
 sig15=[signature(r) for r in rows15];sig16=[signature(r) for r in score16]
 feat15=semantic.features(torch,model.transformer.wte.weight.detach(),sig15);feat16=semantic.features(torch,model.transformer.wte.weight.detach(),sig16)
 reverse=[tuple((b,a) for a,b in item) for item in sig15]
 direction_error=float((feat15-semantic.features(torch,model.transformer.wte.weight.detach(),reverse)).abs().max())
 folds={};v16_predictions=[]
 for held in (0,1):
  train_idx=[i for i,r in enumerate(rows15) if r["group_number"]%2==1-held];held_idx=[i for i,r in enumerate(rows15) if r["group_number"]%2==held]
  labels_train=torch.tensor([LABEL[rows15[i]["transform_id"]] for i in train_idx]);centroids=semantic.fit(torch,feat15[train_idx],labels_train)
  pred15,scores15=semantic.predict(torch,feat15[held_idx],centroids);pred16,scores16=semantic.predict(torch,feat16,centroids);v16_predictions.append(pred16)
  folds[str(held)]={"v15":{**report([rows15[i] for i in held_idx],pred15.tolist()),"predictions":pred15.tolist(),"scores":scores15.tolist()},
                    "v16":{**report(score16,pred16.tolist()),"predictions":pred16.tolist(),"scores":scores16.tolist()},
                    "centroid_norms":[float(x.norm()) for x in centroids]}
 nonoverlap=set(sig15).isdisjoint(set(sig16));disjoint=all(set(rows15[i]["row_id"] for i,r in enumerate(rows15) if r["group_number"]%2==p).isdisjoint(set(rows15[i]["row_id"] for i,r in enumerate(rows15) if r["group_number"]%2==1-p)) for p in (0,1))
 A=bool(authority and config==expected_config and len(rows15)==len(rows16)==64 and len(score16)==48 and len(excluded16)==16 and aligned15 and aligned16 and c_excluded and nonoverlap and disjoint and direction_error<=1e-6 and finite(folds) and PRICE=={"checkpoint_loads":1,"model_forwards":0,"transformer_backwards":0,"model_updates":0,"example_evaluations":256,"fit_parameters":6912})
 B=all(folds[str(h)]["v15"]["macro_accuracy"]==1 and folds[str(h)]["v15"]["control_predicted_target_count"]==0 for h in (0,1))
 C=all(folds[str(h)]["v16"]["macro_accuracy"]>=.75 and folds[str(h)]["v16"]["control_predicted_target_count"]==0 for h in (0,1))
 D=all(folds[str(h)]["v16"]["confusion"][0][0]>=12 and folds[str(h)]["v16"]["confusion"][1][1]>=12 for h in (0,1))
 E=bool(torch.equal(v16_predictions[0],v16_predictions[1]))
 predictions=dict(zip(("pred_a_authority_alignment_direction_invariance_finiteness_and_price","pred_b_v15_crossfit_router_is_selective",
  "pred_c_v16_semantic_router_generalizes","pred_d_both_v16_target_constructions_are_covered","pred_e_frozen_folds_agree_on_v16"),map(bool,(A,B,C,D,E))))
 terminal="invalid" if not A else "embedding_semantic_router_candidate" if all((B,C,D,E)) else "embedding_semantic_router_ood_null" if B else "embedding_router_in_distribution_null"
 result={"schema":"temporal_iswas_v15_tied_embedding_semantic_router_v16_result_v1","candidate_id":CANDIDATE_ID,
  "started_utc":datetime.now(timezone.utc).isoformat(),"serial_seconds":time.perf_counter()-start,"authority_sha256":EXPECTED,
  "folds":folds,"instrument":{"model_config":config,"v15_all_aligned":aligned15,"v16_a1_a2_p_aligned":aligned16,
  "v16_c_explicitly_excluded":c_excluded,"v15_v16_signature_nonoverlap":nonoverlap,"train_held_disjoint":disjoint,"direction_invariance_max_abs_error":direction_error},
  "predictions":predictions,"terminal":terminal,"price":PRICE}
 atomic_create_json(OUT,result);print(json.dumps({k:result[k] for k in ("predictions","terminal","folds","instrument","price")},sort_keys=True))

if __name__=="__main__":main()
