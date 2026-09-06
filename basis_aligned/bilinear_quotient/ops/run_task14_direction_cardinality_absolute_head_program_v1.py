#!/usr/bin/env python3
"""Export and validate ten absolute Task14 L11H3 head terms."""

# BQGATE: EXPERIMENT pred_a_immutable_export_and_instrument pred_b_absolute_head_substitutes_exact_effect pred_c_each_direction_template_and_cardinality_recurs pred_d_native_l11h3_base_dependency_removed_narrowly pred_e_fixed_price
from __future__ import annotations
import hashlib,json,math,os,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_candidate_task14_fixed_reader_transfer as train_authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as validation
import run_task14_mlp6_7_direction_cardinality_prototype_export as exporter
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/task14_direction_cardinality_absolute_head_program_v1.json";DISP=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json";REFERENCE=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json";ARTIFACT_OUT=ROOT/"circuits/followups/task14_direction_cardinality_absolute_head_program_v1_artifact.json";OUT=ROOT/"circuits/followups/task14_direction_cardinality_absolute_head_program_v1_result.json"
CANDIDATE_ID="subject_verb.number_agreement.direction_cardinality_absolute_head_program_v1";SUBSETS=factor_gate.BACKGROUND_SUBSETS;CHUNK=256
EXPECTED={PRIOR:"e3c0418426bef3b2f6ab1f415cf096462641a921af7678372b1610039b13c721",DISP:"cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",REFERENCE:"9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0",Path(exporter.__file__):"8c857eb00dc55f33bd66657c33943e3fdc68dd8186c046dda3dd353f05572cfe",Path(validation.__file__):"8b4c4c645cf333f26cf3a81669d36ca5d952c21704aa637089bae98adfa849a4"}
BARS={"maximum_closure_error":1e-4,"minimum_cosine":.75,"maximum_relative_l2":.75,"minimum_norm_ratio":.25,"maximum_norm_ratio":2.,"minimum_sign_agreement":.75,"minimum_cell_cosine":.65,"minimum_cell_sign_agreement":.65,"minimum_cardinality_cosine":.65,"minimum_cardinality_sign_agreement":.65}
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 disp,reference=json.loads(DISP.read_text()),json.loads(REFERENCE.read_text())
 if disp.get("terminal")!="prototype_artifact" or reference.get("terminal")!="valid_causal_screen" or not all(reference["score"]["predictions"].values()):raise ValueError("parent program invalid")
 return reference
def compile_plan():
 _load();return {"schema":"task14_direction_cardinality_absolute_head_program_plan_v1","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"displacement_artifact_sha256":EXPECTED[DISP],"reference_validation_sha256":EXPECTED[REFERENCE],"export":{"split":"SECOND_CORPUS_TRAINING_ONLY","rows":32,"role_examples":96,"vectors":10,"width":1152},"validation":{"split":"PROSPECTIVE_THIRD_CORPUS_COMPLETE_CAUSAL_LATTICE","rows":32,"backgrounds":16,"methods":["base","absolute"],"installations":1024},"bars":dict(BARS),"price":{"physical_model_forwards":6,"example_evaluations":1216,"causal_installations":1024,"stored_fp32_scalars":11520,"backwards":0,"fits":0,"parameter_updates":0}}
def _context(model,rows,torch,F,facade):
 parent=tangent.parent;count=len(rows);device=next(model.parameters()).device;tokens,finals=parent.downstream.depth.parent.v1._role_batch(rows,torch,device);_,captured,projection,closure,inputs=parent._decomposed_forward(model,tokens,finals,torch,F,facade);roles={"recipient":tangent._role_slice(captured,0,count),"opposite":tangent._role_slice(captured,count,2*count)};input_roles={"recipient":tangent._role_slice(inputs,0,count),"opposite":tangent._role_slice(inputs,count,2*count)};function=tangent._head_function(model,roles["recipient"],roles["opposite"],model.transformer.h[parent.LAYER].attn,projection,torch,F);return parent,tokens,function,input_roles,closure
def export_absolute(model,torch,F,facade):
 rows=train_authority.build_rows();_,_,function,inputs,closure=_context(model,rows,torch,F,facade);groups=defaultdict(list)
 with torch.no_grad():
  for subset in SUBSETS:
   exact_head=function(factor_gate._raw_for(inputs["recipient"],inputs["opposite"],subset+"YZ",F)).detach()
   for i,row in enumerate(rows):groups[(row["direction_id"],len(subset))].append(exact_head[i])
 prototypes={}
 for (direction,cardinality),values in groups.items():
  expected=16*math.comb(4,cardinality)
  if len(values)!=expected:raise ValueError("export support changed")
  vector=torch.stack(values).mean(0);prototypes[f"{direction}.cardinality_{cardinality}"]={"direction":direction,"cardinality":cardinality,"support":len(values),"coordinates":vector.cpu().tolist(),"l2_norm":float(vector.norm())}
 return prototypes,closure
def evaluate(model,prototypes,torch,F,facade):
 rows=validation.authority.build_rows();parent,tokens,function,inputs,source_closure=_context(model,rows,torch,F,facade);device=tokens.device;vectors={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=device) for k,v in prototypes.items()};indices=[];replacements=[];specs=[]
 with torch.no_grad():
  for subset in SUBSETS:
   base=function(factor_gate._raw_for(inputs["recipient"],inputs["opposite"],subset,F)).detach()
   for i,row in enumerate(rows):
    for method,value in (("base",base[i]),("absolute",vectors[f'{row["direction_id"]}.cardinality_{len(subset)}'])):indices.append(i);replacements.append(value);specs.append((i,subset,method))
  index=torch.tensor(indices,dtype=torch.long,device=device);patch_tokens=tokens[:len(rows)][index];finals=torch.full_like(index,parent.SUBJECT_POSITION);replacement=torch.stack(replacements);margins={};closures=[]
  for start in range(0,len(specs),CHUNK):
   stop=min(start+CHUNK,len(specs));logits,_,_,closure=parent.downstream._decomposed_forward(model,patch_tokens[start:stop],finals[start:stop],torch,F,facade,replacement_heads=replacement[start:stop],native_reinstall_mask=torch.zeros(stop-start,dtype=torch.bool,device=device));closures.append(closure)
   for local,(i,subset,method) in enumerate(specs[start:stop]):
    endpoint=rows[i]["endpoints"]["opposite_same_lemma"];margins[(i,subset,method)]=float(logits[local,parent.SUBJECT_POSITION,endpoint["answer_id"]]-logits[local,parent.SUBJECT_POSITION,endpoint["foil_id"]])
 reference={(r["row_id"],r["background"]):r for r in _load()["score"]["joined_evidence"]};evidence=[]
 for i,row in enumerate(rows):
  for subset in SUBSETS:
   ref=reference[(row["row_id"],subset)];evidence.append({"row_id":row["row_id"],"direction":row["direction_id"],"template":row["template_id"],"background":subset,"cardinality":len(subset),"native_exact_q":ref["native_exact_q"],"absolute_program_q":margins[(i,subset,"absolute")]-margins[(i,subset,"base")]})
 return evidence,source_closure,closures
def _metrics(rows):
 a=[r["native_exact_q"] for r in rows];p=[r["absolute_program_q"] for r in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/(an*pn),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/an,"predicted_to_actual_norm_ratio":pn/an,"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(rows)}
def score(evidence,export_closure,source_closure,closures):
 errors=[export_closure["input_state_closure_max_absolute_error"],export_closure["input_normalized_closure_max_absolute_error"],source_closure["input_state_closure_max_absolute_error"],source_closure["input_normalized_closure_max_absolute_error"]]+[v for c in closures for k,v in c.items() if k in ("state_sum_max_absolute_error","normalized_state_max_absolute_error")];overall=_metrics(evidence);cells={f"{d}.{t}":_metrics([r for r in evidence if r["direction"]==d and r["template"]==t]) for d in ("plural_to_singular","singular_to_plural") for t in sorted({r["template"] for r in evidence})};cards={str(c):_metrics([r for r in evidence if r["cardinality"]==c]) for c in range(5)};instrument=len(evidence)==512 and max(errors)<=BARS["maximum_closure_error"];transfer=overall["cosine"]>=BARS["minimum_cosine"] and overall["relative_l2_error"]<=BARS["maximum_relative_l2"] and BARS["minimum_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_sign_agreement"];recurrence=all(v["cosine"]>=BARS["minimum_cell_cosine"] and v["sign_agreement"]>=BARS["minimum_cell_sign_agreement"] for v in cells.values()) and all(v["cosine"]>=BARS["minimum_cardinality_cosine"] and v["sign_agreement"]>=BARS["minimum_cardinality_sign_agreement"] for v in cards.values());pred={"pred_a_immutable_export_and_instrument":instrument,"pred_b_absolute_head_substitutes_exact_effect":instrument and transfer,"pred_c_each_direction_template_and_cardinality_recurs":instrument and recurrence,"pred_d_native_l11h3_base_dependency_removed_narrowly":True,"pred_e_fixed_price":compile_plan()["price"]=={"physical_model_forwards":6,"example_evaluations":1216,"causal_installations":1024,"stored_fp32_scalars":11520,"backwards":0,"fits":0,"parameter_updates":0}};return {"closure_max_absolute_error":max(errors),"overall":overall,"by_direction_template":cells,"by_cardinality":cards,"dependency_removed":"third-corpus native L11H3 base head is not read when constructing edited replacement","dependencies_retained":["all other upstream context","native downstream suffix"],"predictions":pred,"terminal":"program_screen" if all(pred.values()) else "null" if instrument else "invalid"}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if ARTIFACT_OUT.exists() or OUT.exists():raise ValueError("refusing overwrite")
 torch,F,facade=tangent.parent.factors._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():prototypes,export_closure=export_absolute(model,torch,F,facade);evidence,source_closure,closures=evaluate(model,prototypes,torch,F,facade)
 artifact={"schema":"task14_direction_cardinality_absolute_head_program_artifact_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"prototypes":prototypes,"terminal":"prototype_artifact"};artifact_bytes=managed.atomic_create_json(ARTIFACT_OUT,artifact);scored=score(evidence,export_closure,source_closure,closures);payload=managed.atomic_create_json(OUT,{"schema":"task14_direction_cardinality_absolute_head_program_result_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"artifact_sha256":hashlib.sha256(artifact_bytes).hexdigest(),"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":evidence,"terminal":scored["terminal"]});print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"artifact_sha256":hashlib.sha256(artifact_bytes).hexdigest(),"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
