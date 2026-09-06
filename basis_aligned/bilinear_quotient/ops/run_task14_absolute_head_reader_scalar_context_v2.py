#!/usr/bin/env python3
"""One-reader-scalar base-context repair of the failed Task14 absolute program."""

# BQGATE: EXPERIMENT pred_a_immutable_reader_aligned_instrument pred_b_one_scalar_context_restores_effect pred_c_each_direction_template_and_cardinality_recurs pred_d_context_dependency_reduced_to_one_scalar pred_e_fixed_price
from __future__ import annotations
import hashlib,json,math,os,sys
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_managed_runner as managed
import run_task14_direction_cardinality_absolute_head_program_v1 as absolute_run
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as validation
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/task14_absolute_head_reader_scalar_context_v2.json";ABS=ROOT/"circuits/followups/task14_direction_cardinality_absolute_head_program_v1_artifact.json";NULL=ROOT/"circuits/followups/task14_direction_cardinality_absolute_head_program_v1_result.json";DISP=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json";READERS=ROOT/"circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json";REF=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_causal_validation_v1_result.json";OUT=ROOT/"circuits/followups/task14_absolute_head_reader_scalar_context_v2_result.json"
CANDIDATE_ID="subject_verb.number_agreement.absolute_head_reader_scalar_context_v2";SUBSETS=factor_gate.BACKGROUND_SUBSETS;CHUNK=256
EXPECTED={PRIOR:"d1d3e563642d8763f724c062be91781b491c16b658fe91f21571a78acd2f497a",ABS:"f3097e072ba0e5d429f9b3d741665c22832b34367678a8630a9c81d705bcd55b",NULL:"badde085b97cceadd5b1055715597d11bfabd30b621a435951214315ab1bbc03",DISP:"cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",READERS:"9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57",REF:"9a488259efdb85477f35c612696368e8a3e372338a11253615a7b53650c88fe0"}
BARS={"maximum_projection_error":1e-5,"maximum_closure_error":1e-4,"minimum_cosine":.75,"maximum_relative_l2":.75,"minimum_norm_ratio":.25,"maximum_norm_ratio":2.,"minimum_sign_agreement":.75,"minimum_group_cosine":.65,"minimum_group_sign_agreement":.65}
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 absolute,null,disp,readers,ref=(json.loads(p.read_text()) for p in (ABS,NULL,DISP,READERS,REF))
 if absolute.get("terminal")!="prototype_artifact" or null.get("terminal")!="null" or disp.get("terminal")!="prototype_artifact" or readers.get("terminal")!="reader_artifact" or ref.get("terminal")!="valid_causal_screen":raise ValueError("parent status invalid")
 return absolute,disp,readers,ref
def compile_plan():
 _load();return {"schema":"task14_absolute_head_reader_scalar_context_plan_v2","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"absolute_artifact_sha256":EXPECTED[ABS],"absolute_null_sha256":EXPECTED[NULL],"displacement_sha256":EXPECTED[DISP],"readers_sha256":EXPECTED[READERS],"reference_sha256":EXPECTED[REF],"rows":32,"backgrounds":16,"methods":["base","one_scalar_corrected_absolute"],"native_context_scalars_per_edit":1,"discarded_orthogonal_coordinates":1151,"bars":dict(BARS),"price":{"physical_model_forwards":5,"example_evaluations":1120,"causal_installations":1024,"backwards":0,"fits":0,"parameter_updates":0}}
def evaluate(model,torch,F,facade):
 absolute,disp,reader_artifact,reference=_load();rows=validation.authority.build_rows();parent,tokens,function,inputs,source_closure=absolute_run._context(model,rows,torch,F,facade);device=tokens.device;absv={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=device) for k,v in absolute["prototypes"].items()};deltav={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=device) for k,v in disp["prototypes"].items() if ".cardinality_" in k};readers={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=device) for k,v in reader_artifact["readers"].items()};indices=[];replacements=[];specs=[];projection_errors=[]
 with torch.no_grad():
  for subset in SUBSETS:
   base=function(factor_gate._raw_for(inputs["recipient"],inputs["opposite"],subset,F)).detach()
   for i,row in enumerate(rows):
    key=f'{row["direction_id"]}.cardinality_{len(subset)}';reader=readers[row["direction_id"]];mean_base=absv[key]-deltav[key];scalar=torch.dot(reader,base[i]-mean_base);correction=reader*(scalar/torch.dot(reader,reader));corrected=absv[key]+correction;projection_errors.append(float(abs(torch.dot(reader,corrected-absv[key])-scalar)));indices.extend((i,i));replacements.extend((base[i],corrected));specs.extend(((i,subset,"base"),(i,subset,"corrected")))
  index=torch.tensor(indices,dtype=torch.long,device=device);patch_tokens=tokens[:len(rows)][index];finals=torch.full_like(index,parent.SUBJECT_POSITION);replacement=torch.stack(replacements);margins={};closures=[]
  for start in range(0,len(specs),CHUNK):
   stop=min(start+CHUNK,len(specs));logits,_,_,closure=parent.downstream._decomposed_forward(model,patch_tokens[start:stop],finals[start:stop],torch,F,facade,replacement_heads=replacement[start:stop],native_reinstall_mask=torch.zeros(stop-start,dtype=torch.bool,device=device));closures.append(closure)
   for local,(i,subset,method) in enumerate(specs[start:stop]):
    endpoint=rows[i]["endpoints"]["opposite_same_lemma"];margins[(i,subset,method)]=float(logits[local,parent.SUBJECT_POSITION,endpoint["answer_id"]]-logits[local,parent.SUBJECT_POSITION,endpoint["foil_id"]])
 ref={(r["row_id"],r["background"]):r for r in reference["score"]["joined_evidence"]};evidence=[]
 for i,row in enumerate(rows):
  for subset in SUBSETS:evidence.append({"row_id":row["row_id"],"direction":row["direction_id"],"template":row["template_id"],"background":subset,"cardinality":len(subset),"native_exact_q":ref[(row["row_id"],subset)]["native_exact_q"],"one_scalar_corrected_q":margins[(i,subset,"corrected")]-margins[(i,subset,"base")]})
 return evidence,max(projection_errors),source_closure,closures
def _metrics(rows):
 a=[r["native_exact_q"] for r in rows];p=[r["one_scalar_corrected_q"] for r in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/(an*pn),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/an,"predicted_to_actual_norm_ratio":pn/an,"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(rows)}
def score(evidence,projection_error,source_closure,closures):
 errors=[source_closure["input_state_closure_max_absolute_error"],source_closure["input_normalized_closure_max_absolute_error"]]+[v for c in closures for k,v in c.items() if k in ("state_sum_max_absolute_error","normalized_state_max_absolute_error")];overall=_metrics(evidence);cells={f"{d}.{t}":_metrics([r for r in evidence if r["direction"]==d and r["template"]==t]) for d in ("plural_to_singular","singular_to_plural") for t in sorted({r["template"] for r in evidence})};cards={str(c):_metrics([r for r in evidence if r["cardinality"]==c]) for c in range(5)};instrument=len(evidence)==512 and projection_error<=BARS["maximum_projection_error"] and max(errors)<=BARS["maximum_closure_error"];transfer=overall["cosine"]>=BARS["minimum_cosine"] and overall["relative_l2_error"]<=BARS["maximum_relative_l2"] and BARS["minimum_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_sign_agreement"];recurrence=all(v["cosine"]>=BARS["minimum_group_cosine"] and v["sign_agreement"]>=BARS["minimum_group_sign_agreement"] for v in list(cells.values())+list(cards.values()));pred={"pred_a_immutable_reader_aligned_instrument":instrument,"pred_b_one_scalar_context_restores_effect":instrument and transfer,"pred_c_each_direction_template_and_cardinality_recurs":instrument and recurrence,"pred_d_context_dependency_reduced_to_one_scalar":True,"pred_e_fixed_price":compile_plan()["price"]=={"physical_model_forwards":5,"example_evaluations":1120,"causal_installations":1024,"backwards":0,"fits":0,"parameter_updates":0}};return {"reader_projection_max_absolute_error":projection_error,"closure_max_absolute_error":max(errors),"overall":overall,"by_direction_template":cells,"by_cardinality":cards,"context_interface":{"native_base_scalars_per_edit":1,"discarded_orthogonal_coordinates":1151,"frozen_directions":2},"predictions":pred,"terminal":"program_screen" if all(pred.values()) else "null" if instrument else "invalid"}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise ValueError(f"refusing overwrite {OUT}")
 torch,F,facade=tangent.parent.factors._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():evidence,projection_error,source_closure,closures=evaluate(model,torch,F,facade)
 scored=score(evidence,projection_error,source_closure,closures);payload=managed.atomic_create_json(OUT,{"schema":"task14_absolute_head_reader_scalar_context_result_v2","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":evidence,"terminal":scored["terminal"]});print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
