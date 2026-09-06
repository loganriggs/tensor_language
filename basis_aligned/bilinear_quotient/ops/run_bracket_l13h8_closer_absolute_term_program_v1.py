#!/usr/bin/env python3
"""Export three closer-conditioned absolute opener terms and test them prospectively."""

# BQGATE: EXPERIMENT pred_a_temporal_seal_export_and_instrument pred_b_absolute_program_reproduces_exact_swap pred_c_each_ordered_pair_recurrence pred_d_native_opener_term_dependency_removed_narrowly pred_e_fixed_price
from __future__ import annotations
import hashlib,json,math,os,statistics,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as select_authority
import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_ordered_pair_displacement_program_ood_validation as parent
import run_bracket_l13h8_source_region_payload_factorial as exact

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/bracket_l13h8_closer_absolute_term_program_v1.json";ROWS=ROOT/"circuits/prior_art/bracket_absolute_term_fresh_corpus_v1_rows.json";SELECT=ROOT/"pending_opener_three_value_fresh_rows_rung545.json";NEGATIVE=ROOT/"circuits/fast_screens/bracket_l13h8_pair_centered_open_term_final_test_v1_result.json"
ARTIFACT_OUT=ROOT/"circuits/followups/bracket_l13h8_closer_absolute_term_program_v1_artifact.json";OUT=ROOT/"circuits/followups/bracket_l13h8_closer_absolute_term_program_v1_result.json"
CANDIDATE_ID="bracket.pending_opener.l13h8_closer_absolute_term_program_v1"
EXPECTED={PRIOR:"0e546b3a9b6352ef543bdf971cd5c01a65edea6bebbf8843f0f9a58fb73b54ce",ROWS:"92ee66ce0a4bf084789bf0c2af394a107a49df19dbf3c9fbfbebbe467d873c76",SELECT:"07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",NEGATIVE:"e64093354428d62eecd268360a79e8ef6549437babdaf9897d853134a44000f6"}
BARS={"maximum_replay_error":1e-4,"minimum_native_accuracy":.85,"minimum_cosine":.65,"maximum_relative_l2":.90,"minimum_norm_ratio":.25,"maximum_norm_ratio":2.,"minimum_sign_agreement":.75,"minimum_pair_positive_fraction":.75,"minimum_pair_sign_agreement":.70}
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _check():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 fresh=json.loads(ROWS.read_text());negative=json.loads(NEGATIVE.read_text())
 if fresh.get("status")!="rows_frozen_outcomes_unopened" or fresh.get("outcomes_opened")!=[]:raise ValueError("fresh seal broken")
 if negative.get("terminal")!="screen" or negative["screen"]["predictions"]["pred_b_pair_centered_open_term_held"]:raise ValueError("negative boundary changed")
 return fresh
def _fresh():
 rows=[]
 for row in _check()["rows"]:
  item=dict(row)
  for side in ("base","donor"):item[f"{side}_open_position"]=shared.semantic_open_position(item[f"{side}_ids"],item[f"{side}_answer_id"])
  rows.append(item)
 return rows
def compile_plan():
 select=select_authority.build_export_rows();fresh=_fresh();sc=Counter(r[f"{s}_answer_id"] for r in select for s in ("base","donor"));fc=Counter()
 for r in fresh:
  fc[f'{r["base_answer_id"]}->{r["donor_answer_id"]}']+=1;fc[f'{r["donor_answer_id"]}->{r["base_answer_id"]}']+=1
 if len(select)!=72 or sc!={1:48,8:48,60:48} or len(fresh)!=36 or len(fc)!=6 or set(fc.values())!={12}:raise ValueError("balance changed")
 return {"schema":"bracket_l13h8_closer_absolute_term_program_plan_v1","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"select_rows_sha256":EXPECTED[SELECT],"fresh_rows_sha256":EXPECTED[ROWS],"negative_boundary_sha256":EXPECTED[NEGATIVE],"export":{"rows":72,"endpoints":144,"closer_support":{"1":48,"8":48,"60":48},"vectors":3,"width":1152},"validation":{"rows":36,"endpoints":72,"ordered_pair_support":12},"bars":dict(BARS),"price":{"physical_model_forwards":4,"example_evaluations":360,"backwards":0,"fits":0,"parameter_updates":0,"stored_fp32_scalars":3456}}
def _terms(model,rows,torch,F,facade):
 endpoints,tokens,finals,sources=parent._pad(rows,torch,next(model.parameters()).device);logits,factors=exact.factor_forward(model,tokens,finals,{},torch,F,facade);arange=torch.arange(len(endpoints),device=tokens.device);terms=factors["p"][arange,sources].unsqueeze(-1)*factors["u"][arange,sources];head_rebuilt=torch.einsum("bk,bkd->bd",factors["p"],factors["u"]);replay=float((head_rebuilt-factors["head"]).abs().max());return endpoints,tokens,finals,sources,logits,terms,replay
def export(model,torch,F,facade):
 rows=select_authority.build_export_rows();endpoints,_,_,_,_,terms,replay=_terms(model,rows,torch,F,facade);groups={closer:[] for closer in (1,8,60)}
 for i,(row,side) in enumerate(endpoints):groups[row[f"{side}_answer_id"]].append(terms[i])
 vectors={str(k):torch.stack(v).mean(0).detach() for k,v in groups.items()};return vectors,replay
def evaluate(model,vectors,torch,F,facade):
 rows=_fresh();endpoints,tokens,finals,sources,replay,terms,replay_error=_terms(model,rows,torch,F,facade);donor_terms=terms[torch.arange(len(endpoints),device=tokens.device)^1];absolute=[]
 for row,side in endpoints:
  other="donor" if side=="base" else "base";absolute.append(vectors[str(row[f"{other}_answer_id"])])
 absolute=torch.stack(absolute);exact_logits=exact.factor_forward(model,tokens,finals,{},torch,F,facade,replacement_terms=donor_terms,source_positions=sources)[0];program_logits=exact.factor_forward(model,tokens,finals,{},torch,F,facade,replacement_terms=absolute,source_positions=sources)[0]
 records=[]
 for i,(row,side) in enumerate(endpoints):
  q=int(finals[i]);other="donor" if side=="base" else "base";direction="base_to_donor" if side=="base" else "donor_to_base";pair=f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}'
  records.append({"row_id":row["row_id"],"side":side,"ordered_pair":pair,"native_recipient_correct":bool(exact.closer_margin(replay[i,q],row[f"{side}_answer_id"])>0),"exact_donorward_effect":exact.endpoint_change(replay[i,q],exact_logits[i,q],row,direction),"absolute_program_donorward_effect":exact.endpoint_change(replay[i,q],program_logits[i,q],row,direction)})
 return records,replay_error
def _metrics(rows):
 a=[r["exact_donorward_effect"] for r in rows];p=[r["absolute_program_donorward_effect"] for r in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/(an*pn),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/an,"predicted_to_actual_norm_ratio":pn/an,"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(rows),"positive_program_fraction":sum(x>0 for x in p)/len(rows),"native_accuracy":sum(r["native_recipient_correct"] for r in rows)/len(rows),"median_absolute_effect_error":statistics.median(abs(x-y) for x,y in zip(a,p))}
def score(records,export_replay,fresh_replay):
 overall=_metrics(records);pairs={k:_metrics([r for r in records if r["ordered_pair"]==k]) for k in sorted({r["ordered_pair"] for r in records})};instrument=len(records)==72 and len(pairs)==6 and all(v["count"]==12 for v in pairs.values()) and max(export_replay,fresh_replay)<=BARS["maximum_replay_error"] and overall["native_accuracy"]>=BARS["minimum_native_accuracy"] and all(v["native_accuracy"]>=BARS["minimum_native_accuracy"] for v in pairs.values());transfer=overall["cosine"]>=BARS["minimum_cosine"] and overall["relative_l2_error"]<=BARS["maximum_relative_l2"] and BARS["minimum_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_sign_agreement"];recurrence=all(v["positive_program_fraction"]>=BARS["minimum_pair_positive_fraction"] and v["sign_agreement"]>=BARS["minimum_pair_sign_agreement"] for v in pairs.values());pred={"pred_a_temporal_seal_export_and_instrument":instrument,"pred_b_absolute_program_reproduces_exact_swap":instrument and transfer,"pred_c_each_ordered_pair_recurrence":instrument and recurrence,"pred_d_native_opener_term_dependency_removed_narrowly":True,"pred_e_fixed_price":compile_plan()["price"]=={"physical_model_forwards":4,"example_evaluations":360,"backwards":0,"fits":0,"parameter_updates":0,"stored_fp32_scalars":3456}};terminal="program_screen" if all(pred.values()) else "capability_stop" if max(export_replay,fresh_replay)<=BARS["maximum_replay_error"] and not instrument else "null" if instrument else "invalid";return {"export_head_replay_max_absolute_error":export_replay,"fresh_head_replay_max_absolute_error":fresh_replay,"overall":overall,"by_ordered_pair":pairs,"dependency_removed":"recipient native L13H8 semantic-opener term is not read when constructing edited replacement","dependencies_retained":["all other upstream context","semantic opener position","native suffix for causal execution"],"negative_boundary_retained":"pair-centered selective necessity remains false","predictions":pred,"terminal":terminal}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if OUT.exists() or ARTIFACT_OUT.exists():raise ValueError("refusing overwrite")
 torch,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():vectors,export_replay=export(model,torch,F,facade);records,fresh_replay=evaluate(model,vectors,torch,F,facade)
 artifact={"schema":"bracket_l13h8_closer_absolute_term_program_artifact_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"checkpoint_weights_sha256":checkpoint.weights_sha256,"plan":plan,"prototypes":{k:{"closer_id":int(k),"support":48,"coordinates":v.cpu().tolist()} for k,v in vectors.items()},"terminal":"prototype_artifact"};artifact_bytes=managed.atomic_create_json(ARTIFACT_OUT,artifact);scored=score(records,export_replay,fresh_replay);payload=managed.atomic_create_json(OUT,{"schema":"bracket_l13h8_closer_absolute_term_program_result_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"artifact_sha256":hashlib.sha256(artifact_bytes).hexdigest(),"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":records,"terminal":scored["terminal"]});print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"artifact_sha256":hashlib.sha256(artifact_bytes).hexdigest(),"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
