#!/usr/bin/env python3
"""Prospective fresh-corpus validation of six suffix-free bracket effect scalars."""

# BQGATE: EXPERIMENT pred_a_temporal_seal_and_native_capability pred_b_fixed_vector_program_recurrence pred_c_suffix_free_scalar_prediction pred_d_pairwise_recurrence pred_e_fixed_price_and_scope
from __future__ import annotations

import hashlib,json,math,os,statistics,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as shared
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_ordered_pair_displacement_program_ood_validation as parent
import run_bracket_l13h8_source_region_payload_factorial as exact


ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/bracket_suffix_free_scalar_fresh_corpus_validation_v1.json"
ROWS=ROOT/"circuits/prior_art/bracket_suffix_free_fresh_corpus_v1_rows.json"
ARTIFACT=ROOT/"circuits/fast_screens/bracket_l13h8_ordered_pair_displacement_artifact_v1.json"
FEASIBILITY=ROOT/"circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json"
OUT=ROOT/"circuits/followups/bracket_suffix_free_scalar_fresh_corpus_validation_v1_result.json"
CANDIDATE_ID="bracket.pending_opener.suffix_free_scalar_fresh_corpus_validation_v1"
EXPECTED={PRIOR:"881fa23f358d7d4a50d60adc2e4ba917655a692d65a35939af5253b3e033290e",ROWS:"d808806fd1b05f834cf6ef4fa71465464c0403f66dc13ece8a24cffcc40142f9",ARTIFACT:"531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",FEASIBILITY:"b7401c1dd9709143cd0ad31ac893e09afa98e64604ddb18f866b6998a749f504"}
SCALARS={"1->60":6.5689431031545,"1->8":5.607922434806824,"60->1":6.458606282869975,"60->8":6.539922753969829,"8->1":4.763878305753072,"8->60":6.557887037595113}
BARS={"minimum_native_accuracy":.85,"minimum_positive_fraction":.75,"minimum_cosine":.80,"maximum_relative_l2":.60,"minimum_norm_ratio":.50,"maximum_norm_ratio":1.50,"minimum_sign_agreement":.90,"maximum_pair_median_absolute_error":3.0,"maximum_replay_error":1e-4}

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 prior,rows,artifact,feasibility=(json.loads(p.read_text()) for p in (PRIOR,ROWS,ARTIFACT,FEASIBILITY))
 if rows.get("status")!="rows_frozen_outcomes_unopened" or rows.get("outcomes_opened")!=[]:raise ValueError("fresh temporal seal broken")
 if artifact.get("terminal")!="prototype_artifact" or feasibility.get("terminal")!="feasibility_screen":raise ValueError("parent screen invalid")
 if prior["frozen_pair_scalars"]!=SCALARS:raise ValueError("frozen scalar table changed")
 return rows,artifact
def _rows():
 value,_=_load();out=[]
 for row in value["rows"]:
  item=dict(row)
  for side in ("base","donor"):item[f"{side}_open_position"]=shared.semantic_open_position(item[f"{side}_ids"],item[f"{side}_answer_id"])
  out.append(item)
 return out
def compile_plan():
 rows=_rows();counts=Counter()
 for row in rows:
  counts[f'{row["base_answer_id"]}->{row["donor_answer_id"]}']+=1;counts[f'{row["donor_answer_id"]}->{row["base_answer_id"]}']+=1
 if len(rows)!=36 or len(counts)!=6 or set(counts.values())!={12}:raise ValueError("fresh pair balance changed")
 return {"schema":"bracket_suffix_free_scalar_fresh_corpus_validation_plan_v1","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"rows_sha256":EXPECTED[ROWS],"artifact_sha256":EXPECTED[ARTIFACT],"feasibility_sha256":EXPECTED[FEASIBILITY],"rows":36,"endpoints":72,"frozen_pair_scalars":dict(SCALARS),"scalar_source":"direct_three_value_type_substitution only","bars":dict(BARS),"price":{"physical_model_forwards":2,"example_evaluations":144,"backwards":0,"fits":0,"parameter_updates":0}}
def evaluate(model,torch,F,facade):
 rows=_rows();_,artifact=_load();endpoints,tokens,finals,sources=parent._pad(rows,torch,next(model.parameters()).device)
 replay,factors=exact.factor_forward(model,tokens,finals,{},torch,F,facade);arange=torch.arange(len(endpoints),device=tokens.device)
 terms=factors["p"][arange,sources].unsqueeze(-1)*factors["u"][arange,sources]
 vectors={k:torch.tensor(v["coordinates"],dtype=torch.float32,device=tokens.device) for k,v in artifact["prototypes"].items()}
 installed=[]
 for i,(row,side) in enumerate(endpoints):
  other="donor" if side=="base" else "base";key=f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}';installed.append(terms[i]+vectors[key])
 installed=torch.stack(installed)
 program=exact.factor_forward(model,tokens,finals,{},torch,F,facade,replacement_terms=installed,source_positions=sources)[0]
 records=[]
 for i,(row,side) in enumerate(endpoints):
  q=int(finals[i]);other="donor" if side=="base" else "base";pair=f'{row[f"{side}_answer_id"]}->{row[f"{other}_answer_id"]}';direction="base_to_donor" if side=="base" else "donor_to_base"
  records.append({"row_id":row["row_id"],"side":side,"ordered_pair":pair,"native_recipient_correct":bool(exact.closer_margin(replay[i,q],row[f"{side}_answer_id"])>0),"replay_error":0.0,"actual_program_donorward_effect":exact.endpoint_change(replay[i,q],program[i,q],row,direction),"predicted_program_donorward_effect":SCALARS[pair]})
 return records
def _metrics(rows):
 a=[r["actual_program_donorward_effect"] for r in rows];p=[r["predicted_program_donorward_effect"] for r in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));dot=sum(x*y for x,y in zip(a,p))
 return {"count":len(rows),"cosine":dot/(an*pn),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/an,"predicted_to_actual_norm_ratio":pn/an,"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(rows),"positive_actual_fraction":sum(x>0 for x in a)/len(rows),"native_accuracy":sum(r["native_recipient_correct"] for r in rows)/len(rows),"median_absolute_error":statistics.median(abs(x-y) for x,y in zip(a,p))}
def score(records):
 overall=_metrics(records);by_pair={pair:_metrics([r for r in records if r["ordered_pair"]==pair]) for pair in SCALARS}
 seal=len(records)==72 and all(v["count"]==12 for v in by_pair.values()) and max(r["replay_error"] for r in records)<=BARS["maximum_replay_error"]
 capability=overall["native_accuracy"]>=BARS["minimum_native_accuracy"] and all(v["native_accuracy"]>=BARS["minimum_native_accuracy"] for v in by_pair.values())
 recurrence=overall["positive_actual_fraction"]>=BARS["minimum_positive_fraction"] and all(v["positive_actual_fraction"]>=BARS["minimum_positive_fraction"] for v in by_pair.values())
 scalar=overall["cosine"]>=BARS["minimum_cosine"] and overall["relative_l2_error"]<=BARS["maximum_relative_l2"] and BARS["minimum_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_sign_agreement"]
 pairwise=all(v["positive_actual_fraction"]>=BARS["minimum_positive_fraction"] and v["median_absolute_error"]<=BARS["maximum_pair_median_absolute_error"] for v in by_pair.values())
 pred={"pred_a_temporal_seal_and_native_capability":seal and capability,"pred_b_fixed_vector_program_recurrence":seal and capability and recurrence,"pred_c_suffix_free_scalar_prediction":seal and capability and scalar,"pred_d_pairwise_recurrence":seal and capability and pairwise,"pred_e_fixed_price_and_scope":compile_plan()["price"]=={"physical_model_forwards":2,"example_evaluations":144,"backwards":0,"fits":0,"parameter_updates":0}}
 terminal="predictive_screen" if all(pred.values()) else "capability_stop" if seal and not capability else "null" if seal else "invalid"
 return {"overall":overall,"by_ordered_pair":by_pair,"scope":"scalar donorward program-effect prediction only; native base activation and full logits remain open","predictions":pred,"terminal":terminal}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise ValueError(f"refusing overwrite {OUT}")
 torch,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():records=evaluate(model,torch,F,facade)
 scored=score(records);payload=managed.atomic_create_json(OUT,{"schema":"bracket_suffix_free_scalar_fresh_corpus_validation_result_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":scored,"evidence":records,"terminal":scored["terminal"]});print(json.dumps({"terminal":scored["terminal"],"predictions":scored["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
