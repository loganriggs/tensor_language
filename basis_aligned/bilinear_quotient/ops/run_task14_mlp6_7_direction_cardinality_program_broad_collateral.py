#!/usr/bin/env python3
"""Install all ten Task14 writes on four additional behavior circuits."""

# BQGATE: EXPERIMENT pred_a_authority_native_and_noop pred_b_all_ten_writes_live pred_c_polarity_preserved pred_d_narrative_tense_preserved pred_e_preposition_and_voice_preserved pred_f_complete_fixed_program_and_price
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics
from datetime import datetime,timezone
from pathlib import Path

import circuit_fast_screen_candidate_task14_program_broad_collateral as authority
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import run_task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral as base

ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/"circuits/prior_art/task14_mlp6_7_direction_cardinality_program_broad_collateral_v1.json"
OUT=ROOT/"circuits/followups/task14_mlp6_7_direction_cardinality_program_broad_collateral_v1_result.json"
PRIOR_SHA="693bccfc45239a495a98da0b258634216c0cba8883e99ef59860b168f6a6d9f9"
AUTHORITY_FILE_SHA="ca2541290fdd2ae70109a3b4a9e2c3770804c7bf05b4a487b1397c7192aea031"
OLD_COLLATERAL=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_cross_circuit_collateral_v2_result.json"
OLD_COLLATERAL_SHA="4efe92e1c925ebb9d9db0d1c5158a6d26bcd6e3bdcd3c0336bc162b479705e04"
CHUNK=128;MAX_NOOP=1e-4;MAX_INSTALL=5e-5;MAX_MEDIAN=.10;MAX_ROW=.25;MIN_ROWS=28;MAX_FLIPS=2
PRED_KEYS=("pred_a_authority_native_and_noop","pred_b_all_ten_writes_live","pred_c_polarity_preserved","pred_d_narrative_tense_preserved","pred_e_preposition_and_voice_preserved","pred_f_complete_fixed_program_and_price")
class BroadCollateralError(ValueError):pass
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive_price():return {"physical_model_forwards":12,"example_evaluations":1536,"program_installations":1280,"zero_add_replays":128,"native_evaluations":128,"backwards":0,"parameter_updates":0,"maximum_batch":CHUNK}
def validate_preflight():
 for path,expected,label in ((PRIOR,PRIOR_SHA,"prior art"),(Path(authority.__file__),AUTHORITY_FILE_SHA,"authority"),(OLD_COLLATERAL,OLD_COLLATERAL_SHA,"old collateral"),(base.PROTOTYPES,base.PROTOTYPE_SHA256,"prototypes")):
  if _sha(path)!=expected:raise BroadCollateralError(f"{label} changed")
 rows=authority.build_rows();prototypes=base._load_prototypes()
 if len(rows)!=128 or len(prototypes)!=10:raise BroadCollateralError("authority/program cardinality changed")
def compile_plan():
 validate_preflight();rows=authority.build_rows();return {"schema":"task14_mlp6_7_direction_cardinality_program_broad_collateral_plan_v1","candidate_id":"subject_verb.number_agreement.direction_cardinality_program_broad_collateral_v1","row_count":len(rows),"behaviors":list(authority.SOURCES),"rows_per_behavior":32,"prototype_count":10,"projected_write_layer":11,"projected_write_width":1152,"prior_art_sha256":PRIOR_SHA,"authority_sha256":authority.validate_rows(rows),"bars":{"maximum_noop_absolute_logit_error":MAX_NOOP,"maximum_install_absolute_error":MAX_INSTALL,"maximum_median_normalized_effect":MAX_MEDIAN,"maximum_row_normalized_effect":MAX_ROW,"minimum_rows_below_row_bar":MIN_ROWS,"maximum_answer_flips":MAX_FLIPS,"pool_behaviors":False,"pool_prototypes":False},"predictions":{key:key for key in PRED_KEYS},"price":derive_price()}
def _run_projected(executor,rows,vectors):
 pairs=[];error=0.0
 for start in range(0,len(rows),CHUNK):
  stop=min(start+CHUNK,len(rows));chunk_vectors=None if vectors is None else vectors[start:stop];out,e=base._projected_add(executor,base._batch(rows[start:stop]),chunk_vectors);pairs.extend(out);error=max(error,e)
 return pairs,error
def evaluate(executor):
 source=authority.build_rows();rows=[{**x,"run_id":x["row_id"]} for x in source];native=executor.native(base._batch(rows),capture=False).answer_foil;noop,noop_install=_run_projected(executor,rows,None);prototypes=base._load_prototypes();program_rows=[];vectors=[]
 for row in source:
  for key,value in sorted(prototypes.items()):program_rows.append({**row,"source_row_id":row["row_id"],"run_id":f"{row['row_id']}:{key}","prototype_key":key});vectors.append(value["coordinates"])
 vector_tensor=executor.torch.tensor(vectors,dtype=executor.torch.float32,device=executor.device);patched,install=_run_projected(executor,program_rows,vector_tensor);native_map={x["row_id"]:p for x,p in zip(source,native)};noop_error=max(abs(a-b) for p,q in zip(native,noop) for a,b in zip(p,q));scales={behavior:statistics.median(p[0]-p[1] for x,p in zip(source,native) if x["behavior"]==behavior) for behavior in authority.SOURCES};evidence=[]
 for row,pair in zip(program_rows,patched):
  n=native_map[row["source_row_id"]];native_margin=n[0]-n[1];patched_margin=pair[0]-pair[1];evidence.append({"row_id":row["source_row_id"],"behavior":row["behavior"],"prototype_key":row["prototype_key"],"native_margin":native_margin,"patched_margin":patched_margin,"normalized_absolute_effect":abs(patched_margin-native_margin)/scales[row["behavior"]],"answer_flipped":patched_margin<=0})
 summaries={}
 for behavior in authority.SOURCES:
  for key in sorted(prototypes):
   subset=[x for x in evidence if x["behavior"]==behavior and x["prototype_key"]==key];effects=[x["normalized_absolute_effect"] for x in subset];summaries[f"{behavior}.{key}"]={"behavior":behavior,"prototype_key":key,"row_count":len(subset),"native_scale":scales[behavior],"median_normalized_absolute_effect":statistics.median(effects),"rows_at_or_below_0_25":sum(x<=MAX_ROW for x in effects),"answer_flips":sum(x["answer_flipped"] for x in subset),"passed_preservation":statistics.median(effects)<=MAX_MEDIAN and sum(x<=MAX_ROW for x in effects)>=MIN_ROWS and sum(x["answer_flipped"] for x in subset)<=MAX_FLIPS}
 instrument=all(p[0]-p[1]>0 for p in native) and all(x>0 for x in scales.values()) and noop_error<=MAX_NOOP;live=len(prototypes)==10 and all(math.isfinite(float(x["l2_norm"])) and float(x["l2_norm"])>0 for x in prototypes.values()) and max(noop_install,install)<=MAX_INSTALL;preserved={behavior:all(x["passed_preservation"] for x in summaries.values() if x["behavior"]==behavior) for behavior in authority.SOURCES};complete=len(evidence)==1280 and len({(x["row_id"],x["prototype_key"]) for x in evidence})==1280;predictions=dict(zip(PRED_KEYS,(instrument,live,preserved["polarity"],preserved["narrative_tense"],preserved["preposition_selection"] and preserved["voice_frame"],complete and derive_price()["physical_model_forwards"]<=12)))
 return {"noop_max_absolute_logit_error":noop_error,"maximum_install_absolute_error":max(noop_install,install),"behavior_scales":scales,"behavior_preserved":preserved,"behavior_prototype_results":summaries,"evidence":evidence,"predictions":predictions}
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);plan=compile_plan()
 if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise BroadCollateralError(f"refusing overwrite {OUT}")
 executor=producer.Bilin18TorchBackend.load("cuda");score=evaluate(executor);instrument=score["predictions"][PRED_KEYS[0]] and score["predictions"][PRED_KEYS[1]] and score["predictions"][PRED_KEYS[5]];terminal="screen" if all(score["predictions"].values()) else "null" if instrument else "invalid";payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_direction_cardinality_program_broad_collateral_result_v1","candidate_id":plan["candidate_id"],"terminal":terminal,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score,"limits":"Six measured unrelated behaviors after joining the prior panel; not universal selectivity."});print(json.dumps({"terminal":terminal,"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
