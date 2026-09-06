#!/usr/bin/env python3
"""Bracket-program robustness under outcome-free norm-matched Task14 stress."""

# BQGATE: EXPERIMENT pred_a_immutable_artifacts_gain_and_corners pred_b_norm_matched_task14_stress_is_live pred_c_bracket_program_is_robust pred_d_no_original_gain_upgrade pred_e_fixed_price
from __future__ import annotations
from datetime import datetime,timezone
import copy,hashlib,json,math,os,statistics,sys
from pathlib import Path
import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as exact
import run_task14_bracket_fixed_program_stress_composition as parent

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/bracket_under_norm_matched_task14_program_stress_v3.json"
TA=parent.TASK14_ARTIFACT; BA=parent.BRACKET_ARTIFACT; BR=parent.BRACKET_RESULT
OUT=ROOT/"circuits/followups/bracket_under_norm_matched_task14_program_stress_v3_result.json"
EXPECTED={PRIOR:"e7828b663973f506ac81c66c5f158668e2e36d453f866731d0e1ece9b8177cb0",TA:"cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",BA:"531434535daf526ebf584564afbfd5834c75df7bd636014ea233f9ae71206db0",BR:"3b267f069647824fb7557e9784c63becb0366f94fe4d274fea343ae2bc802e5f"}
CANDIDATE_ID="cross_behavior.bracket_under_norm_matched_task14_program_stress_v3"
FROZEN_GAIN=137.63389776183087

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _norm(v):return math.sqrt(sum(float(x)*float(x) for x in v))
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 ta,ba,br=json.loads(TA.read_text()),json.loads(BA.read_text()),json.loads(BR.read_text())
 if ta.get("terminal")!="prototype_artifact" or ba.get("terminal")!="prototype_artifact" or br.get("terminal")!="program_screen":raise ValueError("parent artifact invalid")
 return ta,ba,br
def derive_gain():
 ta,ba,_=_load();tn=[_norm(v["coordinates"]) for k,v in ta["prototypes"].items() if ".cardinality_" in k];bn=[_norm(v["coordinates"]) for v in ba["prototypes"].values()]
 return statistics.median(bn)/statistics.median(tn)
def compile_plan():
 gain=derive_gain()
 if abs(gain-FROZEN_GAIN)>1e-12:raise ValueError("artifact-derived gain changed")
 return {"schema":"bracket_under_norm_matched_task14_program_stress_plan_v3","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"task14_artifact_sha256":EXPECTED[TA],"bracket_artifact_sha256":EXPECTED[BA],"bracket_parent_sha256":EXPECTED[BR],"gain":gain,"gain_source":"artifact median L2 norm ratio; zero outcome values","rows":72,"endpoints":144,"arms":list(parent.ARMS),"scope":"norm-matched robustness only; original-gain reverse and two-sided composition remain closed","bars":dict(parent.BARS),"price":{"physical_model_forwards":1,"example_evaluations":576,"backwards":0,"fits":0,"parameter_updates":0,"vector_selection_changes":0}}
def evaluate(model,torch,F,facade):
 ta,ba,_=_load();scaled=copy.deepcopy(ta)
 for key,value in scaled["prototypes"].items():
  if ".cardinality_" in key:value["coordinates"]=[FROZEN_GAIN*float(x) for x in value["coordinates"]]
 return parent._bracket_panel(model,torch,F,facade,scaled,ba)
def score(rows):
 _,_,br=_load();prior={(x["row_id"],x["side"]):x["program_donorward_effect"] for x in br["evidence"] if x["program_role"]=="target"}
 replay=max(abs(x["isolated_own"]-prior[(x["row_id"],x["side"])]) for x in rows);panel=parent._panel_score(rows);p=panel["preservation"]
 instrument=len(rows)==144 and replay<=1e-4
 live=panel["foreign_stress_to_own_norm_ratio"]>=.05
 robust=p["cosine"]>=.90 and p["relative_l2_error"]<=.40 and p["sign_agreement"]>=.90 and panel["interaction_to_own_norm_ratio"]<=.40
 pred={"pred_a_immutable_artifacts_gain_and_corners":instrument and abs(derive_gain()-FROZEN_GAIN)<=1e-12,"pred_b_norm_matched_task14_stress_is_live":instrument and live,"pred_c_bracket_program_is_robust":instrument and robust,"pred_d_no_original_gain_upgrade":True,"pred_e_fixed_price":True}
 terminal="robustness_screen" if all(pred.values()) else "inconclusive" if instrument and not live else "interaction_null" if instrument else "invalid"
 return {"gain":FROZEN_GAIN,"bracket":panel,"parent_replay_max_absolute_error":replay,"scope":"norm-matched robustness only","explicitly_not_promoted":["original-gain reverse composition","two-sided composition"],"predictions":pred,"terminal":terminal}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise ValueError(f"refusing overwrite {OUT}")
 torch,F,facade=exact._dependencies();model,checkpoint=facade.load_bilin18(device="cuda",dtype=torch.float32,verify_weights_sha256=True)
 with torch.no_grad():rows=evaluate(model,torch,F,facade)
 score_value=score(rows);payload=managed.atomic_create_json(OUT,{"schema":"bracket_under_norm_matched_task14_program_stress_result_v3","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"checkpoint_weights_sha256":checkpoint.weights_sha256,"score":score_value,"evidence":rows,"terminal":score_value["terminal"]});print(json.dumps({"terminal":score_value["terminal"],"predictions":score_value["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
