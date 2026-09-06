#!/usr/bin/env python3
"""Compile and verify the strongest predictive/manipulable two-program package."""

# BQGATE: EXPERIMENT pred_a_exact_selected_compilation pred_b_task14_reader_contraction pred_c_bracket_absolute_and_no_edit_dispatch pred_d_literal_smaller_price pred_e_residual_boundary
# BQLANE: cpu
from __future__ import annotations
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_managed_runner as managed
import task14_bracket_predictive_dispatcher as dispatcher

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/task14_bracket_compiled_predictive_dispatcher_v3.json";TASK=ROOT/"circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json";READERS=ROOT/"circuits/fast_screens/task14_mlp6_7_fixed_direction_reader_artifact_v2_result.json";ABS=ROOT/"circuits/followups/bracket_l13h8_closer_absolute_term_program_v1_artifact.json";BROAD=ROOT/"circuits/followups/bracket_absolute_term_program_ood_control_validation_v2_result.json";EFFECT=ROOT/"circuits/followups/bracket_suffix_free_scalar_fresh_corpus_validation_v1_result.json";PACKAGE=ROOT/"circuits/followups/task14_bracket_compiled_predictive_dispatcher_v3_artifact.json";OUT=ROOT/"circuits/followups/task14_bracket_compiled_predictive_dispatcher_v3_result.json"
CANDIDATE_ID="cross_behavior.task14_bracket_compiled_predictive_dispatcher_v3"
EXPECTED={PRIOR:"256ab1173bc392429c74ee1e0fa471e975c02b42dfdc994153761f03ba431125",TASK:"cce00d8f2309b0f9e0329c094238505819f2cf8a21c04e276af2085e068c1d07",READERS:"9db4eefe16498cb65fb9c21ea3f2475c790c89ebb2e65a70e8ad6b7886f2ae57",ABS:"c365a9d3e5fccc8ba8463099571c1fbdf059e49ed8426cf0e32dddec30644930",BROAD:"2540bc308c7e79b93b5054a0ef5355e1166625b27708f99276b184817d1e47ed",EFFECT:"6b8db79cc8c72500586a01966eb11c9d9cde89b35221f35b8ada0928d5c78bdf"}
OLD_COMPLETE_SCALARS=20742;NEW_SCALARS=14992
def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _load():
 for p,h in EXPECTED.items():
  if _sha(p)!=h:raise ValueError(f"immutable input changed: {p}")
 task,readers,absolute,broad,effect=(json.loads(p.read_text()) for p in (TASK,READERS,ABS,BROAD,EFFECT))
 if task.get("terminal")!="prototype_artifact" or readers.get("terminal")!="reader_artifact" or absolute.get("terminal")!="prototype_artifact" or broad.get("terminal")!="program_screen" or effect.get("terminal")!="predictive_screen":raise ValueError("source status invalid")
 return task,readers,absolute,effect
def compile_plan():
 _load();return {"schema":"task14_bracket_compiled_predictive_dispatcher_plan_v3","candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],"sources":{p.name:h for p,h in EXPECTED.items() if p!=PRIOR},"inventory":{"task14_displacement_coordinates":11520,"task14_precomputed_effects":10,"bracket_absolute_term_coordinates":3456,"bracket_predicted_effects":6,"total_fp32_scalars":NEW_SCALARS,"total_fp32_bytes":NEW_SCALARS*4},"old_complete_operational_scalars":OLD_COMPLETE_SCALARS,"storage_reduction_fraction":1-NEW_SCALARS/OLD_COMPLETE_SCALARS,"price":{"model_forwards":0,"example_evaluations":0,"backwards":0,"fits":0,"parameter_updates":0}}
def build(task,absolute,effect):
 task_program={k:{"displacement":v["coordinates"],"predicted_donorward_effect":v["frozen_reader_q"]} for k,v in task["prototypes"].items() if ".cardinality_" in k};bracket_effects=effect["plan"]["frozen_pair_scalars"]
 return {"schema":"task14_bracket_compiled_predictive_dispatcher_artifact_v3","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"api":{"task14":["recipient_number","donor_number","cardinality"],"bracket":["recipient_closer_id","donor_closer_id"]},"programs":{"task14":task_program,"bracket":{"absolute_terms":{k:v["coordinates"] for k,v in absolute["prototypes"].items()},"predicted_effects":bracket_effects}},"semantics":{"task14_edit":"add selected displacement to native L11H3 final-position term","bracket_edit":"replace L13H8 semantic-opener term by donor-closer absolute term","bracket_no_edit":"perform no intervention"},"residual_dependencies":{"effect_prediction":["external intervention specification"],"causal_execution":["model execution to installation site","native downstream execution"],"task14_only":["native L11H3 base term"],"bracket_only":["semantic opener position"]},"explicitly_not_provided":["full logits","whole native context","whole-model replacement"]}
def _rejects(fn):
 try:fn()
 except dispatcher.DispatchError:return True
 return False
def evaluate(package,task,readers,absolute,effect):
 entries=package["programs"]["task14"];max_dot_error=0.
 for key,item in entries.items():
  direction=key.split(".")[0];dot=sum(a*b for a,b in zip(item["displacement"],readers["readers"][direction]["coordinates"]));max_dot_error=max(max_dot_error,abs(dot-item["predicted_donorward_effect"]))
 task_calls=[dispatcher.dispatch_task14(package,recipient_number=r,donor_number=d,cardinality=c) for r,d in (("singular","plural"),("plural","singular")) for c in range(5)];task_ok=len(task_calls)==10 and {x["key"] for x in task_calls}==set(entries) and all(x["operation"]=="add_displacement" for x in task_calls) and _rejects(lambda:dispatcher.dispatch_task14(package,recipient_number="singular",donor_number="singular",cardinality=0))
 edits=[dispatcher.dispatch_bracket(package,recipient_closer_id=a,donor_closer_id=b) for a in dispatcher.CLOSERS for b in dispatcher.CLOSERS if a!=b];noedits=[dispatcher.dispatch_bracket(package,recipient_closer_id=a,donor_closer_id=a) for a in dispatcher.CLOSERS];bracket_ok=len(edits)==6 and all(x["operation"]=="replace_absolute" and x["predicted_donorward_effect"]==effect["plan"]["frozen_pair_scalars"][x["key"]] and x["vector"]==absolute["prototypes"][x["key"].split("->")[1]]["coordinates"] for x in edits) and all(x["operation"]=="no_edit" and x["vector"] is None and x["predicted_donorward_effect"]==0 for x in noedits) and _rejects(lambda:dispatcher.dispatch_bracket(package,recipient_closer_id=2,donor_closer_id=8))
 actual=sum(len(x["displacement"]) + 1 for x in entries.values())+sum(len(x) for x in package["programs"]["bracket"]["absolute_terms"].values())+len(package["programs"]["bracket"]["predicted_effects"]);price_ok=actual==NEW_SCALARS and NEW_SCALARS*4==59968 and abs((1-NEW_SCALARS/OLD_COMPLETE_SCALARS)-.27721531192749016)<1e-15
 boundary=package["residual_dependencies"]=={"effect_prediction":["external intervention specification"],"causal_execution":["model execution to installation site","native downstream execution"],"task14_only":["native L11H3 base term"],"bracket_only":["semantic opener position"]}
 pred={"pred_a_exact_selected_compilation":len(entries)==10 and len(package["programs"]["bracket"]["absolute_terms"])==3 and len(package["programs"]["bracket"]["predicted_effects"])==6,"pred_b_task14_reader_contraction":max_dot_error<=1e-7 and task_ok,"pred_c_bracket_absolute_and_no_edit_dispatch":bracket_ok,"pred_d_literal_smaller_price":price_ok,"pred_e_residual_boundary":boundary};return {"task14_max_reader_contraction_absolute_error":max_dot_error,"dispatch_cases":{"task14_edits":10,"bracket_edits":6,"bracket_no_edits":3},"storage":{"old_complete_operational_scalars":OLD_COMPLETE_SCALARS,"new_scalars":actual,"new_bytes":actual*4,"reduction_fraction":1-actual/OLD_COMPLETE_SCALARS},"residual_dependencies":package["residual_dependencies"],"classification":"predictive_manipulable_interface_not_whole_model","predictions":pred,"terminal":"screen" if all(pred.values()) else "null"}
def main():
 plan=compile_plan()
 if os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1" or "--dry-run" in sys.argv:print(json.dumps(plan,sort_keys=True));return
 if PACKAGE.exists() or OUT.exists():raise ValueError("refusing overwrite")
 task,readers,absolute,effect=_load();package=build(task,absolute,effect);package_bytes=managed.atomic_create_json(PACKAGE,package);score=evaluate(package,task,readers,absolute,effect);payload=managed.atomic_create_json(OUT,{"schema":"task14_bracket_compiled_predictive_dispatcher_result_v3","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"artifact_sha256":hashlib.sha256(package_bytes).hexdigest(),"score":score,"terminal":score["terminal"]});print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"artifact_sha256":hashlib.sha256(package_bytes).hexdigest(),"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()

