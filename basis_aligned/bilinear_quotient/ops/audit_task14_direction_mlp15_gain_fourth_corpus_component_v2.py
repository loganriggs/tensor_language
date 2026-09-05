#!/usr/bin/env python3
"""Promote only the preregistered prospective MLP15 component."""

# BQGATE: EXPERIMENT pred_a_immutable_registered_component pred_b_mlp15_prospective_transfer pred_c_both_unseen_templates_directional pred_d_joint_failure_preserved pred_e_zero_reopening_and_price
from __future__ import annotations
import argparse,hashlib,json,math,os
from datetime import datetime,timezone
from pathlib import Path
import circuit_fast_screen_managed_runner as managed

ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/"circuits/prior_art/task14_mlp6_7_direction_mlp15_gain_fourth_corpus_component_v2.json"
PARENT=ROOT/"circuits/followups/task14_direction_mediator_gain_fourth_corpus_causal_validation_v1_result.json"
OUT=ROOT/"circuits/followups/task14_direction_mlp15_gain_fourth_corpus_component_v2_result.json"
PRIOR_SHA="83320fca238bbacad7b538ce125923f6136f84a765d1b83f4f86687f700238bc"
PARENT_SHA="d27b0bb652a1f0405aa312fce848b8c15b841f95a4f0b55fdb0cc394ce8c3e0e"
CANDIDATE_ID="subject_verb.number_agreement.direction_mlp15_gain_fourth_corpus_component_v2"
class AuditError(ValueError):pass
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _load():
 if _sha(PRIOR)!=PRIOR_SHA or _sha(PARENT)!=PARENT_SHA:raise AuditError("immutable input changed")
 p=json.loads(PARENT.read_text())
 if p.get("terminal")!="null":raise AuditError("parent is not the expected null")
 return p
def compile_plan():
 _load();return {"schema":"task14_direction_mlp15_gain_fourth_corpus_component_audit_plan_v2","candidate_id":CANDIDATE_ID,"parent_sha256":PARENT_SHA,"prior_art_sha256":PRIOR_SHA,"only_promoted_prediction":"pred_c_mlp15_transfer","price":{"gpu_model_forwards":0,"causal_interventions":0,"backwards":0,"parameter_updates":0}}
def _stats(rows):
 a=[x["m15"] for x in rows];p=[x["sealed_m15"] for x in rows];an=math.sqrt(sum(x*x for x in a));pn=math.sqrt(sum(x*x for x in p));return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(a)}
def evaluate():
 p=_load();rows=p["score"]["joined_evidence"];groups={f"{d}/{t}":_stats([x for x in rows if x["direction"]==d and x["template"]==t]) for d in ("singular_to_plural","plural_to_singular") for t in ("under_beyond","beyond_under")};pred={"pred_a_immutable_registered_component":p["score"]["predictions"]["pred_a_authority_capability_seal_and_instrument"] and p["score"]["predictions"]["pred_f_exact_fixed_program_and_price"],"pred_b_mlp15_prospective_transfer":p["score"]["predictions"]["pred_c_mlp15_transfer"],"pred_c_both_unseen_templates_directional":all(x["cosine"]>0 and x["sign_agreement"]>=.65 for x in groups.values()),"pred_d_joint_failure_preserved":not p["score"]["predictions"]["pred_d_mlp17_and_interaction_transfer"] and not p["score"]["predictions"]["pred_e_joint_and_each_template_transfer"],"pred_e_zero_reopening_and_price":True};return {"parent_terminal":p["terminal"],"overall_mlp15":p["score"]["mlp15"],"mlp15_by_direction_template":groups,"rejected_components":["mlp17","interaction","joint_mediation"],"predictions":pred,"terminal":"screen" if all(pred.values()) else "null"}
def main(argv=None):
 q=argparse.ArgumentParser();q.add_argument("--dry-run",action="store_true");a=q.parse_args(argv);plan=compile_plan()
 if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1":print(json.dumps(plan,sort_keys=True));return
 if OUT.exists():raise AuditError(f"refusing overwrite {OUT}")
 score=evaluate();payload=managed.atomic_create_json(OUT,{"schema":"task14_direction_mlp15_gain_fourth_corpus_component_audit_result_v2","candidate_id":CANDIDATE_ID,"terminal":score["terminal"],"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score});print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__":main()
