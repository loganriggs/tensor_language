#!/usr/bin/env python3
"""Immutable one-sided audit of Task14 under live bracket-program stress."""

# BQGATE: EXPERIMENT pred_a_parent_instrument_and_price pred_b_bracket_stress_is_live_on_task14 pred_c_task14_composes_under_bracket_stress pred_d_no_reverse_or_semantic_upgrade pred_e_zero_execution_price
# BQLANE: cpu
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
import circuit_fast_screen_managed_runner as managed

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/"circuits/prior_art/task14_under_bracket_program_stress_composition_v2.json"
PARENT=ROOT/"circuits/followups/task14_bracket_fixed_program_stress_composition_v1_result.json"
OUT=ROOT/"circuits/followups/task14_under_bracket_program_stress_composition_v2_result.json"
PRIOR_SHA="b4151cdf8c6996f489d193b4a302e703aa501b1a41f0fd4275ffc9a5ecf9b48b"
PARENT_SHA="d6df30a66b50ce97745df382bda3af6e7d0f6f66a1636e57bf30850dd7bf0d74"
CANDIDATE_ID="cross_behavior.task14_under_bracket_program_stress_composition_v2"

def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def _load():
    if _sha(PRIOR)!=PRIOR_SHA or _sha(PARENT)!=PARENT_SHA: raise ValueError("immutable input changed")
    value=json.loads(PARENT.read_text())
    if value.get("terminal")!="inconclusive" or value["score"]["predictions"]["pred_b_foreign_stress_is_live"] is not False:
        raise ValueError("parent is not the registered asymmetric result")
    return value
def compile_plan():
    _load(); return {"schema":"task14_under_bracket_program_stress_composition_audit_plan_v2","candidate_id":CANDIDATE_ID,"prior_art_sha256":PRIOR_SHA,"parent_sha256":PARENT_SHA,"only_promoted_panel":"task14 under bracket-program stress","excluded_panel":"bracket under Task14 stress","changes":{"vectors":0,"rows":0,"assignments":0,"bars":0,"outcomes":0},"price":{"model_forwards":0,"example_evaluations":0,"backwards":0,"fits":0,"parameter_updates":0,"interventions":0}}
def evaluate():
    parent=_load(); score=parent["score"]; task=score["task14"]; p=task["preservation"]
    pred={
      "pred_a_parent_instrument_and_price":score["predictions"]["pred_a_immutable_programs_and_exact_corners"] and score["predictions"]["pred_e_fixed_price_and_no_refit"],
      "pred_b_bracket_stress_is_live_on_task14":task["foreign_stress_to_own_norm_ratio"]>=.05,
      "pred_c_task14_composes_under_bracket_stress":score["predictions"]["pred_c_task14_survives_bracket_program"] and p["cosine"]>=.90 and p["relative_l2_error"]<=.40 and p["sign_agreement"]>=.90 and task["interaction_to_own_norm_ratio"]<=.40,
      "pred_d_no_reverse_or_semantic_upgrade":True,
      "pred_e_zero_execution_price":True,
    }
    return {"task14":task,"parent_bracket_stress_to_own_norm_ratio":score["bracket"]["foreign_stress_to_own_norm_ratio"],"exported_claim":"Task14 program composes under deterministic live bracket-program stress","explicitly_not_exported":["reverse composition","two-sided composition","joint semantic instantiation"],"predictions":pred,"terminal":"screen" if all(pred.values()) else "null"}
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--dry-run",action="store_true");args=parser.parse_args(argv);plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True));return
    if OUT.exists(): raise ValueError(f"refusing overwrite {OUT}")
    score=evaluate();payload=managed.atomic_create_json(OUT,{"schema":"task14_under_bracket_program_stress_composition_audit_result_v2","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score,"terminal":score["terminal"]});print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))
if __name__=="__main__": main()
