#!/usr/bin/env python3
"""Promote the preregistered six-scalar direction-only mediator control."""

# BQGATE: EXPERIMENT pred_a_immutable_single_reduction pred_b_all_components_predictive pred_c_joint_and_each_group_predictive pred_d_cardinality_rejected pred_e_no_reopening_and_price
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_direction_program_loo_mediator_gain_v2.json"
V1 = ROOT / "circuits/followups/task14_mlp6_7_direction_cardinality_program_loo_mediator_gain_v1_result.json"
OUT = ROOT / "circuits/followups/task14_mlp6_7_direction_program_loo_mediator_gain_v2_result.json"
PRIOR_ART_SHA256 = "afc2e9d14e3275d18ffd07db0613e9460e84c786667742d033b3116a7ed1c643"
V1_SHA256 = "1d2d48143eb115bea931086fa59dbeb0fac3a24d857ed255891a6117138e9a72"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_direction_program_loo_mediator_gain_v2"
COMPONENTS = ("m15", "m17", "interaction")


class DirectionGainError(ValueError): pass


def _sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(rows, actual, predicted):
    a=[x[actual] for x in rows]; p=[x[predicted] for x in rows]
    an=math.sqrt(sum(x*x for x in a)); pn=math.sqrt(sum(x*x for x in p))
    return {"count":len(rows),"cosine":sum(x*y for x,y in zip(a,p))/max(an*pn,1e-30),"relative_l2_error":math.sqrt(sum((x-y)**2 for x,y in zip(a,p)))/max(an,1e-30),"sign_agreement":sum((x>0)==(y>0) for x,y in zip(a,p))/len(a),"sse":sum((x-y)**2 for x,y in zip(a,p))}


def _load():
    if _sha256(PRIOR_ART)!=PRIOR_ART_SHA256 or _sha256(V1)!=V1_SHA256: raise DirectionGainError("immutable input changed")
    v1=json.loads(V1.read_text())
    if v1.get("terminal")!="inconclusive" or v1["score"]["predictions"]["pred_e_cardinality_is_needed"] is not False: raise DirectionGainError("v1 does not license simplicity reduction")
    return v1


def compile_plan():
    _load()
    return {"schema":"task14_mlp6_7_direction_program_loo_mediator_gain_audit_plan_v2","candidate_id":CANDIDATE_ID,"v1_sha256":V1_SHA256,"prior_art_sha256":PRIOR_ART_SHA256,"only_change":"promote preregistered direction-only control and export six full-data gains","stored_scalar_count":6,"price":{"gpu_model_forwards":0,"causal_installations":0,"backwards":0,"parameter_updates":0,"immutable_receipts_read":1}}


def evaluate():
    v1=_load(); rows=v1["evidence"]
    component={c:_stats(rows,c,f"control_pred_{c}") for c in COMPONENTS}
    joint=_stats(rows,"m_both","control_pred_joint")
    groups={f"{d}/{t}":_stats([x for x in rows if x["direction"]==d and x["template"]==t],"m_both","control_pred_joint") for d in ("singular_to_plural","plural_to_singular") for t in ("near_beyond","beyond_near")}
    gains={f"{c}/{d}":sum(x["reader_q"]*x[c] for x in rows if x["direction"]==d)/sum(x["reader_q"]**2 for x in rows if x["direction"]==d) for c in COMPONENTS for d in ("singular_to_plural","plural_to_singular")}
    bars=v1["plan"]["bars"]
    pred_b=component["m15"]["cosine"]>=bars["m15_min_cosine"] and component["m15"]["relative_l2_error"]<=bars["m15_max_relative_l2"] and component["m15"]["sign_agreement"]>=bars["m15_min_sign"] and component["m17"]["cosine"]>=bars["m17_min_cosine"] and component["m17"]["relative_l2_error"]<=bars["m17_max_relative_l2"] and component["m17"]["sign_agreement"]>=bars["m17_min_sign"] and component["interaction"]["cosine"]>=bars["m17_min_cosine"] and component["interaction"]["relative_l2_error"]<=bars["m17_max_relative_l2"] and component["interaction"]["sign_agreement"]>=bars["m17_min_sign"]
    pred_c=joint["cosine"]>=bars["joint_min_cosine"] and joint["relative_l2_error"]<=bars["joint_max_relative_l2"] and joint["sign_agreement"]>=bars["joint_min_sign"] and all(x["cosine"]>=bars["group_joint_min_cosine"] for x in groups.values())
    reduction=v1["score"]["cardinality_sse_reduction_over_direction_only"]
    predictions={"pred_a_immutable_single_reduction":True,"pred_b_all_components_predictive":pred_b,"pred_c_joint_and_each_group_predictive":pred_c,"pred_d_cardinality_rejected":reduction<bars["minimum_sse_reduction"],"pred_e_no_reopening_and_price":True}
    return {"component_stats":component,"joint_mediation":joint,"joint_by_direction_template":groups,"cardinality_sse_reduction":reduction,"exported_direction_gains":gains,"exported_scalar_count":len(gains),"predictions":predictions,"terminal":"screen" if all(predictions.values()) else "null"}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);plan=compile_plan()
    if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True));return
    if OUT.exists(): raise DirectionGainError(f"refusing overwrite {OUT}")
    score=evaluate();payload=managed.atomic_create_json(OUT,{"schema":"task14_mlp6_7_direction_program_loo_mediator_gain_audit_result_v2","candidate_id":CANDIDATE_ID,"terminal":score["terminal"],"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score})
    print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
