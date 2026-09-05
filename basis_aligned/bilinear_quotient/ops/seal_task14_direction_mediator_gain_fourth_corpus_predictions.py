#!/usr/bin/env python3
"""Seal fourth-corpus fixed program and mediator predictions before outcomes."""

# BQGATE: EXPERIMENT pred_a_authority_and_license pred_b_complete_fixed_predictions pred_c_zero_target_outcomes
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_direction_mediator_gain_transfer as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_direction_mediator_gain_transfer_native_capability as capability
import run_task14_mlp6_7_direction_cardinality_prototype_causal_validation as program
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


ROOT=Path(__file__).resolve().parent.parent
PRIOR_ART=ROOT/"circuits/prior_art/task14_mlp6_7_direction_mediator_gain_fourth_corpus_transfer_v1.json"
GAINS=ROOT/"circuits/followups/task14_mlp6_7_direction_program_loo_mediator_gain_v2_result.json"
OUT=ROOT/"circuits/fast_screens/task14_direction_mediator_gain_fourth_corpus_predictions_v1.json"
PRIOR_ART_SHA256="ff633ae538629fe0a2cbded343fd0c1efab8511d8d617fee6d77577edb02cd53"
GAIN_SHA256="cb792d42cacd2fc88974c8e522fae3c3942c5dc4d35a9f0b4e6a54eab48d2bd5"
CAPABILITY_RESULT_SHA256="8e204febc3477ea1a6e8e914356e77ff05ea3eb527d8fb0dbd6e69e247af9733"
CAPABILITY_LICENSE_SHA256="f8dba12da6d6d9668017ae5687af64dd90868e5110c215ec2006d20ad01b2e5a"
SUBSETS=factor_gate.BACKGROUND_SUBSETS


class SealError(ValueError): pass
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_preflight():
    for path,expected,label in ((PRIOR_ART,PRIOR_ART_SHA256,"prior art"),(GAINS,GAIN_SHA256,"gains"),(program.PROTOTYPES,program.PROTOTYPE_SHA256,"prototypes"),(capability.RESULT,CAPABILITY_RESULT_SHA256,"capability"),(capability.LICENSE,CAPABILITY_LICENSE_SHA256,"license")):
        if _sha(path)!=expected: raise SealError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(),capability.RESULT,capability.LICENSE,expected_license_sha256=CAPABILITY_LICENSE_SHA256,causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)


def compile_plan():
    validate_preflight(); rows=authority.build_rows()
    return {"schema":"task14_direction_mediator_gain_fourth_corpus_prediction_plan_v1","candidate_id":authority.CAUSAL_CANDIDATE_ID,"row_count":len(rows),"background_subsets":list(SUBSETS),"prediction_count":len(rows)*len(SUBSETS),"prior_art_sha256":PRIOR_ART_SHA256,"gain_sha256":GAIN_SHA256,"prototype_sha256":program.PROTOTYPE_SHA256,"capability_license_sha256":CAPABILITY_LICENSE_SHA256,"predictions":{"pred_a_authority_and_license":"authority and candidate-scoped license validate","pred_b_complete_fixed_predictions":"exactly 512 fixed reader and three mediator-component predictions","pred_c_zero_target_outcomes":"no fourth-corpus model, causal, exact displacement, or mediator outcome is read"},"price":{"gpu_model_forwards":0,"causal_installations":0,"predictions":512,"backwards":0,"parameter_updates":0}}


def build_predictions():
    plan=compile_plan(); rows=authority.build_rows(); artifact=json.loads(program.PROTOTYPES.read_text()); gain=json.loads(GAINS.read_text())["score"]["exported_direction_gains"]
    evidence=[]
    for row in rows:
        direction=row["direction_id"]
        for subset in SUBSETS:
            q=float(artifact["prototypes"][f"{direction}.cardinality_{len(subset)}"]["frozen_reader_q"])
            m15=float(gain[f"m15/{direction}"])*q; m17=float(gain[f"m17/{direction}"])*q; interaction=float(gain[f"interaction/{direction}"])*q
            evidence.append({"row_id":row["row_id"],"direction":direction,"template":row["template_id"],"background":subset,"cardinality":len(subset),"sealed_reader_q":q,"sealed_m15":m15,"sealed_m17":m17,"sealed_interaction":interaction,"sealed_joint_mediation":m15+m17+interaction})
    if len(evidence)!=512 or len({(x["row_id"],x["background"]) for x in evidence})!=512: raise SealError("prediction coverage failed")
    return plan,evidence


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");a=p.parse_args(argv);plan=compile_plan()
    if a.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True));return
    if OUT.exists(): raise SealError(f"refusing overwrite {OUT}")
    plan,evidence=build_predictions(); payload=managed.atomic_create_json(OUT,{"schema":"task14_direction_mediator_gain_fourth_corpus_predictions_v1","candidate_id":authority.CAUSAL_CANDIDATE_ID,"terminal":"sealed_prediction","created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"fourth_corpus_causal_outcomes_opened":False,"fourth_corpus_model_forwards":0,"predictions":{"pred_a_authority_and_license":True,"pred_b_complete_fixed_predictions":True,"pred_c_zero_target_outcomes":True},"evidence":evidence})
    print(json.dumps({"terminal":"sealed_prediction","prediction_count":len(evidence),"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
