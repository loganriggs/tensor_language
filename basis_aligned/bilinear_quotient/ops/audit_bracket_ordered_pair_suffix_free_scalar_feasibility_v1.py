#!/usr/bin/env python3
"""Cross-family suffix-free six-scalar feasibility audit for the bracket program."""

# BQGATE: EXPERIMENT pred_a_immutable_balanced_cross_family_instrument pred_b_suffix_free_transfer_overall pred_c_each_direction_transfers pred_d_six_scalar_price pred_e_scope_is_diagnostic
# BQLANE: cpu
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/bracket_ordered_pair_suffix_free_scalar_feasibility_v1.json"
PROGRAM = ROOT / "circuits/followups/bracket_l13h8_ordered_pair_displacement_program_ood_validation_v1_result.json"
FOLD = ROOT / "circuits/fast_screens/bracket_l13h8_mu_delta_direct_readout_fold_v2_result.json"
OUT = ROOT / "circuits/followups/bracket_ordered_pair_suffix_free_scalar_feasibility_v1_result.json"
CANDIDATE_ID = "bracket.pending_opener.ordered_pair_suffix_free_scalar_feasibility_v1"
EXPECTED = {
    PRIOR: "acd064559e60f770b87572ec9e8db0c98a8c717a87ea08a64cb6ac56e4e88853",
    PROGRAM: "3b267f069647824fb7557e9784c63becb0366f94fe4d274fea343ae2bc802e5f",
    FOLD: "31a40ed62409181e7977ca4182f34dbd5d7d9fa0d92c3270888bae53a94a21ae",
}
BARS = {"minimum_overall_cosine": .80, "maximum_overall_relative_l2": .60,
        "minimum_overall_norm_ratio": .50, "maximum_overall_norm_ratio": 1.50,
        "minimum_overall_sign_agreement": .90, "minimum_fold_cosine": .70,
        "minimum_fold_sign_agreement": .85}


def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _load():
    for path, expected in EXPECTED.items():
        if _sha(path) != expected: raise ValueError(f"immutable input changed: {path}")
    program, fold = json.loads(PROGRAM.read_text()), json.loads(FOLD.read_text())
    if program.get("terminal") != "program_screen" or not all(program["score"]["predictions"].values()):
        raise ValueError("program validation is not a passing immutable screen")
    if fold.get("terminal") != "screen" or not fold["predictions"]["pred_a_instrument_live"]:
        raise ValueError("readout-fold instrument is not valid")
    return program, fold


def compile_plan():
    _load()
    return {"schema":"bracket_ordered_pair_suffix_free_scalar_feasibility_plan_v1",
            "candidate_id":CANDIDATE_ID,"prior_art_sha256":EXPECTED[PRIOR],
            "program_sha256":EXPECTED[PROGRAM],"readout_fold_sha256":EXPECTED[FOLD],
            "method":"bidirectional leave-target-family-out ordered-pair means",
            "bars":dict(BARS),"stored_fp32_scalars":6,"stored_fp32_bytes":24,
            "price":{"model_forwards":0,"example_evaluations":0,"backwards":0,"fits":0,"parameter_updates":0}}


def _metrics(rows):
    predicted = [row["predicted"] for row in rows]; actual = [row["actual"] for row in rows]
    dot = sum(a*b for a,b in zip(predicted,actual)); pn = math.sqrt(sum(x*x for x in predicted)); an = math.sqrt(sum(x*x for x in actual))
    error = math.sqrt(sum((a-b)**2 for a,b in zip(predicted,actual)))
    return {"count":len(rows),"cosine":dot/(pn*an),"relative_l2_error":error/an,
            "predicted_to_actual_norm_ratio":pn/an,
            "sign_agreement":sum((a>0)==(b>0) for a,b in zip(predicted,actual))/len(rows)}


def evaluate():
    program, _ = _load()
    targets = [row for row in program["evidence"] if row["program_role"] == "target"]
    families = sorted({row["family_id"] for row in targets}); pairs = sorted({row["ordered_pair"] for row in targets})
    counts = Counter((row["family_id"],row["ordered_pair"]) for row in targets)
    balanced = len(targets)==144 and len(families)==2 and len(pairs)==6 and set(counts.values())=={12}
    tables = {}
    evidence = []
    if balanced:
        for held in families:
            train = families[1-families.index(held)]
            table = {pair:statistics.mean(row["program_donorward_effect"] for row in targets if row["family_id"]==train and row["ordered_pair"]==pair) for pair in pairs}
            tables[f"{train}->{held}"] = table
            evidence.extend({"row_id":row["row_id"],"side":row["side"],"ordered_pair":row["ordered_pair"],"train_family":train,"held_family":held,"predicted":table[row["ordered_pair"]],"actual":row["program_donorward_effect"]} for row in targets if row["family_id"]==held)
    overall = _metrics(evidence) if evidence else {}
    by_fold = {key:_metrics([row for row in evidence if f'{row["train_family"]}->{row["held_family"]}'==key]) for key in tables}
    p2 = bool(evidence) and overall["cosine"]>=BARS["minimum_overall_cosine"] and overall["relative_l2_error"]<=BARS["maximum_overall_relative_l2"] and BARS["minimum_overall_norm_ratio"]<=overall["predicted_to_actual_norm_ratio"]<=BARS["maximum_overall_norm_ratio"] and overall["sign_agreement"]>=BARS["minimum_overall_sign_agreement"]
    p3 = len(by_fold)==2 and all(value["cosine"]>=BARS["minimum_fold_cosine"] and value["sign_agreement"]>=BARS["minimum_fold_sign_agreement"] for value in by_fold.values())
    predictions={"pred_a_immutable_balanced_cross_family_instrument":balanced,
                 "pred_b_suffix_free_transfer_overall":balanced and p2,
                 "pred_c_each_direction_transfers":balanced and p3,
                 "pred_d_six_scalar_price":True,
                 "pred_e_scope_is_diagnostic":True}
    terminal="feasibility_screen" if all(predictions.values()) else "null" if balanced else "invalid"
    return {"families":families,"ordered_pairs":pairs,"cross_family_scalar_tables":tables,"overall":overall,"by_fold":by_fold,
            "classification":"retrospective_cross_construction_feasibility_only",
            "explicitly_requires_next":"fresh-corpus prospective validation before predictive-program promotion",
            "predictions":predictions,"terminal":terminal}, evidence


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument("--dry-run",action="store_true");args=parser.parse_args(argv);plan=compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN")=="1" or os.environ.get("BQLIB_NO_MODEL")=="1": print(json.dumps(plan,sort_keys=True));return
    if OUT.exists(): raise ValueError(f"refusing overwrite {OUT}")
    score,evidence=evaluate();payload=managed.atomic_create_json(OUT,{"schema":"bracket_ordered_pair_suffix_free_scalar_feasibility_result_v1","candidate_id":CANDIDATE_ID,"created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"plan":plan,"score":score,"evidence":evidence,"terminal":score["terminal"]});print(json.dumps({"terminal":score["terminal"],"predictions":score["predictions"],"result_sha256":hashlib.sha256(payload).hexdigest()},sort_keys=True))


if __name__=="__main__": main()
