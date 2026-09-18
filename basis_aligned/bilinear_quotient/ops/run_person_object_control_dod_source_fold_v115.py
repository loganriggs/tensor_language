#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_pronoun_position_is_largest_source_for_every_head pred_c_token_readers_are_token_only pred_d_pooled_inherited_share_at_least_050 pred_e_contrast_positive_for_every_head
"""Reflexive object control DoD (v115): SOURCE FOLD of {13.1, 8.1, 10.5, 15.1} on the v105 fresh rows (v80's exact fold), the person
family's second line. Parent: v112 (on the subject-antecedent line 8.1 / 13.1 / 15.1 are token-only readers of the I / you token; 10.5
contextual). Here the antecedent is an OBJECT pronoun (me / you) inside "told me to trust": same heads, different token.
PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                                     <= 1e-3
    pred_b_pronoun_position_is_largest_source_for_every_head  the me / you position carries the largest |share| for each of the four heads
    pred_c_token_readers_are_token_only                     8.1 and 13.1: pronoun share >= 0.80 and inherited share >= 0.50 (as on v112)
    pred_d_pooled_inherited_share_at_least_050              pooled token-only branch >= 0.50 (registered the way v112 came out)
    pred_e_contrast_positive_for_every_head                 > 0 for all four
PRICE (registered maximum): 3 batches x (capture + 4 block folds) = 15 forwards; 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80
import run_person_object_control_dod_battery_v105 as line

OUT = v80.ROOT / "circuits/followups/person_object_control_dod_source_fold_v115_result.json"
CANDIDATE_ID = "reflexive_object_control.me_vs_you.dod_source_fold_v115"
CLOSURE_TOL, TOKEN_MIN, INH_MIN, POOLED_MIN = 1e-3, 0.80, 0.50, 0.50
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_pronoun_position_is_largest_source_for_every_head": "pronoun top x 4", "pred_c_token_readers_are_token_only": "8.1, 13.1 >= 0.80 / 0.50",
               "pred_d_pooled_inherited_share_at_least_050": ">= 0.50", "pred_e_contrast_positive_for_every_head": "> 0 x 4"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    pronouns = {L._single(" me"), L._single(" you")}
    objs = {L._single(" " + o) for o in objects}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(line.HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "token_min": TOKEN_MIN, "inh_min": INH_MIN, "pooled_min": POOLED_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    result, forwards = v80.fold_line(fw, rows, pos, neg, line.HEADS, pronouns, objs, "object_control")
    H = result["heads"]
    predictions = {"pred_a_fold_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_pronoun_position_is_largest_source_for_every_head": all(h["largest_source"] == "noun" for h in H.values()),
                   "pred_c_token_readers_are_token_only": all(H[k]["shares"]["noun"] >= TOKEN_MIN and H[k]["inherited_share"] >= INH_MIN for k in ("8.1", "13.1")),
                   "pred_d_pooled_inherited_share_at_least_050": result["pooled_inherited_share"] >= POOLED_MIN, "pred_e_contrast_positive_for_every_head": all(h["mean_contrast"] > 0 for h in H.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "person_object_control_dod_source_fold_result_v115", "candidate_id": CANDIDATE_ID, "plan": plan, "line": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled": {"pronoun": round(result["pooled_noun_share"], 3), "inherited": round(result["pooled_inherited_share"], 3)}, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
