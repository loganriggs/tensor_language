#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_head_8_1_is_a_token_only_pronoun_reader pred_c_pronoun_position_is_largest_source_for_every_head pred_d_pooled_inherited_share_at_most_050 pred_e_contrast_positive_for_every_head
"""Reflexive person DoD (v112): SOURCE FOLD of the four readout coefficients at {8.1, 13.1, 10.5, 15.1} on the v104 fresh rows (v80's exact fold).

Lane: Claude circuit lane. Parents: v104 (42% on fresh rows), v107 (85% direct), v110/v111 (the set tracks the cue token on natural rows).
Per head, c_h = v_h . z_h(final) split by source category -- the I / you token ("noun" slot of v80's category function), the object noun,
the final token, other -- and by value branch (current block state vs token-only block-0 value). Registered reading: 8.1 is a token-only
reader of the pronoun token here, as it is of since/by (temporal) and as 10.1 is of gendered nouns.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                                     sum of terms = captured v . w within relative 1e-3, every row and head
    pred_b_head_8_1_is_a_token_only_pronoun_reader          8.1: pronoun-position share >= 0.80 and inherited (token-only) share >= 0.50
    pred_c_pronoun_position_is_largest_source_for_every_head  the pronoun position carries the largest |share| for each of the four heads. Prior: unsure.
    pred_d_pooled_inherited_share_at_most_050               pooled over the four heads the token-only branch is <= 0.50. Prior: unsure.
    pred_e_contrast_positive_for_every_head                 mean oriented contrast > 0 for all four heads

PRICE (registered maximum): 3 batches x (capture + 4 block folds) = 15 forwards; 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80
import run_person_reflexive_dod_battery_v104 as line

OUT = v80.ROOT / "circuits/followups/person_dod_source_fold_v112_result.json"
CANDIDATE_ID = "reflexive_person.i_vs_you.dod_source_fold_v112"
CLOSURE_TOL, TOKEN_MIN, INHERITED_81_MIN, INHERITED_MAX = 1e-3, 0.80, 0.50, 0.50
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_head_8_1_is_a_token_only_pronoun_reader": ">= 0.80 pronoun, >= 0.50 inherited",
               "pred_c_pronoun_position_is_largest_source_for_every_head": "pronoun top x 4", "pred_d_pooled_inherited_share_at_most_050": "<= 0.50", "pred_e_contrast_positive_for_every_head": "> 0 x 4"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    pronouns = {L._single(" I"), L._single("I"), L._single(" you"), L._single("you")}
    objs = {L._single(" " + o) for o in objects}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(line.HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "token_min": TOKEN_MIN, "inherited_81_min": INHERITED_81_MIN, "inherited_max": INHERITED_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    result, forwards = v80.fold_line(fw, rows, pos, neg, line.HEADS, pronouns, objs, "person")
    H = result["heads"]
    b81 = H["8.1"]["shares"]["noun"] >= TOKEN_MIN and H["8.1"]["inherited_share"] >= INHERITED_81_MIN
    predictions = {"pred_a_fold_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_head_8_1_is_a_token_only_pronoun_reader": bool(b81),
                   "pred_c_pronoun_position_is_largest_source_for_every_head": all(h["largest_source"] == "noun" for h in H.values()),
                   "pred_d_pooled_inherited_share_at_most_050": result["pooled_inherited_share"] <= INHERITED_MAX, "pred_e_contrast_positive_for_every_head": all(h["mean_contrast"] > 0 for h in H.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "person_dod_source_fold_result_v112", "candidate_id": CANDIDATE_ID, "plan": plan, "line": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled": {"pronoun": round(result["pooled_noun_share"], 3), "inherited": round(result["pooled_inherited_share"], 3)}, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
