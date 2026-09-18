#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_head_8_1_is_a_token_only_cue_reader pred_c_cue_position_is_largest_source_for_every_head pred_d_pooled_inherited_share_at_least_050 pred_e_contrast_positive_for_every_head
"""Correlative both/neither DoD (v130): SOURCE FOLD of the correlative set on its fresh rows (v80's exact fold). Parents: the correlative battery, v112 / v115 (the
person family's 8.1 / 13.1 / 15.1 are token-only readers of the pronoun token). Categories: the cue token ("both", "neither"; the "noun" slot of v80's
category function), the object noun, the final token, other; branches current vs token-only (block-0 value). Registered reading: 8.1 is a
token-only reader of the correlative's first element, as it is of since / by and of I / you; the other three heads are the open question.
PREDICTIONS (scored as written; failures preserved): pred_a closure <= 1e-3; pred_b 8.1: cue-position share >= 0.80 and token-only share >= 0.50;
pred_c the cue position is the largest |share| for each of the four heads (prior: unsure); pred_d pooled token-only >= 0.50 (prior: unsure);
pred_e mean oriented contrast > 0 for all four heads.
PRICE (registered maximum): 3 batches x (capture + 4 block folds) = 15 forwards; 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80
import run_correlative_both_neither_dod_battery_v120 as line

OUT = v80.ROOT / "circuits/followups/correlative_both_neither_dod_source_fold_v130_result.json"
CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_source_fold_v130"
CUES = ("both", "neither",)
CLOSURE_TOL, TOKEN_MIN, INH_MIN, POOLED_MIN = 1e-3, 0.80, 0.50, 0.50
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_head_8_1_is_a_token_only_cue_reader": ">= 0.80 cue, >= 0.50 inherited", "pred_c_cue_position_is_largest_source_for_every_head": "cue top x 4",
               "pred_d_pooled_inherited_share_at_least_050": ">= 0.50", "pred_e_contrast_positive_for_every_head": "> 0 x 4"}


def main() -> None:
    rows, pos, neg, agents, objects = line.build()
    cues = {L._single(" " + c) for c in CUES}
    objs = {L._single(" " + o) for o in objects}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(line.HEADS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "token_min": TOKEN_MIN, "inh_min": INH_MIN, "pooled_min": POOLED_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    result, forwards = v80.fold_line(fw, rows, pos, neg, line.HEADS, cues, objs, "correlative_both_neither")
    H = result["heads"]
    predictions = {"pred_a_fold_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_head_8_1_is_a_token_only_cue_reader": bool(H["8.1"]["shares"]["noun"] >= TOKEN_MIN and H["8.1"]["inherited_share"] >= INH_MIN),
                   "pred_c_cue_position_is_largest_source_for_every_head": all(h["largest_source"] == "noun" for h in H.values()),
                   "pred_d_pooled_inherited_share_at_least_050": result["pooled_inherited_share"] >= POOLED_MIN, "pred_e_contrast_positive_for_every_head": all(h["mean_contrast"] > 0 for h in H.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "correlative_both_neither_dod_source_fold_result_v130", "candidate_id": CANDIDATE_ID, "plan": plan, "line": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled": {"cue": round(result["pooled_noun_share"], 3), "inherited": round(result["pooled_inherited_share"], 3)}, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
