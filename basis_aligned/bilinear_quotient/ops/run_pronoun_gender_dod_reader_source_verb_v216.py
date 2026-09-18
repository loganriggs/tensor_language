#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_pooled_verb_share_020 pred_c_9_6_verb_at_least_noun pred_d_10_1_is_a_noun_reader pred_e_token_only_small
"""Pronoun gender he/she DoD (v216): is the VERB site a family fact? v214 (number line): the readers take 33% of their coefficients from the verb position
(9.6 50%, 10.5 34%, 12.4 32%, 15.1 17%), where the MLP-8 detectors re-fire. Same exact source fold (v80's `fold_line`, categories noun / verb / object /
final / other) on the gender line's v71 rows ("The hero lost the compass and so" -> he / she) for the gender set {10.1, 9.6, 12.4, 15.1}. v80 found
10.1 a token-only noun reader (noun 98%, inherited 72%) and 9.6 / 12.4 / 15.1 contextual (noun 40-58%).
PREDICTIONS (scored as written; failures preserved; priors from v80 / v214)
    pred_a_fold_closure          sum of terms = captured v . w within relative 1e-3, every row and head
    pred_b_pooled_verb_share_020 pooled over the four heads (weighted by mean contrast) the verb carries >= 0.20
    pred_c_9_6_verb_at_least_noun  9.6: verb share >= 0.80 x noun share (on the number line 0.50 vs 0.48)
    pred_d_10_1_is_a_noun_reader 10.1: noun share >= 0.80 and verb share <= 0.10 (the token copier reads the noun only)
    pred_e_token_only_small      for 9.6, 12.4 and 15.1 the token-only branch is <= 0.20
PRICE (registered maximum): 3 batches x (capture + 4 block folds) = 15 forwards; 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80
import run_pronoun_gender_dod_battery_v71 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_gender_dod_reader_source_verb_v216_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_reader_source_verb_v216"
HEADS = ((10, 1), (9, 6), (12, 4), (15, 1))
CLOSURE_TOL, POOLED_MIN, RATIO_96, NOUN_101_MIN, VERB_101_MAX, INH_MAX = 1e-3, 0.20, 0.80, 0.80, 0.10, 0.20
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_pooled_verb_share_020": ">= 0.20", "pred_c_9_6_verb_at_least_noun": ">= 0.80 x noun", "pred_d_10_1_is_a_noun_reader": "noun >= 0.80, verb <= 0.10", "pred_e_token_only_small": "<= 0.20 x 3"}


def main() -> None:
    rows, they, he = g.build()          # they = he-token (positive), he = she-token (negative): v214's names kept for the shared body
    nouns = {L._single(" " + w) for p in g.PAIRS for w in p}
    objs = set()
    noun_pos = {row.row_id: next(i for i, t in enumerate(row.ids) if t in nouns) for row in rows}
    def category(row, s, nouns_, objs_):
        if s == row.final: return "final"
        if s == noun_pos[row.row_id]: return "noun"
        if s == noun_pos[row.row_id] + 1: return "verb"
        return "object" if row.ids[s] in objs_ else "other"
    v80.category = category; v80.CATEGORIES = ("noun", "verb", "object", "final", "other")
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(HEADS), "categories": list(v80.CATEGORIES), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "pooled_min": POOLED_MIN, "ratio_96": RATIO_96, "noun_101_min": NOUN_101_MIN, "verb_101_max": VERB_101_MAX, "inh_max": INH_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    result, forwards = v80.fold_line(fw, rows, they, he, HEADS, nouns, objs, "gender_verb")
    H = result["heads"]; tot = sum(r["mean_contrast"] for r in H.values())
    pooled_verb = sum(r["shares"]["verb"] * r["mean_contrast"] for r in H.values()) / tot
    print("pooled verb share", round(pooled_verb, 3), {k: {c: round(v, 3) for c, v in r["shares"].items()} for k, r in H.items()})
    predictions = {"pred_a_fold_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_pooled_verb_share_020": pooled_verb >= POOLED_MIN, "pred_c_9_6_verb_at_least_noun": H["9.6"]["shares"]["verb"] >= RATIO_96 * H["9.6"]["shares"]["noun"],
                   "pred_d_10_1_is_a_noun_reader": H["10.1"]["shares"]["noun"] >= NOUN_101_MIN and H["10.1"]["shares"]["verb"] <= VERB_101_MAX, "pred_e_token_only_small": all(H[k]["inherited_share"] <= INH_MAX for k in ("9.6", "12.4", "15.1"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_reader_source_verb_result_v216", "candidate_id": CANDIDATE_ID, "plan": plan, "line": result, "pooled_verb_share": pooled_verb, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
