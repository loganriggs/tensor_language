#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_10_5_reads_the_verb pred_c_verb_beats_noun_for_10_5 pred_d_12_4_ignores_the_verb pred_e_pooled_verb_share_020
"""Pronoun number they/he DoD (v214): do the readers read the VERB? v207: zeroing head 4.5 at the verb costs 10.5 45% of the summed reader-coefficient drop,
9.6 and 15.1 the rest, 12.4 nothing. v80's exact source fold of the four readout coefficients c_h = v_h . z_h(final) on the v76 rows split the sources into
noun / object / final / other; here "other" is split further so the VERB (noun + 1) is its own category: noun / verb / object / final / other, each by value
branch (current vs token-only). Same fold object (`run_pronoun_dod_source_fold_v80.fold_line` with the category function overridden).
PREDICTIONS (scored as written; failures preserved; priors from v80 and v207)
    pred_a_fold_closure            sum of terms = captured v . w within relative 1e-3, every row and head
    pred_b_10_5_reads_the_verb     10.5: verb-position share >= 0.25 of its coefficient
    pred_c_verb_beats_noun_for_10_5  10.5: verb share > noun share (v80: noun 0.315, other 0.516)
    pred_d_12_4_ignores_the_verb   12.4: verb share <= 0.10 (v207: 12.4 lost nothing when 4.5 was zeroed at the verb)
    pred_e_pooled_verb_share_020   pooled over the four heads (weighted by mean contrast) the verb carries >= 0.20
PRICE (registered maximum): 3 batches x (capture + 4 block folds) = 15 forwards; 0 backwards; 0 fits. Bar <= 18.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_dod_source_fold_v80 as v80
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_reader_source_verb_v214_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_reader_source_verb_v214"
HEADS = ((9, 6), (12, 4), (15, 1), (10, 5))
CLOSURE_TOL, VERB_105_MIN, VERB_124_MAX, POOLED_MIN = 1e-3, 0.25, 0.10, 0.20
FORWARDS_MAX = 18
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_10_5_reads_the_verb": ">= 0.25", "pred_c_verb_beats_noun_for_10_5": "verb > noun", "pred_d_12_4_ignores_the_verb": "<= 0.10", "pred_e_pooled_verb_share_020": ">= 0.20"}


def main() -> None:
    rows, they, he, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    objs = {L._single(" " + o) for o in objects}
    noun_pos = {row.row_id: next(i for i, t in enumerate(row.ids) if t in nouns) for row in rows}
    def category(row, s, nouns_, objs_):
        if s == row.final: return "final"
        if s == noun_pos[row.row_id]: return "noun"
        if s == noun_pos[row.row_id] + 1: return "verb"
        return "object" if row.ids[s] in objs_ else "other"
    v80.category = category; v80.CATEGORIES = ("noun", "verb", "object", "final", "other")
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(HEADS), "categories": list(v80.CATEGORIES), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "verb_105_min": VERB_105_MIN, "verb_124_max": VERB_124_MAX, "pooled_min": POOLED_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    result, forwards = v80.fold_line(fw, rows, they, he, HEADS, nouns, objs, "number_verb")
    H = result["heads"]; tot = sum(r["mean_contrast"] for r in H.values())
    pooled_verb = sum(r["shares"]["verb"] * r["mean_contrast"] for r in H.values()) / tot
    print("pooled verb share", round(pooled_verb, 3), {k: {c: round(v, 3) for c, v in r["shares"].items()} for k, r in H.items()})
    predictions = {"pred_a_fold_closure": result["closure_max"] <= CLOSURE_TOL, "pred_b_10_5_reads_the_verb": H["10.5"]["shares"]["verb"] >= VERB_105_MIN, "pred_c_verb_beats_noun_for_10_5": H["10.5"]["shares"]["verb"] > H["10.5"]["shares"]["noun"],
                   "pred_d_12_4_ignores_the_verb": H["12.4"]["shares"]["verb"] <= VERB_124_MAX, "pred_e_pooled_verb_share_020": pooled_verb >= POOLED_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_reader_source_verb_result_v214", "candidate_id": CANDIDATE_ID, "plan": plan, "line": result, "pooled_verb_share": pooled_verb, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
