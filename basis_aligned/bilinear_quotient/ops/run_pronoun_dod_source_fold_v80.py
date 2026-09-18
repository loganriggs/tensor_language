#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_contrast_positive_for_every_head pred_c_gender_noun_is_largest_source_for_every_head pred_d_number_noun_share_below_gender pred_e_inherited_branch_share_at_most_030
"""Pronoun gender + number DoD (v80): SOURCE FOLD of the readout coefficients (better_circuits §3.8: close ports by folding).

Lane: Claude circuit lane. Parents: gender v71-v75 (set {10.1, 9.6, 12.4, 15.1}), number v76-v79 (set {9.6, 12.4, 15.1, 10.5}).
The counter-cases differed: gender's incongruent natural rows moved toward the text under removal (the set tracks the noun),
number's moved away twice (the set carries a resolved number). This fold asks WHERE each head's readout coefficient
c_h = v_h . z_h(final) comes from, exactly: per source position s, p_h(final, s) [(1 - lamb) (V_h^T v_h) . x_s + lamb v_h . v1_h(s)],
on the fresh rows of each line, split by source category -- noun (the gendered / number-marked noun), object, final token,
other -- and by value branch (current block-l state vs token-only block-0 value). Oriented contrast: c_h(positive row) -
c_h(negative row) over aligned pairs (same construction, group). Evidence tag: fold, fresh rows (opened by v71/v76 edits).

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                                sum of terms = captured v_h . w_h within relative 1e-3, every row and head
    pred_b_contrast_positive_for_every_head            mean oriented contrast > 0 for all 8 (line, head) cells
    pred_c_gender_noun_is_largest_source_for_every_head  gender line: the noun category carries the largest |share| for each of the 4 heads
    pred_d_number_noun_share_below_gender              pooled noun share (number line) < pooled noun share (gender line). Prior: unsure.
    pred_e_inherited_branch_share_at_most_030          pooled token-only (v1) branch share <= 0.30 on both lines (readout heads read
                                                       contextual state, unlike the temporal cue reader 8.1). Prior: unsure.

PRICE (registered maximum): gender 2 batches x (capture + 4 block folds) + number 3 batches x (capture + 4) = 25 forwards; 0
backwards; 0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_gender_dod_battery_v71 as v71
import run_pronoun_number_dod_battery_v76 as v76
import dod_battery

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/pronoun_dod_source_fold_v80_result.json"
CANDIDATE_ID = "pronoun.gender_and_number.dod_source_fold_v80"
CLOSURE_TOL, INHERITED_MAX = 1e-3, 0.30
FORWARDS_MAX = 30
CATEGORIES = ("noun", "object", "final", "other")
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_contrast_positive_for_every_head": "> 0 x 8", "pred_c_gender_noun_is_largest_source_for_every_head": "noun top x 4",
               "pred_d_number_noun_share_below_gender": "number < gender", "pred_e_inherited_branch_share_at_most_030": "<= 0.30 both"}


def lines():
    g_rows, he, she = v71.build()
    g_nouns = {L._single(" " + w) for p in v71.PAIRS for w in p}; g_objs = {L._single(" " + o) for o in v71.OBJECTS}
    n_rows, they, he2, agents, objects = v76.build()
    n_nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}; n_objs = {L._single(" " + o) for o in objects}
    return {"gender": (g_rows, he, she, v71.HEADS, g_nouns, g_objs), "number": (n_rows, they, he2, v76.HEADS, n_nouns, n_objs)}


def category(row, s, nouns, objs):
    if s == row.final:
        return "final"
    tid = row.ids[s]
    return "noun" if tid in nouns else "object" if tid in objs else "other"


def fold_line(fw, rows, pos, neg, heads, nouns, objs, candidate):
    comps = dod_battery.LineSpec(candidate, "x", rows, pos, neg, heads).set_components()
    fw.directions = L.readout_directions(fw.backend.model, comps, pos, neg)
    forwards, store, terms = 0, {}, [dict() for _ in rows]
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        store.update(fw.capture(chunk, comps)); forwards += 1
        for comp in comps:
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            for j, entry in enumerate(out):
                for head, val in entry.items():
                    terms[start + j][f"{comp.layer}.{head}"] = (comp, head, val)
    closure_max = 0.0
    for row, t in zip(rows, terms):
        for key, (comp, head, val) in t.items():
            w = store[(row.row_id, comp.name, row.final, head)].float(); v = fw.directions[(comp.name, head)].float(); v = v / v.norm()
            direct = float(w @ v.to(w.device)); closure_max = max(closure_max, abs(val["coefficient"] - direct) / max(abs(direct), 1e-6))
    partner = {}
    for row in rows:
        partner[(row.construction, row.group, row.present)] = row
    report = {}
    for key in sorted({k for t in terms for k in t}):
        pooled = {c: {"current": 0.0, "inherited": 0.0} for c in CATEGORIES}; contrast_sum, n = 0.0, 0
        for row in rows:
            if not row.present:
                continue
            other = partner[(row.construction, row.group, False)]
            a, b = terms[rows.index(row)][key][2], terms[rows.index(other)][key][2]
            if len(a["pattern"]) != len(b["pattern"]):
                raise SystemExit("pair positions are not aligned")
            for s in range(len(a["pattern"])):
                c = category(row, s, nouns, objs)
                pooled[c]["current"] += a["term_current"][s] - b["term_current"][s]; pooled[c]["inherited"] += a["term_inherited"][s] - b["term_inherited"][s]
            contrast_sum += a["coefficient"] - b["coefficient"]; n += 1
        shares = {c: (pooled[c]["current"] + pooled[c]["inherited"]) / contrast_sum for c in CATEGORIES}
        inherited = sum(pooled[c]["inherited"] for c in CATEGORIES) / contrast_sum
        ranked = sorted(CATEGORIES, key=lambda c: -abs(shares[c]))
        report[key] = {"mean_contrast": contrast_sum / n, "pairs": n, "shares": shares, "inherited_share": inherited, "largest_source": ranked[0], "by_category_branch": pooled}
        print(candidate, key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report[key].items() if k != "by_category_branch"}))
    tot = sum(r["mean_contrast"] for r in report.values())
    pooled_noun = sum(r["shares"]["noun"] * r["mean_contrast"] for r in report.values()) / tot
    pooled_inh = sum(r["inherited_share"] * r["mean_contrast"] for r in report.values()) / tot
    return {"heads": report, "closure_max": closure_max, "pooled_noun_share": pooled_noun, "pooled_inherited_share": pooled_inh}, forwards


def main() -> None:
    LN = lines()
    plan = {"candidate_id": CANDIDATE_ID, "rows": {k: len(v[0]) for k, v in LN.items()}, "rows_sha256": {k: L.rows_sha256(v[0]) for k, v in LN.items()}, "heads": {k: list(v[3]) for k, v in LN.items()},
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "inherited_max": INHERITED_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend); fw.backend = backend
    forwards, result = 0, {}
    for name, (rows, pos, neg, heads, nouns, objs) in LN.items():
        result[name], n = fold_line(fw, rows, pos, neg, heads, nouns, objs, name); forwards += n
    G, N = result["gender"], result["number"]
    predictions = {"pred_a_fold_closure": max(G["closure_max"], N["closure_max"]) <= CLOSURE_TOL,
                   "pred_b_contrast_positive_for_every_head": all(r["mean_contrast"] > 0 for x in (G, N) for r in x["heads"].values()),
                   "pred_c_gender_noun_is_largest_source_for_every_head": all(r["largest_source"] == "noun" for r in G["heads"].values()),
                   "pred_d_number_noun_share_below_gender": N["pooled_noun_share"] < G["pooled_noun_share"],
                   "pred_e_inherited_branch_share_at_most_030": G["pooled_inherited_share"] <= INHERITED_MAX and N["pooled_inherited_share"] <= INHERITED_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_dod_source_fold_result_v80", "candidate_id": CANDIDATE_ID, "plan": plan, "lines": result, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "pooled": {k: {"noun": round(v["pooled_noun_share"], 3), "inherited": round(v["pooled_inherited_share"], 3)} for k, v in result.items()}, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
