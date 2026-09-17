#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_contrast_is_positive_for_every_head pred_c_cue_token_is_the_largest_source pred_d_inherited_branch_share_at_least_030 pred_e_top_two_sources_carry_070
"""Aspectual has/had definition-of-done battery, step 10: FOLD the three readout coefficients.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v9 (keep-only: the three
coefficients c_h = w_h . v_h at heads 8.1, 9.1, 9.4 carry 100% of the heads' has/had service).

WHY. The component's remaining ports are those three native scalars. better_circuits §3.8: close
ports by folding, not fitting. The head output is exact:
    z_h(t) = sum_s p_h(t,s) [ (1 - lamb) V_h x_s + lamb v1_h(s) ],   p = (q.k/D)(q2.k2/D),
so c_h = sum_s p_h(t,s) [(1-lamb) (V_h^T v_h) . x_s + lamb (v_h . v1_h(s))]. The second branch
uses the BLOCK-0 value v1, a function of the source TOKEN alone (better_circuits §3.7's
token-only generator); the first uses the source's block-l residual state. This run reports,
for the 32 aligned since/by pairs of each discovery construction, the oriented coefficient
contrast Delta c_h = c_h(since) - c_h(by) split by source position (cue, `last`, period noun,
`the`, agent = self, prefix) and by branch. Evidence tag: fold (no intervention), opened rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                sum of terms equals the captured v_h . w_h within relative
                                       1e-3 for every row and head
    pred_b_contrast_is_positive_for_every_head   mean Delta c_h > 0 for 8.1, 9.1, 9.4 (both
                                       constructions pooled)
    pred_c_cue_token_is_the_largest_source       for each head the cue position carries the
                                       largest |share| of Delta c_h. Prior: unsure -- the released
                                       path says block9 H1/H4 read a contextual trace at
                                       `last`+period+`the`, not the raw cue.
    pred_d_inherited_branch_share_at_least_030   pooled over heads, the token-only v1 branch
                                       carries >= 0.30 of Delta c_h. Prior: unsure.
    pred_e_top_two_sources_carry_070   for each head the two largest source positions carry
                                       >= 0.70 of Delta c_h

PRICE (registered maximum): 2 batches x (1 capture + 2 fold forwards for blocks 8 and 9) = 6
forwards; 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_source_fold_v10_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_source_fold_v10"
TOKENS = {"has": 468, "had": 550}
CLOSURE_TOL, INHERITED_MIN, TOP2_MIN = 1e-3, 0.30, 0.70
FORWARDS_MAX = 12
COMPONENTS = (L.Component("attn8_h1_final", 8, "attn", (1,), "final"),
              L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))
CATEGORIES = ("prefix", "cue", "last", "period", "the", "agent")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "heads": ["8.1", "9.1", "9.4"], "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "inherited_min": INHERITED_MIN, "top2_min": TOP2_MIN}}


def category(row, position):
    last, period, the = row.source_positions
    if position == row.final:
        return "agent"
    if position == the:
        return "the"
    if position == period:
        return "period"
    if position == last:
        return "last"
    if position == last - 1:
        return "cue"
    return "prefix"


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, COMPONENTS, TOKENS["has"], TOKENS["had"])
    forwards = 0
    store = {}
    terms = [dict() for _ in rows]
    lambs = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        store.update(fw.capture(chunk, COMPONENTS)); forwards += 1
        for comp in COMPONENTS:
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            lambs[comp.name] = lamb
            for j, entry in enumerate(out):
                for head, val in entry.items():
                    terms[start + j][f"{comp.name}:{head}"] = (comp, head, val)

    # closure against the captured slices
    closure_max = 0.0
    for row, t in zip(rows, terms):
        for key, (comp, head, val) in t.items():
            w = store[(row.row_id, comp.name, row.final, head)].float()
            v = fw.directions[(comp.name, head)].float(); v = v / v.norm()
            direct = float(w @ v.to(w.device))
            closure_max = max(closure_max, abs(val["coefficient"] - direct) / max(abs(direct), 1e-6))

    partner = L.partner_of(rows)
    heads = ["attn8_h1_final:1", "attn9_h1_h4_final:1", "attn9_h1_h4_final:4"]
    report = {}
    for key in heads:
        pooled = {c: {"current": 0.0, "inherited": 0.0} for c in CATEGORIES}
        contrast_sum, n = 0.0, 0
        for row in rows:
            if not row.present:
                continue
            other = partner[row.row_id]
            a, b = terms[rows.index(row)][key][2], terms[rows.index(other)][key][2]
            if len(a["pattern"]) != len(b["pattern"]):
                raise SystemExit("pair positions are not aligned")
            for s in range(len(a["pattern"])):
                c = category(row, s)
                pooled[c]["current"] += a["term_current"][s] - b["term_current"][s]
                pooled[c]["inherited"] += a["term_inherited"][s] - b["term_inherited"][s]
            contrast_sum += a["coefficient"] - b["coefficient"]; n += 1
        mean_contrast = contrast_sum / n
        shares = {c: (pooled[c]["current"] + pooled[c]["inherited"]) / contrast_sum for c in CATEGORIES}
        inherited_share = sum(pooled[c]["inherited"] for c in CATEGORIES) / contrast_sum
        ranked = sorted(CATEGORIES, key=lambda c: -abs(shares[c]))
        report[key] = {"mean_contrast": mean_contrast, "pairs": n, "shares": shares,
                       "inherited_share": inherited_share, "current_share": 1.0 - inherited_share,
                       "largest_source": ranked[0], "top2_share": shares[ranked[0]] + shares[ranked[1]],
                       "by_category_branch": pooled}
        print(key, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in report[key].items() if k != "by_category_branch"}))
    pooled_inherited = sum(report[k]["inherited_share"] * report[k]["mean_contrast"] for k in heads) / sum(report[k]["mean_contrast"] for k in heads)
    predictions = {
        "pred_a_fold_closure": closure_max <= CLOSURE_TOL,
        "pred_b_contrast_is_positive_for_every_head": all(report[k]["mean_contrast"] > 0 for k in heads),
        "pred_c_cue_token_is_the_largest_source": all(report[k]["largest_source"] == "cue" for k in heads),
        "pred_d_inherited_branch_share_at_least_030": pooled_inherited >= INHERITED_MIN,
        "pred_e_top_two_sources_carry_070": all(report[k]["top2_share"] >= TOP2_MIN for k in heads),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_source_fold_result_v10", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "closure_max_relative_error": closure_max, "lambda_by_block": lambs, "heads": report,
              "pooled_inherited_share": pooled_inherited, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure_max, "pooled_inherited": pooled_inherited,
                      "lambda": lambs, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
