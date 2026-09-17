#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_attn8_fold_closure pred_b_cue_is_the_dominant_source pred_c_inherited_share_at_least_030 pred_d_two_block8_heads_carry_070
"""Aspectual has/had definition-of-done battery, step 14: FOLD attention8's bank write by head and source.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v13 (attention8 is the largest or
second-largest writer of the states heads 9.1/9.4 read at `last`/period/`the`; no token-only share).

WHY. attention8's output at a bank position s, projected on the block-9 reader direction r_h,
is exact: r_h . attn8(s) = sum_{h'} sum_{s'} p_{h'}(s,s') [(1-lamb8)(V_{h'}^T O_{h'}^T r_h).x_{s'}
+ lamb8 (O_{h'}^T r_h).v1_{h'}(s')]. If the since/by token is the dominant source and a large part
travels through the block-0 value branch, the route cue -> attention8@bank -> 9.1/9.4 closes like
head 8.1 did (v11b/v12). If not, folding at this grain stops (kill criterion from review 2).
Oriented since - by contrast, pooled over the three bank positions, per block-9 reader head.
Evidence tag: fold, opened rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_attn8_fold_closure           sum over heads of the per-head totals equals r_h . attn8(s)
                                        from the v13-style trace within relative 1e-3 (every row, position)
    pred_b_cue_is_the_dominant_source   for both reader heads the cue position carries the largest
                                        |share| of the pooled contrast. Prior: unsure.
    pred_c_inherited_share_at_least_030 pooled inherited (token-only) share >= 0.30
    pred_d_two_block8_heads_carry_070   for both reader heads the two largest block-8 heads carry >= 0.70

PRICE (registered maximum): 2 batches x (trace 1 + 3 bank positions x 2 reader heads x 1 fold forward) = 14
forwards; 0 backwards; 0 fits. Bar <= 16.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_attn8_bank_fold_v14_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_attn8_bank_fold_v14"
TOKENS = {"has": 468, "had": 550}
CLOSURE_TOL, INHERITED_MIN, TOP2_MIN = 1e-3, 0.30, 0.70
FORWARDS_MAX = 16
COMP = L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final")
CATS = ("prefix", "cue", "last", "period", "the")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "reader_heads": ["9.1", "9.4"],
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "inherited_min": INHERITED_MIN, "top2_min": TOP2_MIN}}


def cat(row, s):
    last, period, the = row.source_positions
    return {the: "the", period: "period", last: "last", last - 1: "cue"}.get(s, "prefix")


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model = backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), TOKENS["has"], TOKENS["had"])
    readers = L.reader_directions(model, COMP, fw.directions)          # block-9 head -> 1152-d r_h
    O8 = model.transformer.h[8].attn.c_proj.weight.detach().float()    # [1152, 1152]
    dirs = {rh: {hp: O8[:, hp * L.HEAD_DIM:(hp + 1) * L.HEAD_DIM].T @ readers[rh].to(O8.device) for hp in range(9)} for rh in COMP.heads}
    forwards = 0
    traces, folds = [], {rh: {k: [] for k in range(3)} for rh in COMP.heads}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda r: r.source_positions, upto_layer=9)); forwards += 1
        for k in range(3):
            # one forward per bank position serves both reader heads (directions differ per head only)
            out_by_reader = {}
            for rh in COMP.heads:
                out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r, k=k: r.source_positions[k], dirs[rh])
                out_by_reader[rh] = out
            forwards += 1  # registered as one fold forward per position; the second reader re-runs the same forward (accounted below)
            forwards += 1
            for rh in COMP.heads:
                folds[rh][k].extend(out_by_reader[rh])
    closure_max = 0.0
    partner = L.partner_of(rows)
    report = {}
    for rh in COMP.heads:
        r = readers[rh]
        by_cat = {c: {"current": 0.0, "inherited": 0.0} for c in CATS}
        by_head = {hp: 0.0 for hp in range(9)}
        contrast = 0.0
        for i, row in enumerate(rows):
            for k, pos in enumerate(row.source_positions):
                entry = folds[rh][k][i]
                total = sum(entry[hp]["total"] for hp in range(9))
                true = float(r.to(traces[i][("attn:08", pos)].device) @ traces[i][("attn:08", pos)])
                closure_max = max(closure_max, abs(total - true) / max(abs(true), 1e-6))
            if not row.present:
                continue
            j = rows.index(partner[row.row_id])
            for k in range(3):
                a, b = folds[rh][k][i], folds[rh][k][j]
                for hp in range(9):
                    for s in range(len(a[hp]["pattern"])):
                        c = cat(row, s)
                        by_cat[c]["current"] += a[hp]["term_current"][s] - b[hp]["term_current"][s]
                        by_cat[c]["inherited"] += a[hp]["term_inherited"][s] - b[hp]["term_inherited"][s]
                    by_head[hp] += a[hp]["total"] - b[hp]["total"]
                    contrast += a[hp]["total"] - b[hp]["total"]
        shares = {c: (by_cat[c]["current"] + by_cat[c]["inherited"]) / contrast for c in CATS}
        inherited = sum(by_cat[c]["inherited"] for c in CATS) / contrast
        head_shares = {f"8.{hp}": by_head[hp] / contrast for hp in range(9)}
        ranked_heads = sorted(head_shares, key=lambda h: -abs(head_shares[h]))
        report[f"9.{rh}"] = {"contrast": contrast, "source_shares": shares, "inherited_share": inherited,
                             "largest_source": max(CATS, key=lambda c: abs(shares[c])), "head_shares": head_shares,
                             "top2_heads": ranked_heads[:2], "top2_head_share": sum(head_shares[h] for h in ranked_heads[:2])}
        print(f"9.{rh}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[f"9.{rh}"].items() if k not in ("head_shares", "source_shares")}),
              "sources", {c: round(v, 3) for c, v in shares.items()}, "heads", {h: round(v, 3) for h, v in head_shares.items()})
    pooled_inh = sum(report[k]["inherited_share"] * report[k]["contrast"] for k in report) / sum(report[k]["contrast"] for k in report)
    predictions = {
        "pred_a_attn8_fold_closure": closure_max <= CLOSURE_TOL,
        "pred_b_cue_is_the_dominant_source": all(r["largest_source"] == "cue" for r in report.values()),
        "pred_c_inherited_share_at_least_030": pooled_inh >= INHERITED_MIN,
        "pred_d_two_block8_heads_carry_070": all(r["top2_head_share"] >= TOP2_MIN for r in report.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_attn8_bank_fold_result_v14", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "closure_max_relative_error": closure_max, "lambda_block8_value_mix": lamb, "readers": report,
              "pooled_inherited_share": pooled_inh, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure_max, "pooled_inherited": pooled_inh, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
