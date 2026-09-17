#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_two_heads_carry_070_of_each_block pred_c_cue_is_the_dominant_source
"""Temporal will/had DoD battery, step 10 (v37): FOLD attention8/9's NP write by head and source token.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v35/v36 (the NP state 11.3 reads is written by
mlp9, attn9, mlp10, mlp8, attn8 at `the`/agent; 56% causal).

WHY. Same fold as the aspectual v14: at each NP position, attention8's and attention9's write projected on 11.3's
reader direction r = V_h^T v_h expands exactly over the nine heads of the block and their source tokens
(d_{h'} = O_{h'}^T r). Oriented tomorrow - earlier, pooled over the two NP positions.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                     per block, the head totals sum to r . attn(pos) within relative 1e-3
    pred_b_two_heads_carry_070_of_each_block   in each block the two largest heads carry >= 0.70 of that block's
                                            contrast. Prior: unsure.
    pred_c_cue_is_the_dominant_source       for each block the cue position is the largest |share| source

PRICE (registered maximum): 2 batches x (trace 1 + 2 positions x 2 blocks x 1 fold) = 10 forwards; 0 backwards;
0 fits. Bar <= 12.
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
import run_temporal_dod_removal_v28 as v28
import run_temporal_dod_np_writer_fold_v35 as v35

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_np_head_fold_v37_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_np_head_fold_v37"
CLOSURE_TOL, TOP2_MIN = 1e-3, 0.70
FORWARDS_MAX = 12
COMP = L.Component("attn11_h3_final", 11, "attn", (3,), "final")
CATS = ("prefix", "cue", "the1", "agent", "prep", "the2", "place")


def cat(row, s):
    n = len(row.ids)
    cue = next(i for i, t in enumerate(row.ids) if L.ENCODING.decode([t]).strip().lower() in ("tomorrow", "earlier"))
    return {n - 1: "place", n - 2: "the2", n - 3: "prep", n - 4: "agent", n - 5: "the1", cue: "cue"}.get(s, "prefix")


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model = backend.model
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(model, (COMP,), v28.WILL, v28.HAD)
    r = L.reader_directions(model, COMP, fw.directions)[3]
    dirs = {}
    for layer in (8, 9):
        O = model.transformer.h[layer].attn.c_proj.weight.detach().float()
        dirs[layer] = {hp: O[:, hp * L.HEAD_DIM:(hp + 1) * L.HEAD_DIM].T @ r.to(O.device) for hp in range(9)}
    forwards = 0
    traces, folds = [], {(layer, k): [] for layer in (8, 9) for k in range(2)}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, v35.np_positions, upto_layer=11)); forwards += 1
        for layer in (8, 9):
            for k in range(2):
                out, _ = L.head_source_terms_at(fw, chunk, layer, lambda rr, k=k: v35.np_positions(rr)[k], dirs[layer]); forwards += 1
                folds[(layer, k)].extend(out)
    partner = L.partner_of(rows)
    closure, report = 0.0, {}
    for layer in (8, 9):
        by_head = {hp: 0.0 for hp in range(9)}; by_cat = {c: 0.0 for c in CATS}; contrast = 0.0
        for i, row in enumerate(rows):
            for k, pos in enumerate(v35.np_positions(row)):
                entry = folds[(layer, k)][i]
                total = sum(entry[hp]["total"] for hp in range(9))
                true = float(r.to(traces[i][(f"attn:{layer:02d}", pos)].device) @ traces[i][(f"attn:{layer:02d}", pos)])
                closure = max(closure, abs(total - true) / max(abs(true), 1e-6))
            if not row.present: continue
            j = rows.index(partner[row.row_id])
            for k in range(2):
                a, b = folds[(layer, k)][i], folds[(layer, k)][j]
                for hp in range(9):
                    d = a[hp]["total"] - b[hp]["total"]; by_head[hp] += d; contrast += d
                    for s in range(len(a[hp]["pattern"])):
                        by_cat[cat(row, s)] += (a[hp]["term_current"][s] + a[hp]["term_inherited"][s]) - (b[hp]["term_current"][s] + b[hp]["term_inherited"][s])
        head_shares = {f"{layer}.{hp}": by_head[hp] / contrast for hp in range(9)}; ranked = sorted(head_shares, key=lambda h: -abs(head_shares[h]))
        src = {c: by_cat[c] / contrast for c in CATS}
        report[f"attn{layer}"] = {"contrast": contrast, "head_shares": head_shares, "top2": ranked[:2], "top2_share": sum(head_shares[h] for h in ranked[:2]),
                                 "source_shares": src, "largest_source": max(CATS, key=lambda c: abs(src[c]))}
        print(f"attn{layer}", "top heads", [(h, round(head_shares[h], 3)) for h in ranked[:4]], "sources", {c: round(v, 3) for c, v in src.items()})
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_two_heads_carry_070_of_each_block": all(v["top2_share"] >= TOP2_MIN for v in report.values()),
                   "pred_c_cue_is_the_dominant_source": all(v["largest_source"] == "cue" for v in report.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_np_head_fold_result_v37", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "blocks": report,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
