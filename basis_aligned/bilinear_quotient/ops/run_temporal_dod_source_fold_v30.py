#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_contrast_positive_for_every_head pred_c_cue_is_the_largest_source_for_11_3 pred_d_block9_heads_read_context_not_cue
"""Temporal will/had DoD battery, step 3 (v30): FOLD the four readout coefficients over source positions.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v28/v29 (S = {11.3, 9.1, 15.5, 9.4}).
Same exact expansion as the aspectual v10: c_h = sum_s p_h(t,s)[(1-lamb)(V_h^T v_h).x_s + lamb v_h.v1_h(s)],
oriented tomorrow - earlier over the 32 aligned pairs per construction of the v28 fresh rows. Source
categories: cue (tomorrow/earlier), `the` (first determiner), agent, preposition, second `the`, place = final,
prefix (report frame).

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                     sum of terms equals the captured v_h . w_h within relative 1e-3
    pred_b_contrast_positive_for_every_head mean oriented contrast > 0 for all four heads
    pred_c_cue_is_the_largest_source_for_11_3   the cue position is 11.3's largest |share|. Prior: unsure --
                                            in the aspectual line 8.1 read the cue and the block-9 heads read
                                            a contextual bank; 11.3 sits later still.
    pred_d_block9_heads_read_context_not_cue    for 9.1 and 9.4 the cue share is < 0.5

PRICE (registered maximum): 2 batches x (capture 1 + 3 folds for blocks 9, 11, 15) = 8 forwards; 0 backwards;
0 fits. Bar <= 10.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_source_fold_v30_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_source_fold_v30"
CLOSURE_TOL, CUE_MAX = 1e-3, 0.5
FORWARDS_MAX = 10
COMPS = (L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"), L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn15_h5_final", 15, "attn", (5,), "final"))
CATS = ("prefix", "cue", "the1", "agent", "prep", "the2", "place")


def category(row, s):
    ids = row.ids; n = len(ids)
    cue = next(i for i, t in enumerate(ids) if L.ENCODING.decode([t]).strip().lower() in ("tomorrow", "earlier"))
    if s == n - 1: return "place"
    if s == n - 2: return "the2"
    if s == n - 3: return "prep"
    if s == n - 4: return "agent"
    if s == n - 5: return "the1"
    if s == cue: return "cue"
    return "prefix"


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, COMPS, v28.WILL, v28.HAD)
    forwards = 0
    store, terms = {}, [dict() for _ in rows]
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        store.update(fw.capture(chunk, COMPS)); forwards += 1
        for comp in COMPS:
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            for j, entry in enumerate(out):
                for head, val in entry.items():
                    terms[start + j][f"{comp.name}:{head}"] = val
    closure = 0.0
    for row, t in zip(rows, terms):
        for key, val in t.items():
            comp_name, head = key.rsplit(":", 1); head = int(head)
            w = store[(row.row_id, comp_name, row.final, head)].float()
            v = fw.directions[(comp_name, head)].float(); v = v / v.norm()
            direct = float(w @ v.to(w.device)); closure = max(closure, abs(val["coefficient"] - direct) / max(abs(direct), 1e-6))
    partner = L.partner_of(rows)
    report = {}
    for key, label in (("attn9_h1_h4_final:1", "9.1"), ("attn9_h1_h4_final:4", "9.4"), ("attn11_h3_final:3", "11.3"), ("attn15_h5_final:5", "15.5")):
        pooled = {c: 0.0 for c in CATS}; inherited = 0.0; contrast = 0.0; n = 0
        for i, row in enumerate(rows):
            if not row.present: continue
            j = rows.index(partner[row.row_id]); a, b = terms[i][key], terms[j][key]
            if len(a["pattern"]) != len(b["pattern"]): raise SystemExit("pairs not aligned")
            for s in range(len(a["pattern"])):
                c = category(row, s)
                pooled[c] += (a["term_current"][s] + a["term_inherited"][s]) - (b["term_current"][s] + b["term_inherited"][s])
                inherited += a["term_inherited"][s] - b["term_inherited"][s]
            contrast += a["coefficient"] - b["coefficient"]; n += 1
        shares = {c: pooled[c] / contrast for c in CATS}
        report[label] = {"mean_contrast": contrast / n, "shares": shares, "inherited_share": inherited / contrast, "largest_source": max(CATS, key=lambda c: abs(shares[c]))}
        print(label, "contrast", round(contrast / n, 2), "inherited", round(inherited / contrast, 3), {c: round(v, 3) for c, v in shares.items()})
    predictions = {
        "pred_a_fold_closure": closure <= CLOSURE_TOL,
        "pred_b_contrast_positive_for_every_head": all(r["mean_contrast"] > 0 for r in report.values()),
        "pred_c_cue_is_the_largest_source_for_11_3": report["11.3"]["largest_source"] == "cue",
        "pred_d_block9_heads_read_context_not_cue": all(report[k]["shares"]["cue"] < CUE_MAX for k in ("9.1", "9.4")),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_source_fold_result_v30", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "heads": report,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
