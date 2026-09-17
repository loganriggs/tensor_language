#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_contrast_positive_for_every_head pred_c_verb_or_adverb_is_the_largest_source
"""Narrative tense DoD battery, step 3 (v44): FOLD the readout coefficients of 15.5, 11.3, 9.4, 9.1 by source token.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v42/v43 (set {15.5, 11.3, 9.4, 9.1} on
`O_h^T(u_was - u_is)`; 74% of the margin). Same exact expansion as v10/v30, oriented past - present over the 32 aligned
pairs per construction of the v43 rows. Source categories (direct): adverb (Last/Every), season (winter), subject NP
(the + subject), verb (stood/stands), rest of sentence 1, tail (The main reason for the <focus> <place>) with the final
token separate. Relative construction categorized analogously.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                      sum of terms equals the captured v_h . w_h within relative 1e-3
    pred_b_contrast_positive_for_every_head  mean oriented contrast > 0 for all four heads
    pred_c_verb_or_adverb_is_the_largest_source   for 15.5 and 11.3 the largest |share| category is the tensed verb or the
                                             adverb (the tense-bearing tokens), not the tail. Prior: unsure -- the temporal
                                             11.3 read the subject NP state.

PRICE (registered maximum): 2 batches x (capture 1 + 3 folds for blocks 9, 11, 15) = 8 forwards; 0 backwards; 0 fits.
Bar <= 10.
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
import run_narrative_dod_sweep_and_set_v42 as v42
import run_narrative_dod_confirm_v43 as v43

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/narrative_tense_dod_source_fold_v44_result.json"
CANDIDATE_ID = "narrative_tense.past_vs_present.dod_source_fold_v44"
CLOSURE_TOL = 1e-3
FORWARDS_MAX = 10
COMPS = (L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"), L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn15_h5_final", 15, "attn", (5,), "final"))
CATS = ("adverb", "season", "subject_np", "verb", "rest1", "tail", "final")


def category(row, s):
    tok = L.ENCODING.decode([row.ids[s]]).strip().lower()
    n = len(row.ids)
    if s == n - 1: return "final"
    # tail begins at the token "The" of "The main reason" -- find the last "The" (capital) index
    tail_start = max(i for i, t in enumerate(row.ids) if L.ENCODING.decode([t]).strip() == "The" and i > 0) if row.construction == "direct" else max(i for i, t in enumerate(row.ids) if L.ENCODING.decode([t]).strip() == "The" and i > 0)
    if s >= tail_start: return "tail"
    if tok in ("last", "every"): return "adverb"
    if tok == "winter": return "season"
    if tok in ("stood", "stands"): return "verb"
    if row.construction == "direct":
        if s <= 3: return "subject_np"   # "Last winter the <subj>" -> positions 2,3 are the NP; adverb/season handled above
        return "rest1"
    else:
        if s <= 1: return "subject_np"   # "The <subj> that ..."
        return "rest1"


def main() -> None:
    rows = v43.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, COMPS, v42.WAS, v42.IS)
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
            w = store[(row.row_id, comp_name, row.final, head)].float(); v = fw.directions[(comp_name, head)].float(); v = v / v.norm()
            direct = float(w @ v.to(w.device)); closure = max(closure, abs(val["coefficient"] - direct) / max(abs(direct), 1e-6))
    partner = L.partner_of(rows)
    report = {}
    for key, label in (("attn15_h5_final:5", "15.5"), ("attn11_h3_final:3", "11.3"), ("attn9_h1_h4_final:4", "9.4"), ("attn9_h1_h4_final:1", "9.1")):
        pooled = {c: 0.0 for c in CATS}; contrast = 0.0; n = 0; inherited = 0.0
        for i, row in enumerate(rows):
            if not row.present: continue   # present=True is the PAST side here (answer " was")
            j = rows.index(partner[row.row_id]); a, b = terms[i][key], terms[j][key]
            if len(a["pattern"]) != len(b["pattern"]): raise SystemExit("pairs not aligned")
            for s in range(len(a["pattern"])):
                pooled[category(row, s)] += (a["term_current"][s] + a["term_inherited"][s]) - (b["term_current"][s] + b["term_inherited"][s])
                inherited += a["term_inherited"][s] - b["term_inherited"][s]
            contrast += a["coefficient"] - b["coefficient"]; n += 1
        shares = {c: pooled[c] / contrast for c in CATS}
        report[label] = {"mean_contrast": contrast / n, "shares": shares, "inherited_share": inherited / contrast, "largest_source": max(CATS, key=lambda c: abs(shares[c]))}
        print(label, "contrast", round(contrast / n, 2), "inh", round(inherited / contrast, 3), {c: round(v, 3) for c, v in shares.items()})
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_contrast_positive_for_every_head": all(r["mean_contrast"] > 0 for r in report.values()),
                   "pred_c_verb_or_adverb_is_the_largest_source": all(report[k]["largest_source"] in ("verb", "adverb") for k in ("15.5", "11.3"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "narrative_tense_dod_source_fold_result_v44", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "heads": report, "predictions": predictions,
              "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
