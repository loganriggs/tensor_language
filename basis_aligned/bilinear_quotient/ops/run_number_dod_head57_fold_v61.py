#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_subject_token_is_the_largest_source_for_5_7 pred_c_inherited_branch_dominates_5_7 pred_d_readout_pair_keep_retains_most
"""Number family DoD, step 7 (v61): what head 5.7 reads, and keep-only of the one-directional pair {11.3, 7.8}.

Parents: v59 (11.3 and 7.8 one-directional; 5.7 not). Fold: 5.7's coefficient on the were−was readout direction split by
source token and value branch on the v55 rows (oriented plural − singular; positions align across the pair since the
plural adds 's' inside the same token). Edit: keep only the readout projection at 11.3 and 7.8 (5.7, 9.7 native) vs zeroing
those two slices.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                      <= 1e-3
    pred_b_subject_token_is_the_largest_source_for_5_7   the subject-noun position carries the largest |share| of 5.7's
                                             oriented contrast (prior: yes — a number-cue reader)
    pred_c_inherited_branch_dominates_5_7    inherited (block-0 value) share >= 0.50 (prior: unsure; 8.1's analogue was 0.8)
    pred_d_readout_pair_keep_retains_most    keep-only at {11.3, 7.8} retains >= 0.80 of zeroing those two slices

PRICE (registered maximum): 2 batches x (capture 1 + fold 1 + native 1 + zero 1 + keep 1) = 10 forwards (+ producer 2) = 12;
bar <= 16.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_head57_fold_v61_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_head57_fold_v61"
CLOSURE_TOL, INHERITED_MIN, RETAIN_MIN = 1e-3, 0.50, 0.80
FORWARDS_MAX = 16
H57 = L.Component("attn5_h7_final", 5, "attn", (7,), "final")
PAIR = (L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn7_h8_final", 7, "attn", (8,), "final"))
CATS = ("prefix", "the1", "subject", "prep", "the2", "place")


def cat(row, s):
    n = len(row.ids)
    return {n - 1: "place", n - 2: "the2", n - 3: "prep", n - 4: "subject", n - 5: "the1"}.get(s, "prefix")


def main() -> None:
    rows, _ = v55.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (H57,) + PAIR, v55.WERE, v55.WAS)
    forwards = 0
    store, terms = {}, []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        store.update(fw.capture(chunk, (H57,))); forwards += 1
        out, lamb = L.head_source_terms(fw, chunk, H57, fw.directions); forwards += 1
        terms.extend(e[7] for e in out)
    closure = 0.0
    for row, t in zip(rows, terms):
        w = store[(row.row_id, H57.name, row.final, 7)].float(); v = fw.directions[(H57.name, 7)].float(); v = v / v.norm()
        direct = float(w @ v.to(w.device)); closure = max(closure, abs(t["coefficient"] - direct) / max(abs(direct), 1e-6))
    partner = L.partner_of(rows)
    pooled = {c: 0.0 for c in CATS}; inherited = 0.0; contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present: continue
        j = rows.index(partner[row.row_id]); a, b = terms[i], terms[j]
        for s in range(len(a["pattern"])):
            pooled[cat(row, s)] += (a["term_current"][s] + a["term_inherited"][s]) - (b["term_current"][s] + b["term_inherited"][s])
            inherited += a["term_inherited"][s] - b["term_inherited"][s]
        contrast += a["coefficient"] - b["coefficient"]
    shares = {c: pooled[c] / contrast for c in CATS}; inh = inherited / contrast
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=PAIR, mode="zero"); forwards += n
    keep, n = v1._run_arm(fw, rows, components=PAIR, mode="keep_only"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]; retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd
    print("5.7 contrast", round(contrast, 2), "inherited", round(inh, 3), {c: round(v, 3) for c, v in shares.items()}, "| pair zero", round(zd, 3), "keep retention", round(retention, 3))
    predictions = {"pred_a_fold_closure": closure <= CLOSURE_TOL, "pred_b_subject_token_is_the_largest_source_for_5_7": max(CATS, key=lambda c: abs(shares[c])) == "subject",
                   "pred_c_inherited_branch_dominates_5_7": inh >= INHERITED_MIN, "pred_d_readout_pair_keep_retains_most": retention >= RETAIN_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_head57_fold_result_v61", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure, "head_5_7": {"contrast": contrast, "shares": shares, "inherited_share": inh},
                               "pair_keep": {"zero_damage": zd, "retention": retention}, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
