#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_full_recompute_replays_native pred_b_zeroing_8_1_at_four_positions_is_live pred_c_native_cue_term_retains_most pred_d_token_only_table_retains_half
"""Aspectual has/had definition-of-done battery, step 15: head 8.1 as a token-only generator EVERYWHERE it matters.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v12 (8.1 at the final query =
per-cue constant x lamb x v1(cue), retention 0.83 cross-construction), v14 (attention8's write at the
bank positions `last`/period/`the`, as read by 9.1/9.4, is 8.1 reading the cue the same way).

WHY. If head 8.1's entire aspect service -- its direct readout write AND its write into the bank that
the block-9 heads relay -- is one token-only table, then replacing the head at all four positions by
that table must keep most of the component's has/had contribution. The constant per (cue, query
category) is the median native pattern from the OTHER construction; stored constants: 2 cues x 4 query
categories = 8 numbers, declared here. Reference: zero 8.1's slice at the same four positions.

ROWS: 64 discovery rows (opened). Edits at `last`, period, `the`, final (head 8.1 slice only).

PREDICTIONS (scored as written; failures preserved)
    pred_a_full_recompute_replays_native   replacing the four slices by the full recomputation changes
                                           answer/foil logits by <= 1e-3
    pred_b_zeroing_8_1_at_four_positions_is_live   damage >= 0.10 fraction, positive on >= 0.75 rows
    pred_c_native_cue_term_retains_most    cue-only inherited-only term with NATIVE patterns at all four
                                           positions retains >= 0.60 of the zero damage
    pred_d_token_only_table_retains_half   cross-construction constant patterns retain >= 0.50 (pooled:
                                           report rows under fronted constants + fronted rows under
                                           report constants)

PRICE (registered maximum): native 2 + producer 2 + zero 2 + pattern folds 4 positions x 2 batches = 8
+ 3 arms x (2 capture + 2 arm) = 12 -> 26 forwards; 0 backwards; 0 fits. Bar <= 32.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import statistics
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_token_only_8_1_v15_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_token_only_8_1_v15"
TOKENS = {"has": 468, "had": 550}
LIVE_FRACTION, LIVE_POSITIVE, RETAIN_C, RETAIN_D, INSTRUMENT_TOL = 0.10, 0.75, 0.60, 0.50, 1e-3
FORWARDS_MAX = 32
HEAD = L.Component("attn8_h1_all", 8, "attn", (1,), "source_and_final")
QCATS = ("last", "period", "the", "final")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": "8.1",
            "positions": list(QCATS), "stored_constants": 8, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "retain_c": RETAIN_C,
                     "retain_d": RETAIN_D, "instrument_tol": INSTRUMENT_TOL}}


def is_cue(row, s):
    return s == row.source_positions[0] - 1


def qcat(row, t):
    return {row.source_positions[0]: "last", row.source_positions[1]: "period", row.source_positions[2]: "the", row.final: "final"}[t]


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (HEAD,), TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(HEAD,), mode="zero"); forwards += n

    # native pattern of 8.1 on the cue at each of the four query positions
    pattern = {}
    unit = {1: fw.directions[(HEAD.name, 1)]}
    for k in range(4):
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r, k=k: (tuple(r.source_positions) + (r.final,))[k], unit); forwards += 1
            for row, entry in zip(chunk, out):
                pattern[(row.row_id, QCATS[k])] = entry[1]["pattern"][row.source_positions[0] - 1]
    groups = {}
    for row in rows:
        for qc in QCATS:
            groups.setdefault((row.construction, row.present, qc), []).append(pattern[(row.row_id, qc)])
    median = {key: statistics.median(v) for key, v in groups.items()}

    def arm_with(override):
        nonlocal forwards
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            table.update(L.source_restricted_slices(fw, chunk, HEAD, is_cue, ("inherited",), override)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        return arm

    def full_table():
        nonlocal forwards
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            table.update(L.source_restricted_slices(fw, chunk, HEAD, lambda r, s: True, ("current", "inherited"))); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        return arm

    full = full_table()
    native_p = arm_with(None)
    other = {"fronted": "report", "report": "fronted"}
    constant_p = arm_with(lambda r, s, t: median[(other[r.construction], r.present, qcat(r, t))] if is_cue(r, s) else None)
    fw.use_subtract = False

    def retention(arm):
        z = L.summarize(rows, native, zero)["target_damage_mean"]
        a = L.summarize(rows, native, arm)["target_damage_mean"]
        return {"zero_damage": z, "arm_damage": a, "retention": 1.0 - a / z}

    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(full, native))
    zero_summary = L.summarize(rows, native, zero)
    scored = {"native_cue_term": retention(native_p), "token_only_table": retention(constant_p)}
    for k, v in scored.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    predictions = {
        "pred_a_full_recompute_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_zeroing_8_1_at_four_positions_is_live": zero_summary["target_damage_fraction"] >= LIVE_FRACTION and zero_summary["target_damage_positive_fraction"] >= LIVE_POSITIVE,
        "pred_c_native_cue_term_retains_most": scored["native_cue_term"]["retention"] >= RETAIN_C,
        "pred_d_token_only_table_retains_half": scored["token_only_table"]["retention"] >= RETAIN_D,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_token_only_8_1_result_v15", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument, "zero": zero_summary, "scored": scored,
              "constants": {f"{c}/{'since' if pres else 'by'}/{qc}": v for (c, pres, qc), v in median.items()},
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "instrument": instrument, "zero_damage": zero_summary["target_damage_mean"], "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
