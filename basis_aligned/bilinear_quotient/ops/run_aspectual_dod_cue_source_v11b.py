#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_recomputed_head_replays_native pred_b_zeroing_8_1_is_live pred_c_cue_inherited_term_retains_most pred_d_cue_both_branches_retains_more pred_e_every_other_single_source_retains_little
"""Aspectual has/had definition-of-done battery, step 11b: head 8.1 as a token-only cue reader (EDIT).

V11B. v11 (`runlogs/run_aspectual_dod_cue_source_v11.log`, exit 1, preserved) executed all arms but
refused to write its receipt because its registered price (30 forwards) was mis-counted: each of the
8 arms costs 2 recompute-capture forwards + 2 arm forwards = 32, plus native/producer/zero 6 = 38.
Only the price line and identifiers change here; rows, arms, bars and predictions are identical.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v10 fold (83% of 8.1's
coefficient contrast comes from the since/by token, 78% through the block-0 value branch).

WHY. A fold nominates; an edit decides (better_circuits §3.3). If head 8.1's has/had service is
the single term p_{8.1}(final, cue) x lamb x v1_{8.1}(cue) -- a native pattern scalar times a
TOKEN-ONLY value table -- then replacing the head's whole final-query output by that one term
must retain its contribution, and any other single source must not. Arms replace head 8.1's
pre-c_proj slice at the final query by the recomputed attention restricted to one source
position and to the named value branches; the zeroed slice is the reference (v8: 0.345 logits).

ROWS: 64 discovery rows (opened). Source categories: cue (since/by), `last`, period noun,
`the`, agent (self), prefix (report-frame tokens; fronted rows have none, so that arm is scored
on the 32 report rows).

PREDICTIONS (scored as written; failures preserved)
    pred_a_recomputed_head_replays_native   replacing the slice by the FULL recomputation (all
                                            sources, both branches) changes every row's answer/foil
                                            logits by <= 1e-3 (instrument)
    pred_b_zeroing_8_1_is_live              zero-slice damage >= 0.10 fraction, positive >= 0.75
    pred_c_cue_inherited_term_retains_most  cue-only, inherited-branch-only retention >= 0.70
    pred_d_cue_both_branches_retains_more   cue-only, both branches retention >= 0.80
    pred_e_every_other_single_source_retains_little   last / period / the / agent / prefix single-
                                            source arms (both branches) each retain <= 0.30

PRICE (registered maximum): native 2 + producer 2 + zero 2 + 8 arms x (2 capture + 2 arm) = 38
forwards; 0 backwards; 0 fits. Bar <= 44.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_cue_source_v11b_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_cue_source_v11b"
LIVE_FRACTION, LIVE_POSITIVE, RETAIN_C, RETAIN_D, OTHER_MAX, INSTRUMENT_TOL = 0.10, 0.75, 0.70, 0.80, 0.30, 1e-3
FORWARDS_MAX = 44
HEAD = L.Component("attn8_h1_final", 8, "attn", (1,), "final")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": "8.1",
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "retain_c": RETAIN_C,
                     "retain_d": RETAIN_D, "other_max": OTHER_MAX, "instrument_tol": INSTRUMENT_TOL}}


def cat(row, s):
    last, period, the = row.source_positions
    return {row.final: "agent", the: "the", period: "period", last: "last", last - 1: "cue"}.get(s, "prefix")


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    zero, n = v1._run_arm(fw, rows, components=(HEAD,), mode="zero"); forwards += n
    arms_spec = {"full_recompute": (lambda r, s: True, ("current", "inherited")),
                 "cue_inherited": (lambda r, s: cat(r, s) == "cue", ("inherited",)),
                 "cue_both": (lambda r, s: cat(r, s) == "cue", ("current", "inherited")),
                 "last_both": (lambda r, s: cat(r, s) == "last", ("current", "inherited")),
                 "period_both": (lambda r, s: cat(r, s) == "period", ("current", "inherited")),
                 "the_both": (lambda r, s: cat(r, s) == "the", ("current", "inherited")),
                 "agent_both": (lambda r, s: cat(r, s) == "agent", ("current", "inherited")),
                 "prefix_both": (lambda r, s: cat(r, s) == "prefix", ("current", "inherited"))}
    results = {}
    for name, (keep, branches) in arms_spec.items():
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            table.update(L.source_restricted_slices(fw, chunk, HEAD, keep, branches)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        results[name] = arm
    fw.use_subtract = False

    def retention(name, idx=None):
        idx = list(range(len(rows))) if idx is None else idx
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in idx])["target_damage_mean"]
        a = L.summarize(sub_rows, sub_native, [results[name][i] for i in idx])["target_damage_mean"]
        return {"zero_damage": z, "arm_damage": a, "retention": 1.0 - a / z if z else None}

    instrument = max(max(abs(a["answer"] - b["answer"]), abs(a["foil"] - b["foil"])) for a, b in zip(results["full_recompute"], native))
    zero_summary = L.summarize(rows, native, zero)
    report_idx = [i for i, r in enumerate(rows) if r.construction == "report"]
    report = {name: retention(name, report_idx if name == "prefix_both" else None) for name in arms_spec}
    for name, r in report.items():
        print(name, {k: round(v, 4) for k, v in r.items()})
    others = ("last_both", "period_both", "the_both", "agent_both", "prefix_both")
    predictions = {
        "pred_a_recomputed_head_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_zeroing_8_1_is_live": zero_summary["target_damage_fraction"] >= LIVE_FRACTION and zero_summary["target_damage_positive_fraction"] >= LIVE_POSITIVE,
        "pred_c_cue_inherited_term_retains_most": report["cue_inherited"]["retention"] >= RETAIN_C,
        "pred_d_cue_both_branches_retains_more": report["cue_both"]["retention"] >= RETAIN_D,
        "pred_e_every_other_single_source_retains_little": all(report[o]["retention"] <= OTHER_MAX for o in others),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_cue_source_result_v11b", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "instrument_max_abs_error": instrument, "zero": zero_summary, "arms": report,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "instrument": instrument, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
