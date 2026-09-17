#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_cue_term_replays_v11b pred_b_pattern_is_stable_within_cue pred_c_constant_pattern_transfers_to_report_rows pred_d_constant_pattern_transfers_to_fronted_rows pred_e_token_only_table_retains_half_pooled
"""Aspectual has/had definition-of-done battery, step 12: is head 8.1 a token-only generator?

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v11b (8.1's service is the
single term p(final, cue) x lamb x v1(cue); v1 is a function of the cue token alone).

WHY. After v11b the only context-dependent port left in head 8.1's aspect service is the scalar
p_{8.1}(final, cue) = (q.k/D)(q2.k2/D). If that scalar is effectively a constant per cue token,
the head's has/had service is a two-entry LOOKUP TABLE on the cue token -- better_circuits §3.7's
token-only generator, with the port closed. The constant is the MEDIAN of the native scalar over
one construction's rows, per cue token; it is scored only on the OTHER construction, so no row
contributes to both the constant and its test. This is a two-number description, not a fit.

ROWS: 64 discovery rows (32 fronted, 32 report; each 16 since + 16 by). Reference: zero the 8.1
slice (v11b: 0.46 logits). Arms replace the slice by lamb x p x v1(cue) with p = native (replay
of v11b's cue_inherited), p = median over fronted rows (scored on report rows), p = median over
report rows (scored on fronted rows).

PREDICTIONS (scored as written; failures preserved)
    pred_a_native_cue_term_replays_v11b      native-p arm retention within 0.02 of v11b (0.7585)
    pred_b_pattern_is_stable_within_cue      coefficient of variation of p over rows <= 0.35 for
                                             each cue token in each construction. Prior: unsure.
    pred_c_constant_pattern_transfers_to_report_rows    fronted-median p scored on report rows:
                                             retention >= 0.60
    pred_d_constant_pattern_transfers_to_fronted_rows   report-median p scored on fronted rows:
                                             retention >= 0.60
    pred_e_token_only_table_retains_half_pooled   pooled cross-construction retention >= 0.50

PRICE (registered maximum): native 2 + producer 2 + zero 2 + pattern fold 2 + 3 arms x (2
capture + 2 arm) = 20 forwards; 0 backwards; 0 fits. Bar <= 30.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_constant_pattern_v12_result.json"
V11B = ROOT / "circuits/followups/aspectual_anchor_dod_cue_source_v11b_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_constant_pattern_v12"
TOKENS = {"has": 468, "had": 550}
REPLAY_TOL, CV_MAX, TRANSFER_MIN, POOLED_MIN = 0.02, 0.35, 0.60, 0.50
FORWARDS_MAX = 30
HEAD = L.Component("attn8_h1_final", 8, "attn", (1,), "final")


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "head": "8.1",
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "stored_constants": 2, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "cv_max": CV_MAX, "transfer_min": TRANSFER_MIN, "pooled_min": POOLED_MIN}}


def is_cue(row, s):
    return s == row.source_positions[0] - 1


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    v11b = json.loads(V11B.read_text())
    ref_retention = v11b["arms"]["cue_inherited"]["retention"]
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

    # fold: the native pattern scalar on the cue for every row
    pattern = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        out, lamb = L.head_source_terms(fw, chunk, HEAD, fw.directions); forwards += 1
        for row, entry in zip(chunk, out):
            pattern[row.row_id] = entry[1]["pattern"][row.source_positions[0] - 1]
    groups = {}
    for row in rows:
        groups.setdefault((row.construction, row.present), []).append(pattern[row.row_id])
    stats = {f"{c}/{'since' if pres else 'by'}": {"median": statistics.median(v), "mean": statistics.mean(v),
                                                  "stdev": statistics.pstdev(v), "cv": statistics.pstdev(v) / abs(statistics.mean(v))}
             for (c, pres), v in groups.items()}
    median = {(c, pres): statistics.median(v) for (c, pres), v in groups.items()}

    def arm_with(override):
        nonlocal forwards
        table = {}
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]
            table.update(L.source_restricted_slices(fw, chunk, HEAD, is_cue, ("inherited",), override)); forwards += 1
        fw.subtract = table
        arm, n = v1._run_arm(fw, rows, components=(HEAD,), mode="replace"); forwards += n
        return arm

    arms = {"native_p": arm_with(None),
            "fronted_median_p": arm_with(lambda r, s: median[("fronted", r.present)] if is_cue(r, s) else None),
            "report_median_p": arm_with(lambda r, s: median[("report", r.present)] if is_cue(r, s) else None)}
    fw.use_subtract = False

    def retention(arm, idx):
        sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
        z = L.summarize(sub_rows, sub_native, [zero[i] for i in idx])["target_damage_mean"]
        a = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])["target_damage_mean"]
        return {"zero_damage": z, "arm_damage": a, "retention": 1.0 - a / z}

    all_idx = list(range(len(rows)))
    report_idx = [i for i, r in enumerate(rows) if r.construction == "report"]
    fronted_idx = [i for i, r in enumerate(rows) if r.construction == "fronted"]
    scored = {"native_p_all": retention(arms["native_p"], all_idx),
              "fronted_median_on_report": retention(arms["fronted_median_p"], report_idx),
              "report_median_on_fronted": retention(arms["report_median_p"], fronted_idx)}
    # pooled cross-construction: report rows under the fronted constant + fronted rows under the report constant
    cross = [arms["fronted_median_p"][i] if rows[i].construction == "report" else arms["report_median_p"][i] for i in all_idx]
    scored["cross_pooled"] = retention(cross, all_idx)
    for k, v in scored.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    print("pattern stats", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in stats.items()}))
    predictions = {
        "pred_a_native_cue_term_replays_v11b": abs(scored["native_p_all"]["retention"] - ref_retention) <= REPLAY_TOL,
        "pred_b_pattern_is_stable_within_cue": all(v["cv"] <= CV_MAX for v in stats.values()),
        "pred_c_constant_pattern_transfers_to_report_rows": scored["fronted_median_on_report"]["retention"] >= TRANSFER_MIN,
        "pred_d_constant_pattern_transfers_to_fronted_rows": scored["report_median_on_fronted"]["retention"] >= TRANSFER_MIN,
        "pred_e_token_only_table_retains_half_pooled": scored["cross_pooled"]["retention"] >= POOLED_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_constant_pattern_result_v12", "candidate_id": CANDIDATE_ID, "plan": _plan(rows),
              "lambda": lamb, "pattern_stats": stats, "pattern_by_row": pattern, "scored": scored,
              "v11b_cue_inherited_retention": ref_retention, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
