#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_attn9_midpoint_is_live_and_beats_null pred_d_attn9_midpoint_is_selective_over_null pred_e_attn5_midpoint_beats_null pred_f_mlp4_midpoint_beats_null_and_is_selective pred_g_singles_compose_additively
"""Aspectual has/had definition-of-done battery, step 2: MIDPOINT removal with a norm-matched null.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent receipt:
`circuits/followups/aspectual_anchor_dod_removal_v1_result.json` (edit, fresh rows).

WHY A V2. v1 zeroed each component's whole write. The 16 equal-norm random-direction nulls showed
that for attn5 H7/H1/H6/H8 the target damage (0.94) is entirely a norm effect (null median 1.05,
max 1.36) and that for mlp4 half of it is (null median 0.21 of 0.41). Whole-write removal
conflates "this write carries aspect" with "this write is large". The aspect content of a write
is its exact paired difference: for every fresh row the partner row differs only in the cue
(since <-> by), so `d = write(row) - write(partner)` at the same slice/position is a weight-and-
activation-exact object with no fit. Moving the write to the since/by MIDPOINT (`w - d/2`)
removes exactly the cue-dependent half of that slice and nothing shared; the null subtracts a
random vector of norm `|d|/2` at the same slice, so it is matched in removed norm and site.

ROWS, COMPONENTS, READERS: identical to v1 (64 fresh rows, sha 5ec7d32f..., five components,
has-had target, was-were / who-which / night-day unrelated readers). Rows were opened by v1 for
whole-write removal only; no selection step used them.

REGISTERED BARS
    LIVE   target_damage_fraction >= 0.10 AND damage positive on >= 0.75 of rows
    NULL   mean damage > MAX of the 16 random-direction null mean damages
    GATE   for each unrelated reader: mean|move| <= null-mean|move| + 0.25 x target damage
           (movement in EXCESS of the norm-matched null; the v1 gate ignored the null baseline)
    ADD    |joint_midpoint_damage - sum(single midpoint damages)| <= 0.25 x min(single damages
           among LIVE components)

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native    no-edit forward matches producer.native within 1e-4
    pred_b_native_capability            all four construction x cue cells >= 0.85
    pred_c_attn9_midpoint_is_live_and_beats_null
                                        attn9 H1/H4 midpoint passes LIVE and NULL. Prior: yes;
                                        v1 whole-write damage 0.70 vs null max 0.105.
    pred_d_attn9_midpoint_is_selective_over_null
                                        attn9 H1/H4 midpoint passes GATE on all three readers.
    pred_e_attn5_midpoint_beats_null    attn5 H7/H1/H6/H8 midpoint passes NULL (its aspect
                                        content is small but specific; the v1 failure was norm).
                                        Prior: uncertain -- the released path gives attn5 only
                                        ~4-5% recovery, so LIVE may fail while NULL passes.
    pred_f_mlp4_midpoint_beats_null_and_is_selective
                                        mlp4 source-bank midpoint passes NULL and GATE.
    pred_g_singles_compose_additively   ADD holds for the joint midpoint of all five. This is the
                                        first COMPOSES-property measurement on this path; the
                                        random-split null for it is deferred to v3.

PRICE (registered maximum): capture 2 forwards + native 2 + producer.native 2 + 5 x (1 midpoint
+ 16 nulls) x 2 = 170 + joint 2 = 178 forwards; 0 backwards; 0 fits. Bar <= 200.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_removal_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_removal_v2"
EXPECTED_ROWS_SHA256 = v1.EXPECTED_ROWS_SHA256
NULL_SEEDS = tuple(range(101, 117))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, ADD_RATIO, CAPABILITY_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.25, 0.85, 1e-4
FORWARDS_MAX = 200


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "components": [c.name for c in L.COMPONENTS], "null_seeds": list(NULL_SEEDS),
            "mode": "midpoint", "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE,
                     "gate_ratio_over_null": GATE_RATIO, "add_ratio": ADD_RATIO,
                     "capability_min": CAPABILITY_MIN, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("fresh rows changed; refusing to run against an unregistered panel")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0

    store = {}
    for chunk in v1._batches(rows):
        store.update(fw.capture(chunk, L.COMPONENTS)); forwards += 1
    fw.deltas = L.paired_deltas(store, rows, L.COMPONENTS)
    delta_norms = {c.name: sum(float(d.float().norm()) for (rid, name, pos, head), d in fw.deltas.items()
                               if name == c.name) / sum(1 for k in fw.deltas if k[1] == c.name)
                   for c in L.COMPONENTS}
    write_norms = {c.name: sum(float(w.float().norm()) for (rid, name, pos, head), w in store.items()
                               if name == c.name) / sum(1 for k in store if k[1] == c.name)
                   for c in L.COMPONENTS}

    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    pred_a = instrument_err <= INSTRUMENT_TOL
    cells = {}
    for r, e in zip(rows, native):
        cells.setdefault(f"{r.construction}/{'since' if r.present else 'by'}", []).append(
            1.0 if e["answer"] > e["foil"] else 0.0)
    capability = {k: sum(v) / len(v) for k, v in cells.items()}
    pred_b = all(v >= CAPABILITY_MIN for v in capability.values())

    report = {}
    for comp in L.COMPONENTS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="midpoint"); forwards += n
        summary = L.summarize(rows, native, arm)
        nulls = []
        for seed in NULL_SEEDS:
            null_arm, n = v1._run_arm(fw, rows, components=(comp,), mode="midpoint_random", seed=seed)
            forwards += n
            nulls.append(L.summarize(rows, native, null_arm))
        null_damages = [s["target_damage_mean"] for s in nulls]
        null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
        live = (summary["target_damage_fraction"] >= LIVE_FRACTION
                and summary["target_damage_positive_fraction"] >= LIVE_POSITIVE)
        beats_null = summary["target_damage_mean"] > max(null_damages)
        gates = {name: summary[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * summary["target_damage_mean"]
                 for name in L.UNRELATED}
        report[comp.name] = {"midpoint": summary, "live": live, "beats_null": beats_null,
                             "gates": gates, "selective": all(gates.values()),
                             "null_damage_means": null_damages, "null_damage_max": max(null_damages),
                             "null_damage_median": sorted(null_damages)[len(null_damages) // 2],
                             "null_unrelated_abs_move_mean": null_moves,
                             "mean_half_delta_norm": 0.5 * delta_norms[comp.name],
                             "mean_write_norm": write_norms[comp.name]}
        print(comp.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}),
              "live", live, "beats_null", beats_null, "gates", gates,
              "half_delta/write", round(0.5 * delta_norms[comp.name] / write_norms[comp.name], 4))
    joint, n = v1._run_arm(fw, rows, components=L.COMPONENTS, mode="midpoint"); forwards += n
    joint_summary = L.summarize(rows, native, joint)
    singles = {k: v["midpoint"]["target_damage_mean"] for k, v in report.items()}
    live_names = [k for k, v in report.items() if v["live"]]
    add_gap = abs(joint_summary["target_damage_mean"] - sum(singles.values()))
    add_bar = ADD_RATIO * (min(singles[k] for k in live_names) if live_names else 0.0)

    a9, a5, m4 = report["attn9_h1_h4_final"], report["attn5_h7_h1_h6_h8_final"], report["mlp4_source_bank"]
    predictions = {
        "pred_a_instrument_replays_native": bool(pred_a),
        "pred_b_native_capability": bool(pred_b),
        "pred_c_attn9_midpoint_is_live_and_beats_null": bool(a9["live"] and a9["beats_null"]),
        "pred_d_attn9_midpoint_is_selective_over_null": bool(a9["selective"]),
        "pred_e_attn5_midpoint_beats_null": bool(a5["beats_null"]),
        "pred_f_mlp4_midpoint_beats_null_and_is_selective": bool(m4["beats_null"] and m4["selective"]),
        "pred_g_singles_compose_additively": bool(live_names) and add_gap <= add_bar,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_removal_result_v2", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err, "capability": capability,
              "native_mean_margin": L.summarize(rows, native, native)["target_native_mean_margin"],
              "components": report, "joint_all_five": joint_summary, "live_components": live_names,
              "additivity": {"joint": joint_summary["target_damage_mean"], "sum_singles": sum(singles.values()),
                             "gap": add_gap, "bar": add_bar, "singles": singles},
              "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "live": live_names, "additivity": result["additivity"],
                      "forwards": forwards, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
