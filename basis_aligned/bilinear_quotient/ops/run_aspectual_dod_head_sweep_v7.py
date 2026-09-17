#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_heads_9_1_and_9_4_rank_top3 pred_c_no_head_outside_blocks_9_to_15_exceeds_half_of_9_4 pred_d_pair_is_near_additive pred_e_at_most_five_heads_are_live
"""Aspectual has/had definition-of-done battery, step 7: STABLE IDENTIFICATION / simplicity null.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v4/v5 (attention9 H1/H4
weight-only readout removal is live, null-beating, selective and template-stable).

WHY. better_circuits §1 SIMPLE: the component's description length is a count (here: layer 9,
heads 1 and 4, the has/had contrast -- four integers over native weights) and its null is "the
same count for a random component of matching effect size". The honest way to price that is
to run the identical weight-only readout removal for EVERY head of the model singly and see how
special {9.1, 9.4} are: how many heads reach comparable damage, and where they sit. This also
serves stable identification (better_circuits §1, communicating_results §7): the pair must be
recoverable from a blind sweep, not only from the path narrative that named it.

ROWS: the 64 discovery-shaped rows (opened for removal). Removal per head h in block l: subtract
the projection of the pre-c_proj slice onto `O_h^T (u_has - u_had)` at the final query. Damage
as before (oriented has-had margin drop, mean over rows); no null arms here -- v4 already
established the norm-matched null for the claimed heads, and this sweep IS the specificity null.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   no-edit forward matches producer.native <= 1e-4
    pred_b_heads_9_1_and_9_4_rank_top3 both 9.1 and 9.4 are among the three largest damages
    pred_c_no_head_outside_blocks_9_to_15_exceeds_half_of_9_4
                                       every head in blocks 0-8, 16, 17 has damage < 0.5 x dmg(9.4)
    pred_d_pair_is_near_additive       |dmg(9.1 + 9.4 jointly, v4 = 0.7011) - dmg(9.1) - dmg(9.4)|
                                       <= 0.25 x min(dmg(9.1), dmg(9.4))
    pred_e_at_most_five_heads_are_live at most five of the 162 heads pass LIVE (>= 10% fraction
                                       and positive on >= 75% of rows). Prior: unsure; this is the
                                       simplicity price.

PRICE (registered maximum): native 2 + producer 2 + 162 x 2 = 328 forwards; 0 backwards; 0 fits.
Bar <= 340.
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
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_head_sweep_v7_result.json"
V4 = ROOT / "circuits/followups/aspectual_anchor_dod_readout_removal_v4_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_head_sweep_v7"
EXPECTED_ROWS_SHA256 = v1.EXPECTED_ROWS_SHA256
TOKENS = {"has": 468, "had": 550}
LIVE_FRACTION, LIVE_POSITIVE, HALF, ADD_RATIO, LIVE_MAX, INSTRUMENT_TOL = 0.10, 0.75, 0.5, 0.25, 5, 1e-4
FORWARDS_MAX = 340
HEADS = tuple(L.Component(f"attn{layer}_h{head}", layer, "attn", (head,), "final")
              for layer in range(18) for head in range(9))


def _plan(rows):
    return {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows),
            "heads": len(HEADS), "mode": "project_onto_O_h^T(u_has-u_had)", "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "half": HALF,
                     "add_ratio": ADD_RATIO, "live_max": LIVE_MAX, "instrument_tol": INSTRUMENT_TOL}}


def main() -> None:
    rows = L.build_rows()
    if L.rows_sha256(rows) != EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed; refusing to run against an unregistered panel")
    v4 = json.loads(V4.read_text())
    joint_9 = v4["components"]["attn9_h1_h4_final"]["project"]["target_damage_mean"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(rows), indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, HEADS, TOKENS["has"], TOKENS["had"])
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument_err = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    table = {}
    for comp in HEADS:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        s = L.summarize(rows, native, arm)
        table[comp.name] = {"layer": comp.layer, "head": comp.heads[0], "damage": s["target_damage_mean"],
                            "fraction": s["target_damage_fraction"], "positive": s["target_damage_positive_fraction"],
                            "live": s["target_damage_fraction"] >= LIVE_FRACTION and s["target_damage_positive_fraction"] >= LIVE_POSITIVE,
                            "number_move": s["number_was_were_abs_move_mean"]}
    ranked = sorted(table, key=lambda k: -table[k]["damage"])
    top3 = ranked[:3]
    d91, d94 = table["attn9_h1"]["damage"], table["attn9_h4"]["damage"]
    outside = [k for k in table if table[k]["layer"] < 9 or table[k]["layer"] > 15]
    live = [k for k in table if table[k]["live"]]
    predictions = {
        "pred_a_instrument_replays_native": bool(instrument_err <= INSTRUMENT_TOL),
        "pred_b_heads_9_1_and_9_4_rank_top3": "attn9_h1" in top3 and "attn9_h4" in top3,
        "pred_c_no_head_outside_blocks_9_to_15_exceeds_half_of_9_4": all(table[k]["damage"] < HALF * d94 for k in outside),
        "pred_d_pair_is_near_additive": abs(joint_9 - d91 - d94) <= ADD_RATIO * min(d91, d94),
        "pred_e_at_most_five_heads_are_live": len(live) <= LIVE_MAX,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_head_sweep_result_v7", "candidate_id": CANDIDATE_ID,
              "plan": _plan(rows), "instrument_max_abs_error": instrument_err, "heads": table,
              "ranked": ranked, "top10": [(k, round(table[k]["damage"], 4)) for k in ranked[:10]],
              "live_heads": live, "v4_joint_9_1_9_4": joint_9,
              "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "top10": result["top10"], "live": live,
                      "joint_vs_sum": [joint_9, d91 + d94], "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
