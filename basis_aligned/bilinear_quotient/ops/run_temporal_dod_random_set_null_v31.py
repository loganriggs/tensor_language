#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_set_exceeds_random_median_tenfold
"""Temporal will/had DoD battery, step 4 (v31): matched-count random four-head-set null for SIMPLE.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v28 (S = {11.3, 9.1, 15.5, 9.4}, 81%).
Same design as the aspectual v25: sixteen seeded random four-head sets from the 158 other heads, each removed
along its own weight-only will/had readout directions on the v28 fresh rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_set_beats_every_random_quadruple   S damage > max random damage
    pred_c_random_quadruples_are_not_live     no random set passes LIVE (>= 0.10, positive >= 0.75)
    pred_d_set_exceeds_random_median_tenfold  S damage >= 10 x median |random damage|

PRICE (registered maximum): native 2 + producer 2 + S 2 + 16 x 2 = 38 forwards; 0 backwards; 0 fits. Bar <= 44.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import random
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28
import run_aspectual_dod_random_head_set_null_v25 as v25

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_random_set_null_v31_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_random_set_null_v31"
LIVE_FRACTION, LIVE_POSITIVE, TENFOLD, INSTRUMENT_TOL = 0.10, 0.75, 10.0, 1e-4
FORWARDS_MAX = 44
EXCLUDED = {(11, 3), (9, 1), (15, 5), (9, 4)}
POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in EXCLUDED]
SETS = [tuple(sorted(random.Random(2026_09_17_31 + s).sample(POOL, 4))) for s in range(16)]


def main() -> None:
    rows = v28.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "random_sets": SETS, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    all_components = v28.SET + tuple(c for s in SETS for c in v25.components_for(s))
    fw.directions = L.readout_directions(backend.model, all_components, v28.WILL, v28.HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=v28.SET, mode="project"); forwards += n
    comp = L.summarize(rows, native, arm)
    randoms = []
    for s in SETS:
        arm, n = v1._run_arm(fw, rows, components=v25.components_for(s), mode="project"); forwards += n
        sm = L.summarize(rows, native, arm)
        randoms.append({"set": [f"{l}.{h}" for l, h in s], "damage": sm["target_damage_mean"], "fraction": sm["target_damage_fraction"], "positive": sm["target_damage_positive_fraction"],
                        "live": sm["target_damage_fraction"] >= LIVE_FRACTION and sm["target_damage_positive_fraction"] >= LIVE_POSITIVE})
    dmg = [r["damage"] for r in randoms]; med = sorted(abs(x) for x in dmg)[len(dmg) // 2]
    print("S", round(comp["target_damage_mean"], 4), "random max", round(max(dmg), 4), "median|.|", round(med, 4))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_set_beats_every_random_quadruple": comp["target_damage_mean"] > max(dmg),
                   "pred_c_random_quadruples_are_not_live": not any(r["live"] for r in randoms), "pred_d_set_exceeds_random_median_tenfold": comp["target_damage_mean"] >= TENFOLD * med}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_random_set_null_result_v31", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "set": comp, "random": randoms,
              "random_damage_max": max(dmg), "random_damage_median_abs": med, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
