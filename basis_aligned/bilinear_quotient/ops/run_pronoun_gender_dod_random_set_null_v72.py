#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_set_beats_every_random_quadruple pred_c_random_quadruples_are_not_live pred_d_set_fraction_within_band_of_v71
"""Pronoun gender he/she DoD (v72): matched-count random four-head-set null on the v71 fresh rows (SIMPLE, better_circuits §1).

Lane: Claude circuit lane. Parent: v71 (set {10.1, 9.6, 12.4, 15.1}: 0.96–1.11 of the margin on fresh rows, selective,
additive, keep-only retention 1.37; frozen band failed upward). Same rows (fresh for v71, now opened once: the set was NOT
selected on them, but the band below is re-frozen from them). Null: 16 random quadruples drawn from the 158 other heads, each
removed along its own weight-only `O_h^T (u_he - u_she)` direction exactly as the claimed set is; a quadruple that is
"live" (fraction >= 0.10, positive >= 0.75) would show the description length "four heads" buys nothing.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native        manual forward = producer native <= 1e-4
    pred_b_set_beats_every_random_quadruple set damage > max over the 16 random quadruples
    pred_c_random_quadruples_are_not_live   no random quadruple passes LIVE
    pred_d_set_fraction_within_band_of_v71  pooled set fraction within 1.02 +/- 0.05 of v71 (replay on identical rows; a
                                            miss would be an instrument fault, not a finding)

PRICE (registered maximum): 2 batches x (native + producer + set + 16 random) = 38 forwards; 0 backwards; 0 fits. Bar <= 40.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_random_head_set_null_v25 as v25
import run_pronoun_gender_dod_battery_v71 as v71
import dod_battery

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/pronoun_gender_dod_random_set_null_v72_result.json"
CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_random_set_null_v72"
EXCLUDED = set(v71.HEADS)
POOL = [(l, h) for l in range(18) for h in range(9) if (l, h) not in EXCLUDED]
SETS = [tuple(sorted(random.Random(2026_09_18_72 + s).sample(POOL, 4))) for s in range(16)]
LIVE_FRACTION, LIVE_POSITIVE, V71_FRACTION, BAND, INSTRUMENT_TOL = 0.10, 0.75, 1.02, 0.05, 1e-4
FORWARDS_MAX = 40


def main() -> None:
    rows, he, she = v71.build()
    spec = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, v71.HEADS)
    SET = spec.set_components()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": list(v71.HEADS), "random_sets": SETS,
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "v71_fraction": V71_FRACTION, "band": BAND}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SET + tuple(c for s in SETS for c in v25.components_for(s)), he, she)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    comp = L.summarize(rows, native, arm)
    randoms = []
    for s in SETS:
        arm, n = v1._run_arm(fw, rows, components=v25.components_for(s), mode="project"); forwards += n
        sm = L.summarize(rows, native, arm)
        randoms.append({"set": [f"{l}.{h}" for l, h in s], "damage": sm["target_damage_mean"], "fraction": sm["target_damage_fraction"],
                        "live": sm["target_damage_fraction"] >= LIVE_FRACTION and sm["target_damage_positive_fraction"] >= LIVE_POSITIVE})
    dmg = [r["damage"] for r in randoms]
    print("set", round(comp["target_damage_mean"], 3), "fraction", round(comp["target_damage_fraction"], 3), "random max", round(max(dmg), 3), "live randoms", sum(r["live"] for r in randoms))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_set_beats_every_random_quadruple": comp["target_damage_mean"] > max(dmg),
                   "pred_c_random_quadruples_are_not_live": not any(r["live"] for r in randoms),
                   "pred_d_set_fraction_within_band_of_v71": abs(comp["target_damage_fraction"] - V71_FRACTION) <= BAND}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_gender_dod_random_set_null_result_v72", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument,
                               "set": comp, "random": randoms, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
