#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_own_direction_replays_v76 pred_c_person_direction_removes_little pred_d_person_direction_not_live pred_e_own_beats_null
"""Reflexive person I/you DoD (v135): SHARED HEADS, SEPARATE DIRECTIONS? v134's construction on this line: remove at the line's own set along the pronoun-number they-he contrast (a family sharing heads with this one) and compare with the own direction; null = norm-matched random direction x 16. The arm named "person_myself_yourself" in the receipt is the OTHER-family direction (pronoun-number they-he); "own_they_he" is this line's own contrast (names kept from the v134 body).
PREDICTIONS: as v134 (instrument; own replays the battery fraction +/- 0.05; other-direction damage <= 0.25 x own; other direction not live; own > null max).
PRICE (registered maximum): 3 batches x (native + producer + 2 arms + 16 nulls) = 60 forwards; 0 backwards; 0 fits. Bar <= 63.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_person_reflexive_dod_battery_v104 as v76
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/person_reflexive_dod_cross_family_direction_v135_result.json"
CANDIDATE_ID = v76.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_cross_family_direction_v135"
NULL_SEEDS = tuple(range(13501, 13517))
V76_FRACTION, BAND, LITTLE, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = 0.42, 0.05, 0.25, 0.10, 0.75, 1e-4
FORWARDS_MAX = 63
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_own_direction_replays_v76": "0.76 +/- 0.05", "pred_c_person_direction_removes_little": "<= 0.25 x own",
               "pred_d_person_direction_not_live": "not live", "pred_e_own_beats_null": "> null max"}
ARMS = {"own_they_he": (" myself", " yourself"), "person_myself_yourself": (" they", " he")}


def main() -> None:
    rows, they, he, agents, objects = v76.build()
    SET = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, they, he, v76.HEADS).set_components()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": list(v76.HEADS), "arms": {k: list(v) for k, v in ARMS.items()}, "null_seeds": list(NULL_SEEDS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"v76_fraction": V76_FRACTION, "band": BAND, "little": LITTLE, "live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    report = {}
    for name, (a, b) in ARMS.items():
        fw.directions = L.readout_directions(backend.model, SET, L._single(a), L._single(b))
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
        report[name] = L.summarize(rows, native, arm)
        print(name, "damage", round(report[name]["target_damage_mean"], 3), "fraction", round(report[name]["target_damage_fraction"], 3), "positive", report[name]["target_damage_positive_fraction"])
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm)["target_damage_mean"])
    own, per = report["own_they_he"], report["person_myself_yourself"]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_own_direction_replays_v76": abs(own["target_damage_fraction"] - V76_FRACTION) <= BAND,
                   "pred_c_person_direction_removes_little": per["target_damage_mean"] <= LITTLE * own["target_damage_mean"],
                   "pred_d_person_direction_not_live": not (per["target_damage_fraction"] >= LIVE_FRACTION and per["target_damage_positive_fraction"] >= LIVE_POSITIVE),
                   "pred_e_own_beats_null": own["target_damage_mean"] > max(nulls)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_cross_family_direction_result_v134", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": report,
                               "null_damage_max": max(nulls), "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "null_max": max(nulls), "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
