#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_own_direction_replays_v76 pred_c_were_was_direction_carries_half_of_own pred_d_were_was_removal_live_and_beats_null pred_e_gender_direction_removes_less_than_number
"""Pronoun number they/he DoD (v96): does the VERB-agreement number direction (were−was) carry the pronoun-number readout at {9.6, 12.4, 15.1, 10.5}?

Lane: Claude circuit lane. Parents: v95 (weights only: at 9.6 / 15.1 / 8.8 / 11.3 the they−he and were−was readout directions have
|cos| 0.38–0.47; at the unembedding 0.11), v76 (own-direction removal 76% on fresh rows). Edit on the v76 fresh rows, at the same four
heads, along three weight-only directions: own O_h^T(u_they − u_he) (replay), the verb-number direction O_h^T(u_were − u_was), and the
gender direction O_h^T(u_he − u_she) (a same-family control that shares the "he" component). Null: norm-matched random direction x 16
for the were−was arm. If the verb-number direction removes half of what the own direction removes, pronoun number and verb agreement
share one number axis at these heads (a cross-family component), which v70 showed for tense/mood/aspect at the temporal heads.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native                 <= 1e-4
    pred_b_own_direction_replays_v76                 pooled own fraction within 0.76 +/- 0.05
    pred_c_were_was_direction_carries_half_of_own    were−was damage >= 0.50 x own damage. Prior: unsure.
    pred_d_were_was_removal_live_and_beats_null      fraction >= 0.10, positive >= 0.75, > max of 16 null means
    pred_e_gender_direction_removes_less_than_number he−she damage < were−was damage. Prior: unsure (they−he and he−she share "he").

PRICE (registered maximum): 3 batches x (native + producer + 3 arms + 16 nulls) = 63 forwards; 0 backwards; 0 fits. Bar <= 66.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_number_dod_battery_v76 as v76
import dod_battery

ROOT = v76.dod_battery.ROOT
OUT = ROOT / "circuits/followups/pronoun_number_dod_shared_number_axis_v96_result.json"
CANDIDATE_ID = "pronoun_number.they_vs_he.dod_shared_number_axis_v96"
NULL_SEEDS = tuple(range(3201, 3217))
V76_FRACTION, BAND, HALF, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = 0.76, 0.05, 0.50, 0.10, 0.75, 1e-4
FORWARDS_MAX = 66
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_own_direction_replays_v76": "0.76 +/- 0.05", "pred_c_were_was_direction_carries_half_of_own": ">= 0.50 x own",
               "pred_d_were_was_removal_live_and_beats_null": "live, > null max", "pred_e_gender_direction_removes_less_than_number": "he-she < were-was"}
ARMS = {"own_they_he": (" they", " he"), "number_were_was": (" were", " was"), "gender_he_she": (" he", " she")}


def main() -> None:
    rows, they, he, agents, objects = v76.build()
    SET = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, they, he, v76.HEADS).set_components()
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "set": list(v76.HEADS), "arms": {k: list(v) for k, v in ARMS.items()}, "null_seeds": list(NULL_SEEDS),
            "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"v76_fraction": V76_FRACTION, "band": BAND, "half": HALF, "live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "instrument_tol": INSTRUMENT_TOL}}
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
    fw.directions = L.readout_directions(backend.model, SET, L._single(" were"), L._single(" was"))
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm)["target_damage_mean"])
    own, num, gen = report["own_they_he"], report["number_were_was"], report["gender_he_she"]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_own_direction_replays_v76": abs(own["target_damage_fraction"] - V76_FRACTION) <= BAND,
                   "pred_c_were_was_direction_carries_half_of_own": num["target_damage_mean"] >= HALF * own["target_damage_mean"],
                   "pred_d_were_was_removal_live_and_beats_null": num["target_damage_fraction"] >= LIVE_FRACTION and num["target_damage_positive_fraction"] >= LIVE_POSITIVE and num["target_damage_mean"] > max(nulls),
                   "pred_e_gender_direction_removes_less_than_number": gen["target_damage_mean"] < num["target_damage_mean"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "pronoun_number_dod_shared_number_axis_result_v96", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": report,
                               "null_damage_max": max(nulls), "null_damages": nulls, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "null_max": max(nulls), "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
