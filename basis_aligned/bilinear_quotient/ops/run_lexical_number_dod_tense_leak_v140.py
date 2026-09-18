#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_head_11_3_carries_the_tense_move pred_c_set_without_11_3_passes_the_tense_gate pred_d_set_without_11_3_still_live
"""Lexical number were/was DoD (v140): is the tense-reader leak on natural rows (v138: has-had moves 0.59 vs null 0.26 under the set removal) head
11.3's? On the v138 natural rows (64), remove along O_h^T(u_were - u_was) for three arms: the full set {11.3, 5.7, 7.8, 9.7}, 11.3 alone, and the set
without 11.3 {5.7, 7.8, 9.7}; null = norm-matched random direction x 16 for the set-without-11.3 arm. 11.3 is the head where the number and tense
readout directions are entangled in weight space (v69 / v116); if its removal carries the has-had move, the number component's selectivity failure
on natural text is that entanglement and the three-head sub-set is the selective part.
PREDICTIONS (scored as written; failures preserved; congruent rows = plural/were, singular/was)
    pred_a_instrument_replays_native            <= 1e-4
    pred_b_head_11_3_carries_the_tense_move     the 11.3-alone arm's has-had move >= 0.70 x the full set's has-had move (congruent rows)
    pred_c_set_without_11_3_passes_the_tense_gate  set-without-11.3: has-had move <= null move + 0.25 x its own damage (congruent rows)
    pred_d_set_without_11_3_still_live          set-without-11.3: fraction >= 0.10, positive >= 0.75 and damage > null max (congruent rows). Prior: unsure.
PRICE (registered maximum): 2 batches x (native + producer + 3 arms + 16 nulls) = 42 forwards; 0 backwards; 0 fits. Bar <= 46.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55   # sets the number line's readers (has-had tense, who-which, night-day)
import dod_natural_line as N
import dod_battery

ROOT = dod_battery.ROOT
ROWS = ROOT / "circuits/followups/lexical_number_dod_natural_rows_v138.json"
OUT = ROOT / "circuits/followups/lexical_number_dod_tense_leak_v140_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_tense_leak_v140"
ARMS = {"full_set": ((11, 3), (5, 7), (7, 8), (9, 7)), "head_11_3": ((11, 3),), "set_without_11_3": ((5, 7), (7, 8), (9, 7))}
NULL_SEEDS = tuple(range(4001, 4017))
SHARE_MIN, GATE_RATIO, LIVE_FRACTION, LIVE_POSITIVE, INSTRUMENT_TOL = 0.70, 0.25, 0.10, 0.75, 1e-4
FORWARDS_MAX = 46
CONGRUENT = ("natural_plural_were", "natural_singular_was")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_head_11_3_carries_the_tense_move": ">= 0.70 x full", "pred_c_set_without_11_3_passes_the_tense_gate": "<= null + 0.25 x damage", "pred_d_set_without_11_3_still_live": "live, > null"}


def main() -> None:
    rows, sha, were, was = N.rows_from_receipt(ROWS, "were", " were", " was", lambda r: f"natural_{r['cue']}_{r['label']}")
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "arms": {k: list(v) for k, v in ARMS.items()}, "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"share_min": SHARE_MIN, "gate_ratio": GATE_RATIO, "live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    idx = [i for i, r in enumerate(rows) if r.construction in CONGRUENT]
    sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
    comps = {k: dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, were, was, tuple(h)).set_components() for k, h in ARMS.items()}
    fw.directions = L.readout_directions(backend.model, tuple(c for cs in comps.values() for c in cs), were, was)
    report = {}
    for k, cs in comps.items():
        arm, n = v1._run_arm(fw, rows, components=cs, mode="project"); forwards += n
        report[k] = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])
        print(k, "damage", round(report[k]["target_damage_mean"], 3), "fraction", round(report[k]["target_damage_fraction"], 3), "positive", round(report[k]["target_damage_positive_fraction"], 3), "has-had move", round(report[k]["tense_has_had_abs_move_mean"], 3))
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=comps["set_without_11_3"], mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(sub_rows, sub_native, [arm[i] for i in idx]))
    null_max = max(s["target_damage_mean"] for s in nulls); null_move = sum(s["tense_has_had_abs_move_mean"] for s in nulls) / len(nulls)
    S = report["set_without_11_3"]
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_head_11_3_carries_the_tense_move": report["head_11_3"]["tense_has_had_abs_move_mean"] >= SHARE_MIN * report["full_set"]["tense_has_had_abs_move_mean"],
                   "pred_c_set_without_11_3_passes_the_tense_gate": S["tense_has_had_abs_move_mean"] <= null_move + GATE_RATIO * S["target_damage_mean"],
                   "pred_d_set_without_11_3_still_live": S["target_damage_fraction"] >= LIVE_FRACTION and S["target_damage_positive_fraction"] >= LIVE_POSITIVE and S["target_damage_mean"] > null_max}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "lexical_number_dod_tense_leak_result_v140", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "arms": report, "null_damage_max": null_max,
                               "null_tense_move": null_move, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "null_max": null_max, "null_tense_move": null_move, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
