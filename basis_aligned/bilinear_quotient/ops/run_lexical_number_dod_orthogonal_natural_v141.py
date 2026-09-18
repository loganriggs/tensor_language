#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_orthogonalized_set_still_live pred_c_orthogonalized_set_spares_tense pred_d_other_readers_still_within_gate pred_e_damage_retained_at_least_060
"""Lexical number were/was DoD (v141): v57's ORTHOGONALIZED readout removal on the NATURAL rows (v138, congruent cells plural/were, singular/was).
Parents: v57 (on fresh rows the were-was direction orthogonalized against O_h^T(u_has - u_had) at each head stayed live and spared tense),
v138 / v140 (on natural rows the native-direction removal moves has-had 0.59 vs null 0.26, mostly through 11.3). Arms: native-direction set removal
(replay of v138's congruent score) and the orthogonalized set removal; 16 norm-matched nulls for the orthogonalized arm.
PREDICTIONS (scored on the 32 congruent rows; failures preserved)
    pred_a_instrument_replays_native        <= 1e-4
    pred_b_orthogonalized_set_still_live    fraction >= 0.10, positive >= 0.75, > max null
    pred_c_orthogonalized_set_spares_tense  has-had move <= null move + 0.25 x its damage. Prior: unsure -- v140 says the leak is 11.3's,
                                            and v69 says the entanglement is exactly at 11.3, so the orthogonalization should catch it.
    pred_d_other_readers_still_within_gate  who-which and night-day within their gates
    pred_e_damage_retained_at_least_060     orthogonalized damage >= 0.60 x the native-direction damage
PRICE (registered maximum): 2 batches x (native + producer + 2 arms + 16 nulls) = 40 forwards; 0 backwards; 0 fits. Bar <= 44.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55
import dod_natural_line as N
import dod_battery

ROOT = dod_battery.ROOT
ROWS = ROOT / "circuits/followups/lexical_number_dod_natural_rows_v138.json"
OUT = ROOT / "circuits/followups/lexical_number_dod_orthogonal_natural_v141_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_orthogonal_natural_v141"
NULL_SEEDS = tuple(range(4101, 4117))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.60, 1e-4
FORWARDS_MAX = 44
CONGRUENT = ("natural_plural_were", "natural_singular_was")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_orthogonalized_set_still_live": "live, > null", "pred_c_orthogonalized_set_spares_tense": "<= null + 0.25 x damage",
               "pred_d_other_readers_still_within_gate": "two gates", "pred_e_damage_retained_at_least_060": ">= 0.60 x native"}


def main() -> None:
    rows, sha, were, was = N.rows_from_receipt(ROWS, "were", " were", " was", lambda r: f"natural_{r['cue']}_{r['label']}")
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "set": ["11.3", "5.7", "7.8", "9.7"], "null_seeds": list(NULL_SEEDS), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"live_fraction": LIVE_FRACTION, "live_positive": LIVE_POSITIVE, "gate_ratio": GATE_RATIO, "retain_min": RETAIN_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    num = L.readout_directions(backend.model, v55.SINGLES, were, was)
    tense = L.readout_directions(backend.model, v55.SINGLES, L._single(" has"), L._single(" had"))
    orth, cosines = {}, {}
    for key in num:
        a, b = num[key].float(), tense[key].float(); bh = b / b.norm()
        orth[key] = a - (a @ bh) * bh; cosines[key[0]] = float((a @ bh) / a.norm())
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(x["answer"] - r[0]), abs(x["foil"] - r[1])) for x, r in zip(native, ref))
    idx = [i for i, r in enumerate(rows) if r.construction in CONGRUENT]
    sub_rows, sub_native = [rows[i] for i in idx], [native[i] for i in idx]
    fw.directions = num
    arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="project"); forwards += n
    nat = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])
    fw.directions = orth
    arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="project"); forwards += n
    js = L.summarize(sub_rows, sub_native, [arm[i] for i in idx])
    nulls = []
    for seed in NULL_SEEDS:
        a2, n = v1._run_arm(fw, rows, components=v55.SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(sub_rows, sub_native, [a2[i] for i in idx]))
    null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    print("cosines", {k: round(v, 3) for k, v in cosines.items()}, "native-dir damage", round(nat["target_damage_mean"], 3), "has-had", round(nat["tense_has_had_abs_move_mean"], 3), "| orth damage", round(js["target_damage_mean"], 3), "fraction", round(js["target_damage_fraction"], 3), "pos", round(js["target_damage_positive_fraction"], 3), "has-had", round(js["tense_has_had_abs_move_mean"], 3), "null max", round(null_max, 3), "gates", gates)
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_orthogonalized_set_still_live": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
                   "pred_c_orthogonalized_set_spares_tense": gates["tense_has_had"], "pred_d_other_readers_still_within_gate": gates["animacy_who_which"] and gates["canonical_night_day"],
                   "pred_e_damage_retained_at_least_060": js["target_damage_mean"] >= RETAIN_MIN * nat["target_damage_mean"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "lexical_number_dod_orthogonal_natural_result_v141", "candidate_id": CANDIDATE_ID, "plan": plan, "instrument_max_abs_error": instrument, "cosines_were_was_vs_has_had": cosines,
                               "native_direction": nat, "orthogonalized": js, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
