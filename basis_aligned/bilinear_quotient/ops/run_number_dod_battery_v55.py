#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective_with_number_readers pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_head_11_3_leads
"""Number family DoD battery, step 1 (v55): the sweep-led set {11.3, 5.7, 7.8, 9.7} on FRESH lexical-number rows.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v50b–v53 (number decisions share {11.3, 5.7, 7.8,
9.7}; the was−were reader is related there).

Rows: the lexical-number line's two constructions ("The A(s) near the P" -> were/was; "In the report the A(s) beside the
P") on subjects whose singular AND plural are single tokens (checked at build) and places new to the line. Readers:
has−had (tense; unrelated to number), who−which, night−day — was−were is the target axis and is not used as a control.
Contrast `O_h^T(u_were − u_was)`.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_native_capability           all four (construction x number) cells >= 0.85
    pred_c_set_live_and_beats_null     fraction >= 0.10, positive >= 0.75, > max of 16 norm-matched nulls
    pred_d_set_selective_with_number_readers   has−had, who−which, night−day within null + 0.25 x damage. Prior: unsure —
                                       11.3 also serves tense, so has−had may move.
    pred_e_set_is_additive             |joint − Σ singles| <= 0.25 x min single
    pred_f_keep_only_retains_most      keep-only retention >= 0.70; random keeps <= 0.30
    pred_g_head_11_3_leads             11.3 is the largest single

PRICE (registered maximum): 64 rows in 2 batches: native 2 + producer 2 + set 2 + nulls 32 + singles 8 + zero 2 + keep 2 +
random keeps 32 = 82 forwards; 0 backwards; 0 fits. Bar <= 90.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import circuit_fast_screen_candidates as lex
import run_aspectual_dod_removal_v1 as v1

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_battery_v55_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_battery_v55"
L.READERS = {"target_was_were": (" was", " were"), "tense_has_had": (" has", " had"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_has_had", "animacy_who_which", "canonical_night_day")
WERE, WAS = L._single(" were"), L._single(" was")
NULL_SEEDS, KEEP_SEEDS = tuple(range(1601, 1617)), tuple(range(1651, 1667))
CAPABILITY_MIN, LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, ADD_RATIO, RETAIN_MIN, RANDOM_MAX, INSTRUMENT_TOL = 0.85, 0.10, 0.75, 0.25, 0.25, 0.70, 0.30, 1e-4
FORWARDS_MAX = 90
CANDIDATE_SUBJECTS = ("clown", "spy", "shepherd", "brewer", "nun", "cook", "waiter", "tutor", "pastor", "referee", "landlord", "dealer", "printer", "hunter", "knight", "wizard", "monk", "poet", "banker", "miner", "scout", "priest", "sheriff", "dancer", "singer", "soldier", "surgeon", "merchant", "butcher", "tailor", "mayor", "bishop")
PLACES = ("palace", "castle", "temple", "prison", "hospital", "library", "museum", "station", "factory", "school", "church", "cottage", "quarry", "pier", "fountain", "tunnel")
SINGLES = (L.Component("attn11_h3_final", 11, "attn", (3,), "final"), L.Component("attn5_h7_final", 5, "attn", (7,), "final"),
           L.Component("attn7_h8_final", 7, "attn", (8,), "final"), L.Component("attn9_h7_final", 9, "attn", (7,), "final"))
SET = SINGLES


def build():
    used_pp = set(lex._OBJECTS)
    if set(PLACES) & used_pp:
        raise SystemExit("places overlap the corpus object list")
    subjects = []
    for w in CANDIDATE_SUBJECTS:
        if len(L.ENCODING.encode(" " + w)) == 1 and len(L.ENCODING.encode(" " + w + "s")) == 1:
            subjects.append(w)
        if len(subjects) == 16:
            break
    if len(subjects) < 16:
        raise SystemExit(f"only {len(subjects)} subjects with single-token singular and plural")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction in ("bare", "report"):
        for g in range(16):
            subj, place = subjects[g], PLACES[g]
            for plural in (True, False):
                noun = subj + ("s" if plural else "")
                text = f"The {noun} near the {place}" if construction == "bare" else f"In the report the {noun} beside the {place}"
                ids = L.ENCODING.encode(text); answer, foil = (" were", " was") if plural else (" was", " were")
                a_id, f_id = L._single(answer), L._single(foil)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise SystemExit(f"joint tokenization failed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["number_v55", construction, g, plural, text]).encode()).hexdigest()[:24], construction, g, plural, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, subjects


def main() -> None:
    rows, subjects = build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "subjects": subjects, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, SINGLES, WERE, WAS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    capability = {}
    for c in ("bare", "report"):
        for plural in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i, r in enumerate(rows) if r.construction == c and r.present == plural]
            capability[f"{c}/{'plural' if plural else 'singular'}"] = sum(cell) / len(cell)
    joint, n = v1._run_arm(fw, rows, components=SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    singles = {}
    for comp in SINGLES:
        arm, n = v1._run_arm(fw, rows, components=(comp,), mode="project"); forwards += n
        singles[comp.name] = L.summarize(rows, native, arm)["target_damage_mean"]
    gap = abs(js["target_damage_mean"] - sum(singles.values())); bar = ADD_RATIO * min(singles.values())
    zero, n = v1._run_arm(fw, rows, components=SET, mode="zero"); forwards += n
    zd = L.summarize(rows, native, zero)["target_damage_mean"]
    keep, n = v1._run_arm(fw, rows, components=SET, mode="keep_only"); forwards += n
    retention = 1.0 - L.summarize(rows, native, keep)["target_damage_mean"] / zd
    random_ret = []
    for seed in KEEP_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=SET, mode="keep_only_random", seed=seed); forwards += n
        random_ret.append(1.0 - L.summarize(rows, native, arm)["target_damage_mean"] / zd)
    print("capability", capability, "joint", round(js["target_damage_mean"], 3), "fraction", round(js["target_damage_fraction"], 3), "pos", js["target_damage_positive_fraction"], "null_max", round(null_max, 4), "gates", gates,
          "unrelated", {k: round(js[f"{k}_abs_move_mean"], 3) for k in L.UNRELATED}, "null moves", {k: round(v, 3) for k, v in null_moves.items()})
    print("singles", {k: round(v, 3) for k, v in singles.items()}, "gap/bar", round(gap, 4), round(bar, 4), "keep", round(retention, 3), "random keep max", round(max(random_ret), 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_native_capability": all(v >= CAPABILITY_MIN for v in capability.values()),
                   "pred_c_set_live_and_beats_null": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
                   "pred_d_set_selective_with_number_readers": all(gates.values()), "pred_e_set_is_additive": gap <= bar,
                   "pred_f_keep_only_retains_most": retention >= RETAIN_MIN and all(r <= RANDOM_MAX for r in random_ret), "pred_g_head_11_3_leads": max(singles, key=singles.get) == "attn11_h3_final"}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "number_family_dod_battery_result_v55", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "subjects": subjects, "instrument_max_abs_error": instrument,
              "capability": capability, "joint": js, "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates, "singles": singles, "additivity": {"gap": gap, "bar": bar},
              "keep_only": {"zero_damage": zd, "retention": retention, "random_retention": random_ret}, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
