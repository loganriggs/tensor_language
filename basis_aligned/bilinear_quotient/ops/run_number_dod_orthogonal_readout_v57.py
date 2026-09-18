#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_orthogonalized_set_still_live pred_c_orthogonalized_set_spares_tense pred_d_other_readers_still_within_gate pred_e_damage_retained_at_least_060
"""Number family DoD, step 3 (v57): remove only the part of each head's were−was readout direction that is ORTHOGONAL
to its has−had direction (weights only: Gram–Schmidt of `O_h^T(u_were − u_was)` against `O_h^T(u_has − u_had)`; no fit).

Lane: Claude circuit lane. Parents: v55 (set removal moves has−had 0.88), v56 (number heads' two contrasts overlap at
cos −0.4..−0.6; temporal heads' do not). If the tense movement is that weight-level overlap, the orthogonalized removal
should keep most of the number damage while sparing has−had.

Rows: the v55 fresh lexical-number rows. Arms: native; producer; orthogonalized set removal; 16 norm-matched nulls.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4
    pred_b_orthogonalized_set_still_live   fraction >= 0.10, positive >= 0.75, > max null
    pred_c_orthogonalized_set_spares_tense has−had mean|move| <= null mean + 0.25 x damage
    pred_d_other_readers_still_within_gate who−which and night−day within the same gate
    pred_e_damage_retained_at_least_060    orthogonalized damage >= 0.60 x the v55 set damage (2.671)

PRICE (registered maximum): native 2 + producer 2 + set 2 + nulls 32 = 38 forwards; 0 backwards; 0 fits. Bar <= 44.
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
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_orthogonal_readout_v57_result.json"
V55 = ROOT / "circuits/followups/number_family_dod_battery_v55_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_orthogonal_readout_v57"
NULL_SEEDS = tuple(range(1701, 1717))
LIVE_FRACTION, LIVE_POSITIVE, GATE_RATIO, RETAIN_MIN, INSTRUMENT_TOL = 0.10, 0.75, 0.25, 0.60, 1e-4
FORWARDS_MAX = 44


def main() -> None:
    rows, _ = v55.build()
    v55_damage = json.loads(V55.read_text())["joint"]["target_damage_mean"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    num = L.readout_directions(backend.model, v55.SINGLES, v55.WERE, v55.WAS)
    tense = L.readout_directions(backend.model, v55.SINGLES, L._single(" has"), L._single(" had"))
    fw.directions = {}
    cosines = {}
    for key in num:
        a, b = num[key].float(), tense[key].float(); bh = b / b.norm()
        fw.directions[key] = a - (a @ bh) * bh
        cosines[key[0]] = float((a @ bh) / a.norm())
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(x["answer"] - r[0]), abs(x["foil"] - r[1])) for x, r in zip(native, ref))
    joint, n = v1._run_arm(fw, rows, components=v55.SET, mode="project"); forwards += n
    js = L.summarize(rows, native, joint)
    nulls = []
    for seed in NULL_SEEDS:
        arm, n = v1._run_arm(fw, rows, components=v55.SET, mode="project_random", seed=seed); forwards += n
        nulls.append(L.summarize(rows, native, arm))
    null_max = max(s["target_damage_mean"] for s in nulls); null_moves = {name: sum(s[f"{name}_abs_move_mean"] for s in nulls) / len(nulls) for name in L.UNRELATED}
    gates = {name: js[f"{name}_abs_move_mean"] <= null_moves[name] + GATE_RATIO * js["target_damage_mean"] for name in L.UNRELATED}
    print("cosines", {k: round(v, 3) for k, v in cosines.items()}, "damage", round(js["target_damage_mean"], 3), "fraction", round(js["target_damage_fraction"], 3), "pos", js["target_damage_positive_fraction"], "null_max", round(null_max, 4),
          "moves", {k: round(js[f"{k}_abs_move_mean"], 3) for k in L.UNRELATED}, "null moves", {k: round(v, 3) for k, v in null_moves.items()}, "gates", gates, "retained", round(js["target_damage_mean"] / v55_damage, 3))
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
                   "pred_b_orthogonalized_set_still_live": js["target_damage_fraction"] >= LIVE_FRACTION and js["target_damage_positive_fraction"] >= LIVE_POSITIVE and js["target_damage_mean"] > null_max,
                   "pred_c_orthogonalized_set_spares_tense": gates["tense_has_had"], "pred_d_other_readers_still_within_gate": gates["animacy_who_which"] and gates["canonical_night_day"],
                   "pred_e_damage_retained_at_least_060": js["target_damage_mean"] >= RETAIN_MIN * v55_damage}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "number_family_dod_orthogonal_readout_result_v57", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "cosines_were_was_vs_has_had": cosines, "joint": js,
              "null_damage_max": null_max, "null_unrelated_abs_move_mean": null_moves, "gates": gates, "v55_set_damage": v55_damage, "retained": js["target_damage_mean"] / v55_damage,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
