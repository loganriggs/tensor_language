#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_ten_units_move_the_margin pred_c_beats_random_ten_sets pred_d_direction_of_effect pred_e_named_units_3465_493_comparison
"""The second MLP-3 port, edited (v327). v326: ten MLP-3 units (664, 872, 1612, 615, 919, 2570, 1090, 190, 1250, 1335) carry 71% of the harm that
un-conditioning MLP 1 does to the they - he margin through MLP 3; they are not the named number units 3465 / 493. Edits decide on NATIVE rows: zero the
ten at every position on the v76 rows and read the oriented margin, against 12 seeded random 10-unit sets of MLP 3; for comparison, zero {3465, 493}.
PREDICTIONS (scored as written; failures preserved; priors unsure -- the ten were named by their response to a perturbation, not by native carriage)
    pred_a_baseline_replays              the unedited margin replays 2.048 within 1e-3
    pred_b_ten_units_move_the_margin     |margin change| when the ten are zeroed >= 0.05 of the native margin
    pred_c_beats_random_ten_sets         that |change| exceeds 3x the largest among the 12 random 10-sets
    pred_d_direction_of_effect           the change is negative (the ten support the correct number). Prior: unsure.
    pred_e_named_units_3465_493_comparison  |change| for the ten exceeds |change| for {3465, 493}. Prior: unsure.
PRICE (registered maximum): 3 row batches x (1 baseline + 1 ten + 1 pair + 12 null) = 45 forwards; 0 backwards; 0 fits. Bar <= 48.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, random, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp3_second_port_edit_v327_result.json"
CANDIDATE_ID = "mlp3.second_port_edit_v327"
UNITS, PAIR, LAYER, N_NULL, SEED = (664, 872, 1612, 615, 919, 2570, 1090, 190, 1250, 1335), (3465, 493), 3, 12, 327
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
NATIVE_M, M_TOL, MOVE_MIN, NULL_FACTOR = 2.0481, 1e-3, 0.05, 3.0
FORWARDS_MAX = 48
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_ten_units_move_the_margin": ">= 0.05 of native", "pred_c_beats_random_ten_sets": "> 3x null", "pred_d_direction_of_effect": "negative", "pred_e_named_units_3465_493_comparison": "ten > pair"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in UNITS and j not in PAIR]; null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    conditions = [("baseline", None), ("edit", UNITS)] + [(f"null{k}", s) for k, s in enumerate(null_sets)]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "pair": list(PAIR), "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "move_min": MOVE_MIN, "null_factor": NULL_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; fw = L.ManualForward(backend); forwards = 0
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("ten", UNITS), ("pair", PAIR)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, units in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; out += dod_units.forward_margins(backend, fw, chunk, LAYER, None if units is None else (units, lambda row: None), readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum((margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = oriented("baseline"); d_ten = oriented("ten") - base; d_pair = oriented("pair") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_ten": d_ten, "change_pair_3465_493": d_pair, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "replay_gap": abs(base - NATIVE_M)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_ten_units_move_the_margin": abs(d_ten) >= MOVE_MIN * abs(base), "pred_c_beats_random_ten_sets": abs(d_ten) > NULL_FACTOR * report["null_max_abs"], "pred_d_direction_of_effect": d_ten < 0, "pred_e_named_units_3465_493_comparison": abs(d_ten) > abs(d_pair)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_second_port_edit_result_v327", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
