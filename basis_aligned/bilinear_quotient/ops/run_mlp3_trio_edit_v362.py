#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_trio_lowers_the_margin pred_c_beats_random_trios pred_d_trio_exceeds_pair pred_e_114_adds_to_the_pair
"""Edit of the MLP-3 trio that writes head 4.5's copied state (v362). v361: on "The X" frames the state head 4.5 carries to the pronoun reader is written,
within MLP 3, by 3465 (+172k), 114 (+74k) and 493 (+32k). v327: zeroing {3465, 493} on the native v76 rows moved the they - he margin by -2.2%. Edits decide
on native rows: zero {3465, 114, 493} at every position; compare with {3465, 493} and with {114} alone; null: 12 seeded random 3-unit sets of MLP 3; readout
the oriented they - he margin.
PREDICTIONS (scored as written; failures preserved; priors from v327 / v361)
    pred_a_baseline_replays     the unedited margin replays 2.048 within 1e-3
    pred_b_trio_lowers_the_margin  zeroing the trio lowers the oriented margin by >= 0.03 of its native value
    pred_c_beats_random_trios   |trio change| exceeds 3x the largest |change| among the 12 random trios
    pred_d_trio_exceeds_pair    |trio change| > |pair {3465, 493} change|
    pred_e_114_adds_to_the_pair 114 alone lowers the margin (negative change). Prior: unsure -- 114 is also a verb-site mismatch unit.
PRICE (registered maximum): 3 row batches x (1 baseline + 3 edits + 12 null) = 48 forwards; 0 backwards; 0 fits. Bar <= 52.
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
OUT = ROOT / "circuits/followups/mlp3_trio_edit_v362_result.json"
CANDIDATE_ID = "mlp3.trio_edit_v362"
UNITS, PAIR, SOLO, LAYER, N_NULL, SEED = (3465, 114, 493), (3465, 493), (114,), 3, 12, 362
PHRASE = (",", " and", " of", " the", " very"); K, N, BATCH = 8, 32, 32
NATIVE_M, M_TOL, MOVE_MIN, NULL_FACTOR = 2.0481, 1e-3, 0.03, 3.0
FORWARDS_MAX = 52
PREDICTIONS = {"pred_a_baseline_replays": "2.048 +- 1e-3", "pred_b_trio_lowers_the_margin": "<= -0.03 of native", "pred_c_beats_random_trios": "> 3x null", "pred_d_trio_exceeds_pair": "|trio| > |pair|", "pred_e_114_adds_to_the_pair": "negative"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    rng = random.Random(SEED); pool = [j for j in range(4608) if j not in UNITS and j not in PAIR]; null_sets = [tuple(sorted(rng.sample(pool, len(UNITS)))) for _ in range(N_NULL)]
    conditions = [("baseline", None), ("edit", UNITS)] + [(f"null{k}", s) for k, s in enumerate(null_sets)]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": list(UNITS), "pair": list(PAIR), "solo": list(SOLO), "layer": LAYER, "n_null": N_NULL, "seed": SEED, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"native_m": NATIVE_M, "m_tol": M_TOL, "move_min": MOVE_MIN, "null_factor": NULL_FACTOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch; fw = L.ManualForward(backend); forwards = 0
    readers = {"they_he": (L._single(" they"), L._single(" he"))}; margins = {}
    conditions = [("baseline", None), ("trio", UNITS), ("pair", PAIR), ("solo114", SOLO)] + [(f"null{k}", s_) for k, s_ in enumerate(null_sets)]
    for name, units in conditions:
        out = []
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; out += dod_units.forward_margins(backend, fw, chunk, LAYER, None if units is None else (units, lambda row: None), readers); forwards += 1
        margins[name] = out
    def oriented(name): return sum((margins[name][i]["they_he"] if row.present else -margins[name][i]["they_he"]) for i, row in enumerate(rows)) / len(rows)
    base = oriented("baseline"); d_ten = oriented("trio") - base; d_pair = oriented("pair") - base; d_solo = oriented("solo114") - base; nulls = [oriented(f"null{k}") - base for k in range(N_NULL)]
    report = {"native_margin": base, "change_trio": d_ten, "change_pair_3465_493": d_pair, "change_114_alone": d_solo, "null_changes": nulls, "null_max_abs": max(abs(v) for v in nulls), "replay_gap": abs(base - NATIVE_M)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_baseline_replays": report["replay_gap"] <= M_TOL, "pred_b_trio_lowers_the_margin": d_ten <= -MOVE_MIN * abs(base), "pred_c_beats_random_trios": abs(d_ten) > NULL_FACTOR * report["null_max_abs"], "pred_d_trio_exceeds_pair": abs(d_ten) > abs(d_pair), "pred_e_114_adds_to_the_pair": d_solo < 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_second_port_edit_result_v327", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
