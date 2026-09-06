#!/usr/bin/env python3
# BQGATE: frozen predictions; pool, target, gain floor, size cap and bars fixed before the run.
"""v6: one possessive head set for all five passing designs, selected on the POOLED rows.

v5 (`unit_greedy_battery_v5_result.json`): the adjacent-fit set S = {04:05, 03:04, 09:06, 10:05}
travels to medial (0.455, direction 1.05, complement 0.00) and long_simple (0.436, 1.02, 0.02) but
only reaches 0.35 on inanimate_argument and verb_final, where the direction carries 0.66 / 0.72
and the complement 0.36 / 0.31 -- and EVERY sibling's own greedy picked attn:05:head:03 first, a
head S does not contain. S is adjacent-specific. The obvious repair is to select on the union.

  fit rows   even A1 rows of adjacent + medial + long_simple + inanimate_argument + verb_final
             (5 x 16 = 80 rows, one prep)
  select     162-head sweep on the pool, greedy over the top 12 (target 0.50, gain 0.02, <= 6)
  evaluate   the pooled set U, exact and through its pooled diff-in-means direction, on the ODD
             rows of each design; A2 of adjacent; P / C of adjacent through the exact set;
             animate_attractor (terminal null, donor-invalid rows dropped) as the negative case.

REGISTERED BEFORE THE RUN
    pred_a_pooled_set_reaches_bar        U joint >= 0.50 on the pooled fit rows with <= 6 heads
    pred_b_pooled_set_serves_all_five    U exact >= 0.35 on the held-out rows of every design
    pred_c_pooled_direction_serves_all   pooled diff-in-means: fraction of U's exact >= 0.50 and
                                          complement <= 0.30 on the held-out rows of every design
    pred_d_head_05_03_is_in_U            attn:05:head:03 is selected
    pred_e_pooled_set_selective          adjacent A2 >= 0.50, P <= 0.20, C <= 0.35 through U
    pred_f_attractor_degrades            U exact < 0.35 on the attractor's valid rows
    Priors: a, d expected; b, c plausible but the two far-antecedent designs may need heads the
    near ones do not, in which case U grows to 5-6 and the direction fraction drops there; e
    expected (v3/v5 sets were selective); f unsure -- the attractor is a donor-side capability
    failure, so U may well still move the margin on the rows that remain.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_possessive_adjacent as m_adj
import circuit_fast_screen_candidate_possessive_medial as m_med
import circuit_fast_screen_candidate_possessive_long_simple as m_long
import circuit_fast_screen_candidate_possessive_argument as m_arg
import circuit_fast_screen_candidate_possessive_verbfinal as m_vf
import circuit_fast_screen_candidate_possessive_attractor as m_attr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_greedy_pooled_possessive_v6_result.json"
DESIGNS = {"adjacent_antecedent": m_adj, "medial_antecedent": m_med,
           "long_simple_intervener": m_long, "inanimate_argument": m_arg,
           "verb_final_distance_six": m_vf}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 12, 0.50, 0.02, 6
SET_BAR, LO, COMP_BAR, P_BAR, C_BAR = 0.35, 0.50, 0.30, 0.20, 0.35
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 1000, 100000


def _plan():
    return {"candidate_id": "possessive_number.unit_greedy_pooled_v6", "designs": list(DESIGNS),
            "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    a1 = {k: g.rows_of(m, "A1") for k, m in DESIGNS.items()}
    pool_prep = g.prepare(backend, [r for rows in a1.values() for r in rows[0::2]])
    singles, ranked, greedy = g.greedy_heads(backend, pool_prep, pool=POOL, target=TARGET,
                                             min_gain=MIN_GAIN, max_units=MAX_UNITS)
    U = greedy["chosen"]
    q = g.diff_in_means_direction(backend, pool_prep, U)
    q_rand = g.random_subspace(backend, U, rank=1)
    per = {}
    for k in DESIGNS:
        held = g.prepare(backend, a1[k][1::2])
        per[k] = g.direction_battery(backend, held, U, q, q_rand)
        print(k, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in per[k].items()}))
    attr = g.prepare(backend, g.rows_of(m_attr, "A1"), valid_only=True)
    per["animate_attractor"] = dict(g.direction_battery(backend, attr, U, q, q_rand),
                                    rows_dropped_invalid_donor=attr.dropped, rows_used=len(attr.rows))
    scale = g.target_scale(pool_prep)
    a2 = g.prepare(backend, g.rows_of(m_adj, "A2"))
    sel = {"adjacent_a2_exact": g.recovery(a2, g.patched_axis(backend, a2, U)),
           **{f"adjacent_{k.lower()}_effect_exact": v for k, v in g.pc_effects(backend, m_adj, U, scale).items()},
           **{f"adjacent_{k.lower()}_effect_dim": v for k, v in g.pc_effects(backend, m_adj, U, scale, q=q).items()}}
    print("pooled", json.dumps({"U": U, "joint": round(greedy["joint"], 3), **{k: round(v, 3) for k, v in sel.items()}}))

    f = lambda x: x if x is not None else 0.0
    five = [per[k] for k in DESIGNS]
    predictions = {
        "pred_a_pooled_set_reaches_bar": greedy["reached_target"],
        "pred_b_pooled_set_serves_all_five": all(b["exact_set"] >= SET_BAR for b in five),
        "pred_c_pooled_direction_serves_all": all(
            f(b["subspace_fraction"]) >= LO and abs(f(b["complement_fraction"])) <= COMP_BAR for b in five),
        "pred_d_head_05_03_is_in_U": "attn:05:head:03" in U,
        "pred_e_pooled_set_selective": (sel["adjacent_a2_exact"] >= LO
                                        and sel["adjacent_p_effect_exact"] <= P_BAR
                                        and sel["adjacent_c_effect_exact"] <= C_BAR),
        "pred_f_attractor_degrades": per["animate_attractor"]["exact_set"] < SET_BAR,
    }
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_greedy_pooled_result_v6",
              "candidate_id": "possessive_number.unit_greedy_pooled_v6",
              "registered": {"pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
                             "set_bar": SET_BAR, "lo": LO, "complement_bar": COMP_BAR,
                             "p_bar": P_BAR, "c_bar": C_BAR, "direction": "diff_in_means_on_pool"},
              "predictions": predictions, "pooled": {"rows": len(pool_prep.rows), "greedy": greedy,
                                                     "top_heads": {u: singles[u] for u in ranked[:POOL]},
                                                     "chosen": U, "sum_of_singles": sum(singles[u] for u in U)},
              "per_design": per, "selectivity": sel,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
