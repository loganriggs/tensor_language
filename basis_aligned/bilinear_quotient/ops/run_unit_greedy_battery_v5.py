#!/usr/bin/env python3
# BQGATE: frozen predictions; head set, pool, target, gain floor, bars fixed before the run.
"""v5: does the possessive head set travel across its five matched siblings, and which heads carry
aspectual has_vs_had?

Method (user direction 2026-09-06, validated in v2-v4): whole-unit interchange -> greedy smallest
head set -> rank-1 direction; the direction is DIFF-IN-MEANS (v4: it matched the exact-set margin
with an inert complement on every set, where DAS was non-unique), reported with its complement
and a random baseline.

  PART 1  possessive_number: the v3 set S = {04:05, 03:04, 09:06, 10:05} (adjacent_antecedent,
          joint 0.567). For each of the five siblings: exact interchange of S on A1 (fraction of
          the sibling's donor margin); diff-in-means direction fit on adjacent even rows,
          patched into the sibling (fraction of S's exact effect there), and its complement;
          greedy re-selection over the sibling's own top-12 heads, to see whether S is what the
          sibling picks.
  PART 2  aspectual_anchor.has_vs_had: 162-head sweep, greedy set (pool 12, target 0.50, gain
          0.02, at most 6), A2 transfer, exact-set P/C, diff-in-means rank-1 on even/odd split
          with complement and random.

REGISTERED BEFORE THE RUN
    pred_a_possessive_set_transfers        exact S reaches >= 0.35 on all four passing siblings
                                            (0.35 = the v3 joint minus the A2 drop seen in v4)
    pred_b_possessive_direction_transfers  diff-in-means fit on adjacent gives >= 0.50 of S's
                                            exact effect on all four passing siblings, complement
                                            <= 0.30 on each
    pred_c_siblings_repick_the_core        every passing sibling's greedy set shares >= 2 heads with S
    pred_d_aspectual_reaches_bar           aspectual greedy joint >= 0.50 with <= 6 heads
    pred_e_aspectual_selective             exact-set A2 >= 0.50, P <= 0.20, C <= 0.35
    pred_f_aspectual_direction_in_band     diff-in-means held-out and A2 fraction in [0.50, 1.20],
                                            complement <= 0.30, random <= 0.10
    pred_g_attractor_degrades              animate_attractor (a terminal NULL: donor-side capability
                                            failure; rows whose donor does not beat the base are
                                            dropped and counted) gives exact S < 0.35 on the rows
                                            that remain. Registered as the negative transfer case.
    Siblings counted for a-c are the FOUR passing ones (medial, long_simple, argument, verb_final);
    the first enqueue crashed on the attractor's invalid rows (kernel refuses a non-positive donor
    denominator) -- this version drops them instead.
    Priors: a, b plausible (v4 A2 transfer 1.10 for the direction); c unsure -- the attractor and
    verb-final designs put the antecedent at a different distance and may recruit different
    heads; d unsure (Codex's four-head path recovered 0.05, but that was a fixed path, not a
    greedy set); e, f conditional on d.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_aspectual as m_asp
import circuit_fast_screen_candidate_possessive_adjacent as m_adj
import circuit_fast_screen_candidate_possessive_medial as m_med
import circuit_fast_screen_candidate_possessive_long_simple as m_long
import circuit_fast_screen_candidate_possessive_attractor as m_attr
import circuit_fast_screen_candidate_possessive_argument as m_arg
import circuit_fast_screen_candidate_possessive_verbfinal as m_vf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_greedy_battery_v5_result.json"
S = ["attn:04:head:05", "attn:03:head:04", "attn:09:head:06", "attn:10:head:05"]
SIBLINGS = {"medial_antecedent": m_med, "long_simple_intervener": m_long,
            "inanimate_argument": m_arg, "verb_final_distance_six": m_vf,
            "animate_attractor": m_attr}
PASSING = ["medial_antecedent", "long_simple_intervener", "inanimate_argument", "verb_final_distance_six"]
POOL, TARGET, MIN_GAIN, MAX_UNITS = 12, 0.50, 0.02, 6
SET_BAR, LO, HI, COMP_BAR, RANDOM_MAX, P_BAR, C_BAR, SHARE = 0.35, 0.50, 1.20, 0.30, 0.10, 0.20, 0.35, 2
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 120000


def _plan():
    return {"candidate_id": "corpus.unit_greedy_battery_v5", "possessive_set": S,
            "siblings": list(SIBLINGS), "aspectual": "aspectual_anchor.has_vs_had",
            "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _frac(exact, value):
    return (value / exact) if abs(exact) > 1e-6 else None


def _direction_block(backend, prep, units, q_dim, q_rand):
    exact = g.recovery(prep, g.patched_axis(backend, prep, units))
    sub = g.recovery(prep, g.patched_axis(backend, prep, units, q=q_dim))
    comp = g.recovery(prep, g.patched_axis(backend, prep, units, q=q_dim, complement=True))
    rand = g.recovery(prep, g.patched_axis(backend, prep, units, q=q_rand))
    return {"exact_set": exact, "dim": sub, "dim_fraction": _frac(exact, sub),
            "complement": comp, "complement_fraction": _frac(exact, comp),
            "random": rand, "random_fraction": _frac(exact, rand)}


def _greedy(backend, prep, heads):
    singles = g.unit_sweep(backend, prep, heads)
    ranked = sorted(singles, key=singles.get, reverse=True)
    greedy = g.greedy_select(lambda s: g.recovery(prep, g.patched_axis(backend, prep, s)),
                             ranked[:POOL], target=TARGET, min_gain=MIN_GAIN, max_units=MAX_UNITS)
    return singles, ranked, greedy


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    heads = g.all_head_units()

    # Part 1: possessive siblings
    adj = g.rows_of(m_adj, "A1")
    adj_fit = g.prepare(backend, adj[0::2])
    q_dim = g.diff_in_means_direction(backend, adj_fit, S)
    q_rand = g.random_subspace(backend, S, rank=1)
    poss = {"adjacent_heldout": _direction_block(backend, g.prepare(backend, adj[1::2]), S, q_dim, q_rand)}
    for name, module in SIBLINGS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        block = _direction_block(backend, prep, S, q_dim, q_rand)
        block["rows_dropped_invalid_donor"] = prep.dropped
        block["rows_used"] = len(prep.rows)
        singles, ranked, greedy = _greedy(backend, prep, heads)
        block.update({"greedy": greedy, "top_heads": {u: singles[u] for u in ranked[:POOL]},
                      "shared_with_S": sorted(set(greedy["chosen"]) & set(S)),
                      "single_effects_of_S": {u: singles[u] for u in S}})
        poss[name] = block
        print(name, json.dumps({"exact_S": round(block["exact_set"], 3),
                                "dim_frac": block["dim_fraction"], "comp_frac": block["complement_fraction"],
                                "chosen": greedy["chosen"], "joint": round(greedy["joint"], 3)}))

    # Part 2: aspectual
    a1 = g.rows_of(m_asp, "A1")
    prep = g.prepare(backend, a1)
    scale = g.target_scale(prep)
    singles, ranked, greedy = _greedy(backend, prep, heads)
    chosen = greedy["chosen"]
    a2_prep = g.prepare(backend, g.rows_of(m_asp, "A2"))
    same = {}
    for fam in ("P", "C"):
        fp = g.prepare(backend, g.rows_of(m_asp, fam))
        same[fam] = g.same_answer_effect(fp, g.patched_axis(backend, fp, chosen), scale)
    fit, held = g.prepare(backend, a1[0::2]), g.prepare(backend, a1[1::2])
    qd = g.diff_in_means_direction(backend, fit, chosen)
    qr = g.random_subspace(backend, chosen, rank=1)
    modules = g.module_sweep(backend, prep)
    asp = {"module_sweep_top": dict(sorted(modules.items(), key=lambda kv: -kv[1])[:8]),
           "top_heads": {u: singles[u] for u in ranked[:POOL]}, "greedy": greedy, "chosen": chosen,
           "joint": greedy["joint"], "sum_of_singles": sum(singles[u] for u in chosen),
           "a2_exact_set": g.recovery(a2_prep, g.patched_axis(backend, a2_prep, chosen)),
           "p_effect_exact_set": same["P"], "c_effect_exact_set": same["C"],
           "direction": {"a1_heldout": _direction_block(backend, held, chosen, qd, qr),
                         "a2": _direction_block(backend, a2_prep, chosen, qd, qr)}}
    print("aspectual", json.dumps({"chosen": chosen, "joint": round(greedy["joint"], 3),
                                   "a2": round(asp["a2_exact_set"], 3), "P": round(same["P"], 3),
                                   "C": round(same["C"], 3),
                                   "dim_held": asp["direction"]["a1_heldout"]["dim_fraction"],
                                   "dim_a2": asp["direction"]["a2"]["dim_fraction"]}))

    f = lambda x: x if x is not None else 0.0
    sib = [poss[n] for n in PASSING]
    dir_ok = [f(b["dim_fraction"]) >= LO and abs(f(b["complement_fraction"])) <= COMP_BAR for b in sib]
    d = asp["direction"]
    predictions = {
        "pred_a_possessive_set_transfers": all(b["exact_set"] >= SET_BAR for b in sib),
        "pred_b_possessive_direction_transfers": all(dir_ok),
        "pred_c_siblings_repick_the_core": all(len(b["shared_with_S"]) >= SHARE for b in sib),
        "pred_d_aspectual_reaches_bar": greedy["reached_target"],
        "pred_e_aspectual_selective": asp["a2_exact_set"] >= LO and same["P"] <= P_BAR and same["C"] <= C_BAR,
        "pred_f_aspectual_direction_in_band": all(
            LO <= f(d[k]["dim_fraction"]) <= HI and abs(f(d[k]["complement_fraction"])) <= COMP_BAR
            and abs(f(d[k]["random_fraction"])) <= RANDOM_MAX for k in ("a1_heldout", "a2")),
        "pred_g_attractor_degrades": poss["animate_attractor"]["exact_set"] < SET_BAR,
    }
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_greedy_battery_result_v5",
              "candidate_id": "corpus.unit_greedy_battery_v5",
              "registered": {"possessive_set": S, "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN,
                             "max_units": MAX_UNITS, "set_bar": SET_BAR, "band": [LO, HI],
                             "complement_bar": COMP_BAR, "random_max": RANDOM_MAX, "p_bar": P_BAR,
                             "c_bar": C_BAR, "shared_heads": SHARE, "direction": "diff_in_means"},
              "predictions": predictions, "possessive": poss, "aspectual": asp,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
