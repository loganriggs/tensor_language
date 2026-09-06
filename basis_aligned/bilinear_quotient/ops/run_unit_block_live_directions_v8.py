#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, ranks, seeds, steps and bars fixed before the run.
"""v8: the standing DAS targets re-measured under the corrected BLOCK-LIVE semantics.

v7 (`unit_subspace_redteam_v7_result.json`) showed the cached cross-layer patch used by v2-v6
inflates a multi-layer set's direction effects by 1-41% at full rank. Every v5/v6 direction number
on a multi-layer set is therefore suspect: the aspectual 3-head set (layers 8, 9), the possessive
adjacent-fit set S (layers 3, 4, 9, 10) on its five siblings, and the pooled possessive set U
(layers 4, 5, 9, 15) whose rank-1 direction "did not serve the far designs" (fraction 0.58-0.61,
complement 0.42-0.45 -- possibly the inflation). This run re-measures all three, and registers the
possessive rank-2 test in advance (a null at rank 1 is not permission to raise the rank, so the
rank-2 prediction is written here BEFORE seeing the rank-1 block-live numbers).

  aspectual_anchor.has_vs_had    T = {08:01, 09:04, 09:01}; block dim fit on even A1 rows;
                                 held-out odd A1 rows, A2, P, C; cached-vs-live full-rank control;
                                 block DAS seed 0 rank 1 as the check
  possessive S                   {04:05, 03:04, 09:06, 10:05}; block dim fit on ADJACENT even rows;
                                 evaluated on the odd rows of adjacent, medial, long_simple,
                                 inanimate_argument, verb_final; cached bias per design
  possessive U                   {09:06, 05:03, 04:05, 15:01}; block dim fit on the POOLED even rows
                                 of the five designs; block DAS rank 1 and rank 2 per block (seed 0,
                                 120 steps, lr 0.05, exact-set objective, no complement term);
                                 evaluated per design on odd rows; cached bias per design

  Every direction reports subspace fraction, complement fraction, linearity sum S + C, random
  (rank-matched, seed 1). Bars: band [0.50, 1.20] of the exact set, complement <= 0.30,
  S + C in [0.85, 1.15], P <= 0.20, C <= 0.35, |cached bias| <= 0.10.

REGISTERED BEFORE THE RUN
    pred_a_aspectual_bias_small          aspectual T: |cached full-rank - exact| / exact <= 0.10
    pred_b_aspectual_block_dim_selective aspectual T block dim: held-out and A2 in band, complement
                                         <= 0.30, S + C in [0.85, 1.15], P <= 0.20, C <= 0.35
    pred_c_S_direction_travels           S block dim (adjacent fit): fraction >= 0.50 and complement
                                         <= 0.30 on the odd rows of ALL five designs
    pred_d_U_rank1_dim_serves_all        U pooled block dim: fraction >= 0.50 and complement <= 0.30
                                         on all five designs' odd rows
    pred_e_U_rank2_das_serves_all        U block DAS rank 2 per block: fraction in band and
                                         complement <= 0.30 on all five designs' odd rows
    pred_f_U_cached_bias_explains_v6     U: cached full-rank bias > 0.10 on at least one of the two
                                         far designs (inanimate_argument, verb_final)

    Priors. a expected (two adjacent layers; v7 saw 0.7-2.6% on 2-3 layer head sets). b expected
    (v5 cached: 0.985 / 0.816, complement 0.01 / 0.16). c UNSURE: v5 cached gave 0.66 / 0.72 with
    complement 0.36 / 0.31 on the far designs; if that was inflation, block-live passes. d UNSURE,
    same reasoning for v6. e: if d fails, rank 2 is the registered next step and I expect it to
    pass on fraction but not necessarily on complement; if d passes, e is redundant and reported.
    f expected (4 layers, one of them layer 15, the widest span in the corpus).
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
import circuit_fast_screen_candidate_possessive_argument as m_arg
import circuit_fast_screen_candidate_possessive_verbfinal as m_vf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_block_live_directions_v8_result.json"
T_ASP = ["attn:08:head:01", "attn:09:head:04", "attn:09:head:01"]
S_POSS = ["attn:04:head:05", "attn:03:head:04", "attn:09:head:06", "attn:10:head:05"]
U_POSS = ["attn:09:head:06", "attn:05:head:03", "attn:04:head:05", "attn:15:head:01"]
DESIGNS = {"adjacent_antecedent": m_adj, "medial_antecedent": m_med,
           "long_simple_intervener": m_long, "inanimate_argument": m_arg,
           "verb_final_distance_six": m_vf}
FAR = ("inanimate_argument", "verb_final_distance_six")
STEPS, LR = 120, 0.05
LO, HI, COMP_BAR, LIN_LO, LIN_HI, P_BAR, C_BAR, BIAS_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.20, 0.35, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3000, 250000


def _plan():
    return {"candidate_id": "corpus.unit_block_live_directions_v8",
            "sets": {"aspectual": T_ASP, "possessive_S": S_POSS, "possessive_U": U_POSS},
            "designs": list(DESIGNS), "steps": STEPS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 3 * STEPS, "model_updates": 0, "fit_parameters": 2 * 512,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _frac(e, v):
    return (v / e) if abs(e) > 1e-6 else None


def _battery(backend, prep, units, q, q_rand):
    b = g.direction_battery(backend, prep, units, q, q_rand=q_rand)
    s, c = b["subspace_fraction"], b["complement_fraction"]
    b["linearity_sum"] = None if s is None or c is None else s + c
    return b


def _semantics(backend, prep, units):
    torch = backend.torch
    exact = g.recovery(prep, g.patched_axis(backend, prep, units))
    eye = torch.eye(sum(g.unit_dim(u) for u in units), device=backend.device)
    cached = g.recovery(prep, g.patched_axis(backend, prep, units, q=eye))
    block = g.recovery(prep, g.patched_axis(backend, prep, units, q=g.block_identity(backend, units)))
    return {"exact": exact, "cached_full_rank": cached, "block_full_rank": block,
            "cached_bias_fraction": _frac(exact, cached - exact), "block_error": abs(block - exact)}


def _serves(b):
    return b["subspace_fraction"] is not None and b["subspace_fraction"] >= LO \
        and abs(b["complement_fraction"]) <= COMP_BAR


def _in_band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()

    # aspectual
    a1 = g.rows_of(m_asp, "A1")
    fit, held = g.prepare(backend, a1[0::2]), g.prepare(backend, a1[1::2])
    a2 = g.prepare(backend, g.rows_of(m_asp, "A2"))
    scale = g.target_scale(fit)
    q_dim = g.block_diff_in_means(backend, fit, T_ASP)
    q_rand = g.block_random_subspace(backend, T_ASP, rank=1, seed=1)
    q_das, h_das = g.fit_block_subspace(backend, fit, T_ASP, rank=1, steps=STEPS, lr=LR, seed=0)
    asp = {"units": T_ASP, "semantics_heldout": _semantics(backend, held, T_ASP), "directions": {}}
    for name, q in (("dim", q_dim), ("das", q_das)):
        pc = g.pc_effects(backend, m_asp, T_ASP, scale, q=q)
        asp["directions"][name] = {"held": _battery(backend, held, T_ASP, q, q_rand),
                                   "a2": _battery(backend, a2, T_ASP, q, q_rand),
                                   "p_effect": pc["P"], "c_effect": pc["C"]}
    asp["directions"]["das"]["loss_history"] = h_das
    asp["directions"]["das"]["cosine_to_dim"] = g.block_cosines(q_das, q_dim)
    asp["exact_p_c"] = g.pc_effects(backend, m_asp, T_ASP, scale)
    print("aspectual", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                   for k, v in asp["semantics_heldout"].items()}),
          json.dumps({n: {"held": round(d["held"]["subspace_fraction"], 3),
                          "comp": round(d["held"]["complement_fraction"], 3),
                          "sum": round(d["held"]["linearity_sum"], 3),
                          "a2": round(d["a2"]["subspace_fraction"], 3),
                          "P": round(d["p_effect"], 3), "C": round(d["c_effect"], 3)}
                      for n, d in asp["directions"].items()}))

    # possessive: preps per design (odd rows = evaluation), even rows pooled for U
    evals = {name: g.prepare(backend, g.rows_of(mod, "A1")[1::2]) for name, mod in DESIGNS.items()}
    adj_fit = g.prepare(backend, g.rows_of(m_adj, "A1")[0::2])
    pooled_rows = [r for mod in DESIGNS.values() for r in g.rows_of(mod, "A1")[0::2]]
    pooled_fit = g.prepare(backend, pooled_rows)

    qS = g.block_diff_in_means(backend, adj_fit, S_POSS)
    qS_rand = g.block_random_subspace(backend, S_POSS, rank=1, seed=1)
    S_rep = {"units": S_POSS, "fit": "adjacent even rows", "per_design": {}}
    for name, prep in evals.items():
        S_rep["per_design"][name] = {"semantics": _semantics(backend, prep, S_POSS),
                                     "dim": _battery(backend, prep, S_POSS, qS, qS_rand)}

    qU = g.block_diff_in_means(backend, pooled_fit, U_POSS)
    qU_rand1 = g.block_random_subspace(backend, U_POSS, rank=1, seed=1)
    qU_rand2 = g.block_random_subspace(backend, U_POSS, rank=2, seed=1)
    qU_das1, hU1 = g.fit_block_subspace(backend, pooled_fit, U_POSS, rank=1, steps=STEPS, lr=LR, seed=0)
    qU_das2, hU2 = g.fit_block_subspace(backend, pooled_fit, U_POSS, rank=2, steps=STEPS, lr=LR, seed=0)
    U_rep = {"units": U_POSS, "fit": "pooled even rows of five designs", "fit_rows": len(pooled_rows),
             "loss_history": {"das_rank1": hU1, "das_rank2": hU2},
             "cosine_das1_to_dim": g.block_cosines(qU_das1, qU), "per_design": {}}
    for name, prep in evals.items():
        U_rep["per_design"][name] = {
            "semantics": _semantics(backend, prep, U_POSS),
            "dim": _battery(backend, prep, U_POSS, qU, qU_rand1),
            "das_rank1": _battery(backend, prep, U_POSS, qU_das1, qU_rand1),
            "das_rank2": _battery(backend, prep, U_POSS, qU_das2, qU_rand2)}
    adj_a2 = g.prepare(backend, g.rows_of(m_adj, "A2"))
    scale_adj = g.target_scale(adj_fit)
    U_rep["adjacent_a2"] = {n: _battery(backend, adj_a2, U_POSS, q, r) for n, q, r in
                            (("dim", qU, qU_rand1), ("das_rank2", qU_das2, qU_rand2))}
    U_rep["adjacent_p_c"] = {n: g.pc_effects(backend, m_adj, U_POSS, scale_adj, q=q) for n, q in
                             (("exact", None), ("dim", qU), ("das_rank2", qU_das2))}

    for label, rep in (("S", S_rep), ("U", U_rep)):
        for name, d in rep["per_design"].items():
            print(label, name, "exact", round(d["semantics"]["exact"], 3), "bias",
                  round(d["semantics"]["cached_bias_fraction"], 3),
                  {n: (round(d[n]["subspace_fraction"], 3), round(d[n]["complement_fraction"], 3),
                       round(d[n]["linearity_sum"], 3)) for n in d if n != "semantics"})

    ad = asp["directions"]["dim"]
    predictions = {
        'pred_a_aspectual_bias_small': abs(asp["semantics_heldout"]["cached_bias_fraction"]) <= BIAS_BAR,
        'pred_b_aspectual_block_dim_selective': (
            _in_band(ad["held"]) and _in_band(ad["a2"])
            and LIN_LO <= ad["held"]["linearity_sum"] <= LIN_HI
            and ad["p_effect"] <= P_BAR and ad["c_effect"] <= C_BAR),
        'pred_c_S_direction_travels': all(_serves(d["dim"]) for d in S_rep["per_design"].values()),
        'pred_d_U_rank1_dim_serves_all': all(_serves(d["dim"]) for d in U_rep["per_design"].values()),
        'pred_e_U_rank2_das_serves_all': all(_in_band(d["das_rank2"]) for d in U_rep["per_design"].values()),
        'pred_f_U_cached_bias_explains_v6': any(
            abs(U_rep["per_design"][n]["semantics"]["cached_bias_fraction"]) > BIAS_BAR for n in FAR),
    }
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_block_live_directions_result_v8",
              "candidate_id": "corpus.unit_block_live_directions_v8",
              "registered": {"steps": STEPS, "lr": LR, "band": [LO, HI], "complement_bar": COMP_BAR,
                             "linear_band": [LIN_LO, LIN_HI], "p_bar": P_BAR, "c_bar": C_BAR,
                             "bias_bar": BIAS_BAR, "semantics": "block-live",
                             "das_objective": "match_exact_set_patch"},
              "predictions": predictions, "aspectual": asp, "possessive_S": S_rep,
              "possessive_U": U_rep, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
