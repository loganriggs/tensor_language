#!/usr/bin/env python3
# BQGATE: frozen predictions; pool, target, gain floor, size cap and DAS rank fixed before the run.
"""v3: heads-only greedy sets for the two behaviours whose v2 sets contained an MLP; list direction
across constructions.

What v2 showed (`unit_greedy_protocol_v2_result.json`). With the corrected exact-set objective,
head-only sets behave: correlative (3 heads) rank-1 held-out 1.09 / A2 0.95, modal (2 heads)
0.98 / 0.99, P and C at bar-noise, random direction 0.00. The two sets that included an MLP output
unit (polarity: mlp:04; possessive: mlp:08) did not: held-out 3.9 and 2.0, and for polarity a
RANDOM rank-4 subspace moved the margin by -0.37 of the set effect. An 1152-d MLP output at the
semantic position is a high-gain unit whose projections steer; with 16 fitting rows the fit
finds those projections. The list direction (attn:08 heads 3+7) fit on A1 reproduced 0.96 on
held-out A1 and 0.00 on A2 -- construction-specific.

This run answers both, with the same library and the user's original framing (sets of HEADS):

  1. polarity_state and possessive_number: greedy over a pool of the top 12 HEADS only
     (MLP units excluded), then rank-1 joint DAS with the exact-set objective.
  2. numbered_list: the same two heads; rank-1 fit on A2 evaluated on A1, and rank-1 fit on
     the even rows of A1 AND A2 jointly, evaluated on the odd rows of each.

REGISTERED BEFORE THE RUN
    pool 12 heads, target 0.50, gain floor 0.02, at most 6 heads, rank 1, 200 Adam steps,
    fit on even rows / evaluate on odd rows, random baseline seed 1.

    pred_a_heads_only_sets_reach_the_bar     both behaviours: joint >= 0.50 with <= 6 heads
    pred_b_heads_only_rank1_in_band          both: held-out and A2 fraction of exact in
                                             [0.50, 1.20], P <= 0.20, C <= 0.35, random <= 0.10
    pred_c_list_direction_is_construction_specific
                                             fit on A2 -> held-out A2 >= 0.50 and A1 < 0.50
                                             (the mirror of the v2 finding)
    pred_d_one_direction_serves_both_list_constructions
                                             rank-1 fit on A1+A2 (even rows) reaches >= 0.50 on
                                             the odd rows of A1 AND of A2.
        Stated prior on c/d: unsure. If c holds and d fails, the two constructions use
        different directions of these heads. If both hold, one direction exists and each
        single-construction fit simply picked a construction-specific one.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_control_choice as m_list
import circuit_fast_screen_candidate_possessive_adjacent as m_poss
import circuit_fast_screen_candidate_polarity_state as m_pol

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_greedy_heads_only_v3_result.json"
POOL, TARGET, MIN_GAIN, MAX_UNITS, RANK, DAS_STEPS = 12, 0.50, 0.02, 6, 1, 200
P_BAR, C_BAR, LO, HI, RANDOM_MAX = 0.20, 0.35, 0.50, 1.20, 0.10
LIST_HEADS = ["attn:08:head:03", "attn:08:head:07"]
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 5000, 200000


def _plan():
    return {"candidate_id": "corpus.unit_greedy_heads_only_v3",
            "behaviours": ["polarity_state.negative_vs_positive",
                           "possessive_number.adjacent_antecedent",
                           "numbered_list.control_choice_discriminator"],
            "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
            "rank": RANK, "das_steps": DAS_STEPS, "units": "heads only",
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 5 * DAS_STEPS, "model_updates": 0, "fit_parameters": 768,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _ev(backend, prep, units, q):
    exact = g.recovery(prep, g.patched_axis(backend, prep, units))
    sub = g.recovery(prep, g.patched_axis(backend, prep, units, q=q))
    return {"exact_set": exact, "subspace": sub,
            "fraction_of_exact": (sub / exact) if abs(exact) > 1e-6 else None}


def _fam_effects(backend, module, units, q, scale):
    out = {}
    for fam in ("P", "C"):
        fp = g.prepare(backend, g.rows_of(module, fam))
        out[fam] = g.same_answer_effect(fp, g.patched_axis(backend, fp, units, q=q), scale)
    return out


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    instrument = g.verify_against_producer(backend, g.rows_of(m_pol, "A1"), layer=7,
                                           heads=(8,), mlp_layer=4)
    print("instrument:", json.dumps(instrument))
    if not instrument["passed"]:
        raise SystemExit("new forward does not reproduce the producer")

    heads = g.all_head_units()
    report = {}
    for label, module in (("polarity_state.negative_vs_positive", m_pol),
                          ("possessive_number.adjacent_antecedent", m_poss)):
        a1 = g.rows_of(module, "A1")
        prep = g.prepare(backend, a1)
        scale = g.target_scale(prep)
        singles = g.unit_sweep(backend, prep, heads)
        ranked = sorted(singles, key=singles.get, reverse=True)
        greedy = g.greedy_select(lambda s: g.recovery(prep, g.patched_axis(backend, prep, s)),
                                 ranked[:POOL], target=TARGET, min_gain=MIN_GAIN,
                                 max_units=MAX_UNITS)
        chosen = greedy["chosen"]
        a2_prep = g.prepare(backend, g.rows_of(module, "A2"))
        fit_prep, held_prep = g.prepare(backend, a1[0::2]), g.prepare(backend, a1[1::2])
        q, hist = g.fit_joint_subspace(backend, fit_prep, chosen, rank=RANK, steps=DAS_STEPS)
        rq = g.random_subspace(backend, chosen, rank=RANK)
        same = _fam_effects(backend, module, chosen, q, scale)
        exact_same = _fam_effects(backend, module, chosen, None, scale)
        report[label] = {
            "top_heads": {u: singles[u] for u in ranked[:12]}, "greedy": greedy,
            "chosen": chosen, "joint": greedy["joint"],
            "sum_of_singles": sum(singles[u] for u in chosen),
            "a2_exact_set": g.recovery(a2_prep, g.patched_axis(backend, a2_prep, chosen)),
            "p_effect_exact_set": exact_same["P"], "c_effect_exact_set": exact_same["C"],
            "das_rank1": {"loss_history": hist,
                          "a1_fit": _ev(backend, fit_prep, chosen, q),
                          "a1_heldout": _ev(backend, held_prep, chosen, q),
                          "a2": _ev(backend, a2_prep, chosen, q),
                          "p_effect": same["P"], "c_effect": same["C"],
                          "random_baseline_a1_heldout": _ev(backend, held_prep, chosen, rq)}}
        print(label, json.dumps({"chosen": chosen, "joint": round(greedy["joint"], 3),
                                 "held": report[label]["das_rank1"]["a1_heldout"]["fraction_of_exact"],
                                 "a2": report[label]["das_rank1"]["a2"]["fraction_of_exact"],
                                 "P": round(same["P"], 3), "C": round(same["C"], 3)}))

    # numbered list: direction across constructions
    a1, a2 = g.rows_of(m_list, "A1"), g.rows_of(m_list, "A2")
    a1e, a1o, a2e, a2o = (g.prepare(backend, r) for r in (a1[0::2], a1[1::2], a2[0::2], a2[1::2]))
    q_a2, h_a2 = g.fit_joint_subspace(backend, a2e, LIST_HEADS, rank=RANK, steps=DAS_STEPS)
    both = g.prepare(backend, a1[0::2] + a2[0::2])
    q_both, h_both = g.fit_joint_subspace(backend, both, LIST_HEADS, rank=RANK, steps=DAS_STEPS)
    lst = {"heads": LIST_HEADS,
           "fit_on_a2": {"loss_history": h_a2, "a2_heldout": _ev(backend, a2o, LIST_HEADS, q_a2),
                         "a1_all": _ev(backend, g.prepare(backend, a1), LIST_HEADS, q_a2)},
           "fit_on_both": {"loss_history": h_both,
                           "a1_heldout": _ev(backend, a1o, LIST_HEADS, q_both),
                           "a2_heldout": _ev(backend, a2o, LIST_HEADS, q_both)},
           "cosine_between_fits": float((q_a2[:, 0] @ q_both[:, 0]).abs())}
    report["numbered_list.control_choice_discriminator"] = lst
    print("numbered_list", json.dumps({k: (v if not isinstance(v, dict) else
                                           {kk: (vv["fraction_of_exact"] if isinstance(vv, dict) else None)
                                            for kk, vv in v.items() if kk != "loss_history"})
                                       for k, v in lst.items()}))

    def band(k):
        d = report[k]["das_rank1"]
        return (all(LO <= (d[f]["fraction_of_exact"] or 0.0) <= HI for f in ("a1_heldout", "a2"))
                and d["p_effect"] <= P_BAR and d["c_effect"] <= C_BAR
                and abs(d["random_baseline_a1_heldout"]["fraction_of_exact"] or 0.0) <= RANDOM_MAX)
    two = ["polarity_state.negative_vs_positive", "possessive_number.adjacent_antecedent"]
    fr = lambda x: x["fraction_of_exact"] or 0.0
    predictions = {
        "pred_a_heads_only_sets_reach_the_bar": all(report[k]["greedy"]["reached_target"] for k in two),
        "pred_b_heads_only_rank1_in_band": all(band(k) for k in two),
        "pred_c_list_direction_is_construction_specific":
            fr(lst["fit_on_a2"]["a2_heldout"]) >= LO and fr(lst["fit_on_a2"]["a1_all"]) < LO,
        "pred_d_one_direction_serves_both_list_constructions":
            fr(lst["fit_on_both"]["a1_heldout"]) >= LO and fr(lst["fit_on_both"]["a2_heldout"]) >= LO}
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_greedy_heads_only_result_v3",
              "candidate_id": "corpus.unit_greedy_heads_only_v3", "instrument": instrument,
              "registered": {"pool": POOL, "target": TARGET, "min_gain": MIN_GAIN,
                             "max_units": MAX_UNITS, "rank": RANK, "das_steps": DAS_STEPS,
                             "band": [LO, HI], "p_bar": P_BAR, "c_bar": C_BAR,
                             "random_max": RANDOM_MAX, "units": "heads only",
                             "das_objective": "match_exact_set_patch"},
              "predictions": predictions, "behaviours": report,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
