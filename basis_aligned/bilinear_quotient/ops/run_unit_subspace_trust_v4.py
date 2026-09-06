#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, ranks, weights, pools and bars fixed before the run.
"""v4: can a learned rank-1 direction over a head set be trusted, and what is the MLP equivalent?

The user's question after v2/v3: "How are you supposed to trust that for the attention heads and
then how would you do some equivalent for the MLP?" -- and then: "Can't you just optimize for
both this and the complement to have these properties?"

This run puts every set from v2/v3 through the same battery and adds the constrained fit.

  SETS (from `unit_greedy_protocol_v2_result.json` / `unit_greedy_heads_only_v3_result.json`)
    head sets   correlative 3 heads, modal 2 heads, list 2 heads, polarity 4 heads (v3),
                possessive 4 heads (v3)
    mlp sets    polarity v2 [07:08, 08:01, mlp:04, 10:05], possessive v2 [04:05, mlp:08, 09:06, 10:05]

  DIRECTIONS (rank 1 over the concatenated unit space, fit on even A1 rows)
    dim    diff-in-means: normalised mean of (donor - base). No search freedom at all.
    das    plain DAS, exact-set objective (v2 protocol).
    cdas   constrained DAS: exact-set objective + lambda * (complement patch must reproduce the
           BASE margin), lambda = 1. The user's suggestion.
    rand   random unit vector, seed 1.

  MEASUREMENTS for each direction: subspace patch (fraction of the exact-set effect) on held-out
    A1 and on A2; COMPLEMENT patch (swap everything but the direction) on held-out A1 as a
    fraction of the exact-set effect; P and C effects through the subspace patch; cosine to dim.

  MLP EQUIVALENT: for mlp:04 (polarity) and mlp:08 (possessive): exact single-neuron interchange
    of all 4608 hidden units (bilinear product terms Left*Right, pre Down -- the model's own
    basis, no rotation),
    then greedy over the top 12 neurons towards 0.8 of the whole module's effect.

REGISTERED BEFORE THE RUN
    fit even A1 / evaluate odd A1 and all A2; 200 Adam steps, lr 0.05, seed 0; lambda 1;
    band [0.50, 1.20] on fraction of exact; complement bar 0.30; random bar 0.10;
    P <= 0.20, C <= 0.35; neuron pool 12, gain floor 0.02, at most 8 neurons.

    pred_a_diff_in_means_carries_head_sets   all 5 head sets: dim held-out and A2 fraction in band
    pred_b_das_agrees_with_dim_on_head_sets  all 5 head sets: |das - dim| held-out fraction <= 0.30
    pred_c_head_set_complements_inert        all 5 head sets: das complement fraction <= 0.30
    pred_d_mlp_sets_show_the_illusion        both mlp sets: plain das held-out fraction > 1.20 OR
                                             complement fraction > 0.30 (the v2 failure, now
                                             measured with the decisive test)
    pred_e_constrained_das_repairs_mlp_sets  both mlp sets: cdas held-out and A2 in band,
                                             complement <= 0.30, random complement irrelevant,
                                             P/C at bar
    pred_f_dozen_neurons_carry_the_mlp       both MLPs: <= 8 of the top-12 neurons reach 0.8 of
                                             the whole module's exact effect

    Stated priors. a, b, c: expected to hold -- head slices are 128-d and low gain, and v2/v3
    already showed random ~0. d: expected. e: UNSURE, this is the user's question; a fixed
    direction that both matches and leaves an inert complement is exactly what a real variable
    looks like, and I do not know whether one exists for an 1152-d MLP output. f: expected to
    FAIL (hidden units are usually polysemantic; the effect is likely spread), in which case
    the report gives how many neurons the greedy actually needed and how far it got.
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
import circuit_fast_screen_candidate_correlative_pair as m_corr
import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_candidate_possessive_adjacent as m_poss
import circuit_fast_screen_candidate_polarity_state as m_pol

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_subspace_trust_v4_result.json"
RANK, DAS_STEPS, LR, LAMBDA = 1, 200, 0.05, 1.0
LO, HI, COMP_BAR, RANDOM_MAX, P_BAR, C_BAR, GAP = 0.50, 1.20, 0.30, 0.10, 0.20, 0.35, 0.30
NEURON_POOL, NEURON_TARGET_FRACTION, NEURON_MIN_GAIN, NEURON_MAX = 12, 0.8, 0.02, 8
SETS = {
    "correlative_pair.both_vs_neither": (m_corr, "heads",
        ["attn:08:head:01", "attn:07:head:08", "attn:14:head:08"]),
    "modal_remoteness.would_vs_will": (m_modal, "heads", ["attn:09:head:04", "attn:11:head:03"]),
    "numbered_list.control_choice_discriminator": (m_list, "heads",
        ["attn:08:head:03", "attn:08:head:07"]),
    "polarity_state.negative_vs_positive.heads": (m_pol, "heads",
        ["attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:05:head:08"]),
    "possessive_number.adjacent_antecedent.heads": (m_poss, "heads",
        ["attn:04:head:05", "attn:03:head:04", "attn:09:head:06", "attn:10:head:05"]),
    "polarity_state.negative_vs_positive.with_mlp04": (m_pol, "mlp",
        ["attn:07:head:08", "attn:08:head:01", "mlp:04", "attn:10:head:05"]),
    "possessive_number.adjacent_antecedent.with_mlp08": (m_poss, "mlp",
        ["attn:04:head:05", "mlp:08", "attn:09:head:06", "attn:10:head:05"]),
}
NEURON_TARGETS = {"polarity_state.negative_vs_positive": (m_pol, 4),
                  "possessive_number.adjacent_antecedent": (m_poss, 8)}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 9000, 600000


def _plan():
    return {"candidate_id": "corpus.unit_subspace_trust_v4",
            "sets": {k: v[2] for k, v in SETS.items()},
            "neuron_targets": {k: v[1] for k, v in NEURON_TARGETS.items()},
            "rank": RANK, "das_steps": DAS_STEPS, "lambda": LAMBDA,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 3 * DAS_STEPS * len(SETS), "model_updates": 0,
            "fit_parameters": 1536, "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _frac(prep_exact, value):
    return (value / prep_exact) if abs(prep_exact) > 1e-6 else None


def _direction_report(backend, module, units, q, preps, exacts, scale, q_dim):
    held, a2 = preps["held"], preps["a2"]
    sub_h = g.recovery(held, g.patched_axis(backend, held, units, q=q))
    sub_a2 = g.recovery(a2, g.patched_axis(backend, a2, units, q=q))
    comp_h = g.recovery(held, g.patched_axis(backend, held, units, q=q, complement=True))
    same = {}
    for fam in ("P", "C"):
        fp = g.prepare(backend, g.rows_of(module, fam))
        same[fam] = g.same_answer_effect(fp, g.patched_axis(backend, fp, units, q=q), scale)
    return {"heldout_fraction": _frac(exacts["held"], sub_h),
            "a2_fraction": _frac(exacts["a2"], sub_a2),
            "complement_heldout_fraction": _frac(exacts["held"], comp_h),
            "heldout_recovery": sub_h, "a2_recovery": sub_a2, "complement_recovery": comp_h,
            "p_effect": same["P"], "c_effect": same["C"],
            "cosine_to_dim": float((q[:, 0] @ q_dim[:, 0]).abs())}


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

    # exactness control for the new neuron path: swapping ALL 4608 neurons of an MLP must equal
    # swapping the module output (the module has no other state at that position).
    ctrl_prep = g.prepare(backend, g.rows_of(m_pol, "A1"), hidden=True)
    all_neurons = [f"mlp:04:neuron:{j:04d}" for j in range(g.MLP_HIDDEN)]
    ctrl = {"all_neurons_mlp04": g.recovery(ctrl_prep, g.patched_axis(backend, ctrl_prep, all_neurons)),
            "module_mlp04": g.recovery(ctrl_prep, g.patched_axis(backend, ctrl_prep, ["mlp:04"]))}
    ctrl["passed"] = abs(ctrl["all_neurons_mlp04"] - ctrl["module_mlp04"]) < 1e-3
    print("neuron path control:", json.dumps(ctrl))
    if not ctrl["passed"]:
        raise SystemExit("neuron path does not reproduce the module patch")

    sets_report = {}
    for label, (module, kind, units) in SETS.items():
        a1 = g.rows_of(module, "A1")
        preps = {"fit": g.prepare(backend, a1[0::2]), "held": g.prepare(backend, a1[1::2]),
                 "a2": g.prepare(backend, g.rows_of(module, "A2"))}
        exacts = {k: g.recovery(p, g.patched_axis(backend, p, units)) for k, p in preps.items()}
        scale = g.target_scale(preps["fit"])
        q_dim = g.diff_in_means_direction(backend, preps["fit"], units)
        q_das, h_das = g.fit_joint_subspace(backend, preps["fit"], units, rank=RANK,
                                            steps=DAS_STEPS, lr=LR)
        q_cdas, h_cdas = g.fit_joint_subspace(backend, preps["fit"], units, rank=RANK,
                                              steps=DAS_STEPS, lr=LR, complement_weight=LAMBDA)
        q_rand = g.random_subspace(backend, units, rank=RANK)
        directions = {}
        for name, q in (("dim", q_dim), ("das", q_das), ("cdas", q_cdas), ("rand", q_rand)):
            directions[name] = _direction_report(backend, module, units, q, preps, exacts, scale, q_dim)
        directions["das"]["loss_history"] = h_das
        directions["cdas"]["loss_history"] = h_cdas
        directions["cosine_das_cdas"] = float((q_das[:, 0] @ q_cdas[:, 0]).abs())
        sets_report[label] = {"units": units, "kind": kind, "exact_set": exacts,
                              "directions": directions}
        print(label, json.dumps({n: {k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in d.items() if k in
                                     ("heldout_fraction", "a2_fraction",
                                      "complement_heldout_fraction", "p_effect", "c_effect",
                                      "cosine_to_dim")}
                                 for n, d in directions.items() if isinstance(d, dict)}))

    neurons_report = {}
    for label, (module, layer) in NEURON_TARGETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), hidden=True)
        whole = g.recovery(prep, g.patched_axis(backend, prep, [f"mlp:{layer:02d}"]))
        sweep = g.neuron_sweep(backend, prep, layer)
        ranked = sorted(sweep, key=sweep.get, reverse=True)
        target = NEURON_TARGET_FRACTION * whole
        greedy = g.greedy_select(lambda s: g.recovery(prep, g.patched_axis(backend, prep, s)),
                                 ranked[:NEURON_POOL], target=target,
                                 min_gain=NEURON_MIN_GAIN, max_units=NEURON_MAX)
        top_pool_joint = g.recovery(prep, g.patched_axis(backend, prep, ranked[:NEURON_POOL]))
        top100_joint = g.recovery(prep, g.patched_axis(backend, prep, ranked[:100]))
        a2 = g.prepare(backend, g.rows_of(module, "A2"), hidden=True)
        neurons_report[label] = {
            "layer": layer, "whole_module": whole, "target": target,
            "top_neurons": {u: sweep[u] for u in ranked[:NEURON_POOL]},
            "bottom_neurons": {u: sweep[u] for u in ranked[-5:]},
            "sum_of_top_pool_singles": sum(sweep[u] for u in ranked[:NEURON_POOL]),
            "top_pool_joint": top_pool_joint, "top100_joint": top100_joint,
            "greedy": greedy,
            "chosen_fraction_of_whole": _frac(whole, greedy["joint"]),
            "chosen_a2": g.recovery(a2, g.patched_axis(backend, a2, greedy["chosen"])),
            "whole_a2": g.recovery(a2, g.patched_axis(backend, a2, [f"mlp:{layer:02d}"])),
            "neurons_swept": len(sweep)}
        print(label, json.dumps({"whole": round(whole, 3), "chosen": greedy["chosen"],
                                 "joint": round(greedy["joint"], 3),
                                 "top12_joint": round(top_pool_joint, 3),
                                 "top100_joint": round(top100_joint, 3)}))

    heads = [k for k, v in SETS.items() if v[1] == "heads"]
    mlps = [k for k, v in SETS.items() if v[1] == "mlp"]
    D = lambda k, n: sets_report[k]["directions"][n]
    f = lambda x: x if x is not None else 0.0
    in_band = lambda d: LO <= f(d["heldout_fraction"]) <= HI and LO <= f(d["a2_fraction"]) <= HI
    predictions = {
        "pred_a_diff_in_means_carries_head_sets": all(in_band(D(k, "dim")) for k in heads),
        "pred_b_das_agrees_with_dim_on_head_sets": all(
            abs(f(D(k, "das")["heldout_fraction"]) - f(D(k, "dim")["heldout_fraction"])) <= GAP
            for k in heads),
        "pred_c_head_set_complements_inert": all(
            abs(f(D(k, "das")["complement_heldout_fraction"])) <= COMP_BAR for k in heads),
        "pred_d_mlp_sets_show_the_illusion": all(
            f(D(k, "das")["heldout_fraction"]) > HI
            or abs(f(D(k, "das")["complement_heldout_fraction"])) > COMP_BAR for k in mlps),
        "pred_e_constrained_das_repairs_mlp_sets": all(
            in_band(D(k, "cdas")) and abs(f(D(k, "cdas")["complement_heldout_fraction"])) <= COMP_BAR
            and D(k, "cdas")["p_effect"] <= P_BAR and D(k, "cdas")["c_effect"] <= C_BAR
            for k in mlps),
        "pred_f_dozen_neurons_carry_the_mlp": all(
            r["greedy"]["reached_target"] for r in neurons_report.values()),
    }
    predictions = {k: bool(v) for k, v in predictions.items()}
    result = {"schema": "circuit_unit_subspace_trust_result_v4",
              "candidate_id": "corpus.unit_subspace_trust_v4", "instrument": instrument,
              "neuron_path_control": ctrl,
              "registered": {"rank": RANK, "das_steps": DAS_STEPS, "lr": LR, "lambda": LAMBDA,
                             "band": [LO, HI], "complement_bar": COMP_BAR, "random_max": RANDOM_MAX,
                             "p_bar": P_BAR, "c_bar": C_BAR, "gap": GAP,
                             "neuron_pool": NEURON_POOL, "neuron_target_fraction": NEURON_TARGET_FRACTION,
                             "neuron_min_gain": NEURON_MIN_GAIN, "neuron_max": NEURON_MAX,
                             "das_objective": "match_exact_set_patch"},
              "predictions": predictions, "sets": sets_report, "neurons": neurons_report,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
