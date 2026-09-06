#!/usr/bin/env python3
# BQGATE: frozen predictions; pool size, target, gain floor, size cap and DAS ranks fixed before the run.
"""Full-component protocol on five behaviours: module sweep, unit sweep, greedy minimal set, joint DAS.

Replaces the retracted `resid:18` localizations (tautological: 50/50 behaviours at 1.000). Every
number here is a component patched inside a real forward. Library: `circuit_unit_greedy.py`.

Behaviours, chosen to span the corpus's module profile (best whole module from the receipts):
    numbered_list.control_choice_discriminator   attn:08 0.924   concentrated reference
    modal_remoteness.would_vs_will               attn:09 0.464
    possessive_number.adjacent_antecedent        attn:04 0.322
    correlative_pair.both_vs_neither             attn:08 0.276   the cleanest P/C profile
    polarity_state.negative_vs_positive          attn:07 0.264

REGISTERED BEFORE THE RUN
    units      all 162 heads (pre-c_proj 128-d slices) + all 18 MLP outputs, at the semantic position
    pool       the 12 units with the highest single-unit A1 recovery
    greedy     target 0.50 joint recovery (the corpus's A1 bar), gain floor 0.02, at most 6 units
    DAS        rank 1 AND rank |S| on the chosen set S, both registered now; fit on the first 16
               A1 rows (even index), 200 Adam steps, evaluated on the 16 held-out A1 rows, A2, P, C
    P / C      same-answer effect of patching the chosen set with the P- / C-family donors, scaled
               by the A1 median native separation -- the screens' bars are P <= 0.20, C <= 0.35

    pred_a_single_unit_suffices_only_for_the_concentrated_reference
        control_choice has a single unit >= 0.50; none of the other four does.
    pred_b_greedy_set_reaches_the_bar
        for each of the four distributed behaviours the greedy set reaches joint >= 0.50 with
        at most 6 units.  (Stated prior: unsure. If this fails the effect is spread beyond a
        handful of components at this position, and that is the finding.)
    pred_c_chosen_set_is_selective
        for every behaviour whose set reached the bar: P <= 0.20 and C <= 0.35 under the exact
        patch of that set.
    pred_d_joint_rank1_transfers
        rank-1 joint DAS on the set recovers >= 0.50 of the exact-set joint on held-out A1 and
        on A2, for every behaviour whose set reached the bar.
    pred_e_sets_are_subadditive
        joint(S) < sum of singles over S on at least 3 of 5 behaviours (units share information).
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
import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_candidate_possessive_adjacent as m_poss
import circuit_fast_screen_candidate_correlative_pair as m_corr
import circuit_fast_screen_candidate_polarity_state as m_pol

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_greedy_protocol_v1_result.json"
BEHAVIOURS = (("numbered_list.control_choice_discriminator", m_list, True),
              ("modal_remoteness.would_vs_will", m_modal, False),
              ("possessive_number.adjacent_antecedent", m_poss, False),
              ("correlative_pair.both_vs_neither", m_corr, False),
              ("polarity_state.negative_vs_positive", m_pol, False))
POOL, TARGET, MIN_GAIN, MAX_UNITS = 12, 0.50, 0.02, 6
SINGLE_BAR, P_BAR, C_BAR, TRANSFER = 0.50, 0.20, 0.35, 0.50
DAS_STEPS = 200
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 6000, 200000


def _plan():
    return {"candidate_id": "corpus.unit_greedy_protocol_v1",
            "behaviours": [b for b, _, _ in BEHAVIOURS],
            "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
            "das_ranks": "1 and |S|", "das_steps": DAS_STEPS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * DAS_STEPS * len(BEHAVIOURS), "model_updates": 0,
            "fit_parameters": 41472,  # upper bound: (6 units x 1152) x rank 6, per fit
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _das_eval(backend, prep, units, q):
    """Recovery through the subspace alone on an answer-changing family, vs the exact patch."""
    exact = g.recovery(prep, g.patched_axis(backend, prep, units))
    sub = g.recovery(prep, g.patched_axis(backend, prep, units, q=q))
    return {"exact_set": exact, "subspace": sub,
            "fraction_of_exact": (sub / exact) if abs(exact) > 1e-6 else None}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()

    a1_rows = g.rows_of(m_corr, "A1")
    instrument = g.verify_against_producer(backend, a1_rows, layer=8, heads=(1, 3, 7), mlp_layer=8)
    print("instrument:", json.dumps(instrument))
    if not instrument["passed"]:
        raise SystemExit("new forward does not reproduce the producer; nothing below is valid")

    units = g.all_head_units() + g.all_mlp_units()
    report = {}
    for label, module, concentrated in BEHAVIOURS:
        a1 = g.rows_of(module, "A1")
        prep = g.prepare(backend, a1)
        scale = g.target_scale(prep)
        modules = g.module_sweep(backend, prep)
        singles = g.unit_sweep(backend, prep, units)
        ranked = sorted(singles, key=singles.get, reverse=True)
        pool = ranked[:POOL]
        greedy = g.greedy_select(lambda s: g.recovery(prep, g.patched_axis(backend, prep, s)),
                                 pool, target=TARGET, min_gain=MIN_GAIN, max_units=MAX_UNITS)
        chosen = greedy["chosen"]
        sum_singles = sum(singles[u] for u in chosen)

        # selectivity of the EXACT set patch on the same-answer families, and transfer to A2
        a2_prep = g.prepare(backend, g.rows_of(module, "A2"))
        a2_exact = g.recovery(a2_prep, g.patched_axis(backend, a2_prep, chosen))
        same = {}
        for fam in ("P", "C"):
            fprep = g.prepare(backend, g.rows_of(module, fam))
            same[fam] = g.same_answer_effect(fprep, g.patched_axis(backend, fprep, chosen), scale)

        # joint DAS on the chosen set: fit on even A1 rows, evaluate on odd A1 rows and A2
        fit_prep = g.prepare(backend, a1[0::2])
        held_prep = g.prepare(backend, a1[1::2])
        das = {}
        for rank in sorted({1, len(chosen)}):
            q, hist = g.fit_joint_subspace(backend, fit_prep, chosen, rank=rank, steps=DAS_STEPS)
            das[f"rank_{rank}"] = {
                "loss_history": hist,
                "a1_fit": _das_eval(backend, fit_prep, chosen, q),
                "a1_heldout": _das_eval(backend, held_prep, chosen, q),
                "a2": _das_eval(backend, a2_prep, chosen, q),
                "p_effect": g.same_answer_effect(
                    (pp := g.prepare(backend, g.rows_of(module, "P"))),
                    g.patched_axis(backend, pp, chosen, q=q), scale),
                "c_effect": g.same_answer_effect(
                    (cp := g.prepare(backend, g.rows_of(module, "C"))),
                    g.patched_axis(backend, cp, chosen, q=q), scale)}
            del q
        report[label] = {
            "rows_a1": len(a1), "target_scale": scale,
            "modules": modules,
            "best_module": max(modules, key=modules.get),
            "best_module_recovery": max(modules.values()),
            "top_units": {u: singles[u] for u in ranked[:20]},
            "best_unit": ranked[0], "best_unit_recovery": singles[ranked[0]],
            "pool": pool, "greedy": greedy,
            "chosen": chosen, "joint": greedy["joint"], "sum_of_singles": sum_singles,
            "joint_over_sum": (greedy["joint"] / sum_singles) if abs(sum_singles) > 1e-6 else None,
            "a2_exact_set": a2_exact, "p_effect_exact_set": same["P"],
            "c_effect_exact_set": same["C"], "das": das,
            "concentrated_reference": concentrated}
        print(label, json.dumps({"best_module": report[label]["best_module"],
                                 "best_module_recovery": round(max(modules.values()), 3),
                                 "best_unit": ranked[0], "best_unit_recovery": round(singles[ranked[0]], 3),
                                 "chosen": chosen, "joint": round(greedy["joint"], 3),
                                 "a2": round(a2_exact, 3), "P": round(same["P"], 3),
                                 "C": round(same["C"], 3),
                                 "das_rank1_heldout": das["rank_1"]["a1_heldout"],
                                 "das_rank1_a2": das["rank_1"]["a2"]}))
        torch.cuda.empty_cache()

    conc = [k for k, _, c in BEHAVIOURS if c][0]
    dist = [k for k, _, c in BEHAVIOURS if not c]
    reached = [k for k in dist if report[k]["greedy"]["reached_target"]]
    pred_a = (report[conc]["best_unit_recovery"] >= SINGLE_BAR
              and all(report[k]["best_unit_recovery"] < SINGLE_BAR for k in dist))
    pred_b = len(reached) == len(dist)
    pred_c = bool(reached) and all(report[k]["p_effect_exact_set"] <= P_BAR
                                   and report[k]["c_effect_exact_set"] <= C_BAR for k in reached)
    def _tr(k):
        d = report[k]["das"]["rank_1"]
        return all((d[f]["fraction_of_exact"] or 0.0) >= TRANSFER for f in ("a1_heldout", "a2"))
    pred_d = bool(reached) and all(_tr(k) for k in reached)
    subadd = sum(1 for k in report if report[k]["joint"] < report[k]["sum_of_singles"])
    pred_e = subadd >= 3
    predictions = {
        "pred_a_single_unit_suffices_only_for_the_concentrated_reference": bool(pred_a),
        "pred_b_greedy_set_reaches_the_bar": bool(pred_b),
        "pred_c_chosen_set_is_selective": bool(pred_c),
        "pred_d_joint_rank1_transfers": bool(pred_d),
        "pred_e_sets_are_subadditive": bool(pred_e)}
    result = {"schema": "circuit_unit_greedy_protocol_result_v1",
              "candidate_id": "corpus.unit_greedy_protocol_v1",
              "instrument": instrument,
              "registered": {"pool": POOL, "target": TARGET, "min_gain": MIN_GAIN,
                             "max_units": MAX_UNITS, "single_bar": SINGLE_BAR, "p_bar": P_BAR,
                             "c_bar": C_BAR, "transfer": TRANSFER, "das_steps": DAS_STEPS},
              "predictions": predictions,
              "distributed_behaviours_reaching_target": reached,
              "subadditive_count": subadd,
              "behaviours": report,
              "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "reached": reached, "subadditive": subadd,
                      "seconds": result["serial_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
