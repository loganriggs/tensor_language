#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v545: the four hypotheses on FIVE tasks at once, using partners already measured reachable.

WHY THIS IS THE RIGHT SHAPE NOW. The controlling goal is hundreds of high-quality circuits, and the binding
constraint on quality has turned out to be the CONTROL, not the fit: a specificity reading needs a vocabulary-
disjoint partner the fitted nodes actually reach, and v499, v507 and v511 showed only three constructions of six
have one at all. v527 and v529 then measured that within the largest family and found it common but PER-PAIR --
by_from reaches at_to at 0.932 and at_over at only 0.688, the same fitted task with two different partners. So a
four-hypothesis claim cannot inherit a control; each pair must be checked. THE CHECKS ARE ALREADY DONE FOR FIVE
PAIRS, so this rung spends them rather than re-deriving them.
Each target is paired with a partner whose reach was measured in an earlier receipt, and every pair is
vocabulary-disjoint:
    at_to      -> about_for   0.924 (v487)   -- the ANCHOR, carried to reproduce v525 and catch drift
    by_from    -> at_to       0.932 (v527)
    from_for   -> at_to       1.009 (v527)
    in_to      -> about_for   0.911 (v527)
    about_into -> by_from     0.954 (v529)
RANK IS FIXED AT 1 AND REGISTERED, per the protocol. Instruments as the corpus ships them: extraction for A1 and A2,
`same_answer_effect` scaled by `target_scale` for P and C -- and v535 and v537 established that this measure has real
dynamic range, registering 0.54 to 0.82 on tasks the direction should move against 0.003 to 0.028 on tasks it should
not, so a small reading here is inertness rather than a floor effect. No bar is carried across instruments.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_all_A1     ALL FIVE targets reach MINIMUM held-out joint extraction joint_min = 0.80.        prior 85%
  pred_b_all_A2     all five reach 0.80 on their own A2 construction. Untested on four of these directions.
                                                                                                      prior 60%
  pred_c_all_P_small all five P families show same-answer effect at or below p_max = 0.23.            prior 70%
  pred_d_all_C_small all five partners show same-answer effect at or below 0.23.                      prior 65%
  pred_e_anchor_reproduces at_to's four numbers land within tol_pool = 0.05 of v525's 1.009, 0.976, 0.0189 and
                    0.0064. Deterministic fit; if the anchor drifts, nothing else here is comparable to the four
                    targets already on the board.                                                     prior 85%
HOW IT READS. All true: FIVE tasks in the corpus's largest family pass all four hypotheses at rank 1 with controls
that are inert for an informative reason -- which is the first time this corpus produces verified-selective circuits
in bulk rather than one at a time, and it makes the quality ceiling here authoring time as v527 argued. Any clause
failing is reported PER TARGET rather than as a headline count, because these are five independent claims sharing a
runner and not one claim with five parts; a single target failing row 4 says something about that target and nothing
about the other four.
SCOPE. Five tasks of 119, rank 1, this budget, each with one partner. Nothing here is counted -- the count rules are
Codex's and these are DAS receipts, not new behaviours.
Smoke: V545_SMOKE=<out.json> (CPU, V545_SMOKE_ROWS=4 per split).
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_verbprep_four_hypotheses_bulk_v545_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "verb_preposition_"
PAIRS = {"at_to": "about_for", "by_from": "at_to", "from_for": "at_to",
         "in_to": "about_for", "about_into": "by_from"}     # partner reach measured in v487/v527/v529
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
ANCHOR = {"A1": 1.009, "A2": 0.976, "P": 0.0189, "C": 0.0064}       # v525\nPRIOR = {"anchor": ANCHOR}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_verbprep_four_hypotheses_bulk_v545", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())
    ks = list(PAIRS)

    def ps(k):
        return G.get(k, {}).get("per_shape", {})

    def hyp(k):
        return G.get(k, {}).get("hypotheses", {})

    a1 = {k: ps(k).get(VP + k, {}).get("joint_extraction") for k in ks}
    a2 = {k: hyp(k).get("A2_extraction") for k in ks}
    pe = {k: hyp(k).get("P_same_answer_effect") for k in ks}
    ce = {k: hyp(k).get("C_same_answer_effect") for k in ks}
    have = ok and all(v is not None for d in (a1, a2, pe, ce) for v in d.values())
    a = have and all(a1[k] >= B["joint_min"] for k in ks)
    b = have and all(a2[k] >= B["joint_min"] for k in ks)
    c = have and all(abs(pe[k]) <= B["p_max"] for k in ks)
    d = have and all(abs(ce[k]) <= B["p_max"] for k in ks)
    e = have and abs(a1["at_to"] - ANCHOR["A1"]) <= B["tol_pool"] \
        and abs(a2["at_to"] - ANCHOR["A2"]) <= B["tol_pool"] \
        and abs(abs(pe["at_to"]) - ANCHOR["P"]) <= B["tol_pool"] \
        and abs(abs(ce["at_to"]) - ANCHOR["C"]) <= B["tol_pool"]
    return {"pred_a_all_A1": bool(a), "pred_b_all_A2": bool(b),
            "pred_c_all_P_small": bool(c), "pred_d_all_C_small": bool(d),
            "pred_e_anchor_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V545_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V545_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    V = dict(valid_only=True)

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float()
                                for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0)
                for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)

    groups = {}
    for gname, cells in GROUPS.items():
        try:
            pooled_fit, pooled_c_fit, per_shape_rows, extra_rows = [], [], {}, {}
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell == VP + gname:                  # the other two hypotheses, read on the same direction
                    extra_rows["A2"] = cut(held_half(g.rows_of(m, "A2")))
                    extra_rows["P"] = cut(held_half(g.rows_of(m, "P")))
                if cell in cells:                       # only this mapping's own frames enter the fit
                    pooled_fit += cut(fit_half(a1))
                    pooled_c_fit += cut(fit_half(cc))
            P_fit = g.prepare(backend, pooled_fit, **V)
            P_cfit = g.prepare(backend, pooled_c_fit)
            tgt, mg = (0.97, 0.001)          # every pool in this rung uses the relaxed budget
            singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=tgt,
                                                     min_gain=mg, max_units=max_units)
            units = list(greedy["chosen"])
            mu_joint = mu_of(P_fit, units)
            q_joint, hist = g.fit_block_subspace_constrained(
                backend, P_fit, units, rank=RANK, steps=steps, lr=LR, seed=0, complement_weight=CW,
                controls=(P_cfit,), control_weight=LAM, mu=mu_joint)
            per_shape = {}
            for cell in EVAL[gname]:
                held_rows, c_rows = per_shape_rows[cell]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cell] = {
                    "units_recovery": e_exact,
                    "cross_abs_recovery": round(ext(p_held, units, q=q_joint), 3),
                    "joint_extraction": round(ext(p_held, units, q=q_joint) / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                    "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                    "C_damage": cdmg, "C_ub975": cdmg["ce_ub975"], "n_held": len(p_held.base_batch.row_ids)}
            p_a2 = g.prepare(backend, extra_rows["A2"], **V)
            p_p = g.prepare(backend, extra_rows["P"])
            e_a2 = ext(p_a2, units)
            scale = g.target_scale(g.prepare(backend, per_shape_rows[VP + gname][0], **V))
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            p_disj = g.prepare(backend, per_shape_rows[VP + PAIRS[gname]][0], **V)
            hyp = {"A2_units_recovery": e_a2,
                   "scale": round(scale, 4),
                   "P_same_answer_effect": eff(p_p),
                   "C_same_answer_effect": eff(p_disj),
                   "A2_extraction": round(ext(p_a2, units, q=q_joint) / e_a2, 3) if abs(e_a2) > 1e-6 else None,
                   "P_damage": dmg(p_p, units, q_joint, mu_joint),
                   "P_units_recovery": ext(p_p, units),
                   "n_a2": len(p_a2.base_batch.row_ids), "n_p": len(p_p.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "hypotheses": hyp}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_verbprep_four_hypotheses_bulk_v545",
              "candidate_id": "corpus.unit_verbprep_four_hypotheses_bulk_v545", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "lam": LAM, "steps": steps, "lr": LR, "complement_weight": CW},
              "groups": groups, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "groups": {k: (v.get("error") or {c: s["joint_extraction"] for c, s in v["per_shape"].items()})
                                 for k, v in groups.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
