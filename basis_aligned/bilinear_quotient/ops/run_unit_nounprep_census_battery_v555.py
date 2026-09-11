#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v555: take the battery out of one construction, and fold the census into it.

TWO CHANGES AT ONCE, BOTH EARNED BY EARLIER RUNGS. First, every four-hypothesis pass so far except two is in
verb_preposition. v511 measured which constructions have a head set that reaches a vocabulary-disjoint task in their
OWN construction -- the precondition for an informative C row -- and found three of six: correlative, verb_preposition
and noun_preposition at 0.913. correlative has one task with a control I had to author; verb_preposition has twelve
passes. noun_preposition is the remaining family where the battery can run with controls that can fail, and it has
had exactly one target through it (v523).
Second, the census and the battery have been separate rungs because a battery over a pair whose partner is NOT
reached produces an uninformative C row. The per-target reporting added in v551 makes that separable inside ONE
receipt: each target carries its own partner reach, and its pass flag requires BOTH the four bars and a reached
partner. So one rung now does what took two.
Five pairs, five DISTINCT partners, every pair vocabulary-disjoint -- distinct because v529 showed reusing partners
inflates the rate and v531 showed availability is per-pair:
    at_beyond    -> belief        {at,beyond}   vs {in,for}
    between_with -> at_toward     {between,with} vs {at,toward}
    for_about    -> on_toward     {for,about}   vs {on,toward}
    reason       -> between_with  {for,to}      vs {between,with}
    within_for   -> at_beyond     {within,for}  vs {at,beyond}
interest is left out: it passed in v523 already, and no sixth partner in this family is disjoint from its
(` in`, ` for`) vocabulary -- a limit of the family, not a choice.
RANK FIXED AT 1 AND REGISTERED. The dead P recovery call removed in v553 stays out. A CAVEAT THAT TRAVELS WITH EVERY
PASS IN THIS FAMILY: v489 showed the model here is a recency reader -- interpose the other cue noun and all 32 rows
fail -- so a pass says the subspace carries the variable AS THE MODEL COMPUTES IT, by recency, not that it encodes
noun-governed complement selection. Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_measured_A1  every group that produced data reaches MINIMUM held-out joint extraction joint_min = 0.80.
                                                                                                      prior 85%
  pred_b_measured_A2  every measured group reaches 0.80 on its A2 construction.                       prior 75%
  pred_c_measured_P_and_C every measured group has BOTH its P and its partner C same-answer effects at or below
                      p_max = 0.23.                                                                   prior 80%
  pred_d_partners_reached every partner reports units_recovery >= unit_min = 0.80 -- the CENSUS half, and the clause
                      that decides whether each C row means anything. v511 measured one pair in this family at
                      0.913; four of these five are new.                                              prior 65%
  pred_e_all_measured ZERO groups error.                                                              prior 85%
HOW IT READS. All true: five noun_preposition targets pass all four hypotheses with informative controls, the
battery is no longer a single-family result, and census-plus-battery in one rung halves the cycle. d FALSE: some
partners are unreached, and those targets' C rows are uninformative regardless of their value -- reported per target
as measured-but-not-controlled rather than as passes, which is the distinction the whole control thread exists to
draw. b FALSE: A2 is where this family is most likely to break, since its A2 frames differ more from A1 than
verb_preposition's do.
SCOPE. Five tasks of nine in this family, rank 1, one partner each, with the recency caveat attached to any pass.
Smoke: V555_SMOKE=<out.json> (CPU, V555_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_nounprep_census_battery_v555_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "noun_preposition_"
PAIRS = {"at_beyond": "belief", "between_with": "at_toward", "for_about": "on_toward",
         "reason": "between_with", "within_for": "at_beyond"}      # five DISTINCT partners
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v511_family_reach": 0.913, "v489_recency": "model follows the nearest noun here"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_nounprep_census_battery_v555", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ks = list(PAIRS)
    measured = [k for k in ks if k in G and "error" not in G[k]]
    errored = [k for k in ks if k not in G or "error" in G[k]]

    def ps(k):
        return G.get(k, {}).get("per_shape", {})

    def hyp(k):
        return G.get(k, {}).get("hypotheses", {})

    def vals(k):
        return (ps(k).get(VP + k, {}).get("joint_extraction"), hyp(k).get("A2_extraction"),
                hyp(k).get("P_same_answer_effect"), hyp(k).get("C_same_answer_effect"))

    have = [k for k in measured if all(v is not None for v in vals(k))]
    a = bool(have) and all(vals(k)[0] >= B["joint_min"] for k in have)
    b = bool(have) and all(vals(k)[1] >= B["joint_min"] for k in have)
    c = bool(have) and all(abs(vals(k)[2]) <= B["p_max"] and abs(vals(k)[3]) <= B["p_max"] for k in have)
    reach = {k: ps(k).get(VP + PAIRS[k], {}).get("units_recovery") for k in ks}
    d = bool(have) and all(reach[k] is not None and reach[k] >= B["unit_min"] for k in have)
    e = len(errored) == 0
    R["per_target"] = {k: {"measured": k in have, "A1": vals(k)[0], "A2": vals(k)[1],
                           "P": vals(k)[2], "C": vals(k)[3],
                           "reach": reach.get(k),
                           "passes": bool(k in have and (reach.get(k) or 0) >= B["unit_min"]
                                          and vals(k)[0] >= B["joint_min"]
                                          and vals(k)[1] >= B["joint_min"]
                                          and abs(vals(k)[2]) <= B["p_max"]
                                          and abs(vals(k)[3]) <= B["p_max"])}
                       for k in ks}
    R["errored_groups"] = errored
    return {"pred_a_measured_A1": bool(a), "pred_b_measured_A2": bool(b),
            "pred_c_measured_P_and_C": bool(c), "pred_d_partners_reached": bool(d),
            "pred_e_all_measured": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V555_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V555_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                   "n_a2": len(p_a2.base_batch.row_ids), "n_p": len(p_p.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "hypotheses": hyp}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "per_target": R.get("per_target"),
              "errored_groups": R.get("errored_groups"), "schema": "unit_nounprep_census_battery_v555",
              "candidate_id": "corpus.unit_nounprep_census_battery_v555", "bars": BARS,
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
