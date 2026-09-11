#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v565: spend verb_particle's one reachable pair, and ask what a NEAR-MISS partner is worth.

WHY. v557 censused verb_particle with five new pairs and found exactly ONE reachable partner --
down_in -> verb_particle_on_away at 0.815 -- with the v511 anchor reproducing 0.677 exactly. That pair has never
been spent, so the family has no four-hypothesis receipt at all. This rung spends it, which would put the battery in
a FOURTH construction.
THE SECOND QUESTION, AND WHY IT IS NOT AN EXCUSE TO MOVE THE BAR. Two v557 partners landed just under the floor:
away_up -> on_out at 0.772 and out_down -> away_up at 0.779. I have separately seen the specificity instrument still
register real movement at reach 0.694, so the 0.80 floor is a convention rather than a measured cliff, and the
tempting move is to lower it and collect three verb_particle circuits instead of one. I am NOT doing that, because
the objection to a near-miss partner is not the size of its C row. At reach 0.77 roughly a quarter of the
interchange is unaccounted for, so an inert C is consistent with real movement hiding in the unreached quarter, and
no measurement of C can rule that out. So the near-miss rows are registered as PROVISIONAL, are reported as such,
and DO NOT COUNT as circuits whatever they show. What they can do is motivate the follow-up: if they behave exactly
like the reached pair, the next rung raises their reach with the relaxed budget and spends them properly; if they do
not, the floor is doing real work and the question is closed.
RANK FIXED AT 1 AND REGISTERED. Nothing counted here: DAS receipts are not new behaviours.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_primary_A1   the primary pair down_in holds held-out joint extraction at or above joint_min = 0.80.
                                                                                                      prior 88%
  pred_b_primary_A2   down_in reaches 0.80 on its A2 construction.                                     prior 78%
  pred_c_primary_P_and_C down_in has BOTH its P and its partner C same-answer effects at or below p_max = 0.23,
                      which with a and b makes it a verb_particle circuit passing all four.            prior 75%
  pred_d_primary_reach down_in's partner on_away reports units_recovery >= unit_min = 0.80, reproducing v557's
                      0.815 within tol_pool = 0.05.                                                    prior 90%
  pred_e_nearmiss_like_primary BOTH provisional near-miss pairs have C same-answer effects at or below p_max, as the
                      primary does. This bar can fail: the same instrument reads 0.54 to 0.82 when a partner really
                      moves, against 0.003 to 0.028 when it is inert.                                  prior 65%
HOW IT READS, one outcome per branch. a, b, c, d all true: verb_particle gets its first task passing all four and
the battery reaches a FOURTH construction. c FALSE: the family's one reachable partner does not support a clean C
or P row, and verb_particle is a family where row 4 is measurable but not passed -- a different and more interesting
claim than unreachable. e TRUE: the near-misses look like the pair that cleared, which licenses ONE follow-up at the
relaxed budget to raise their reach, not a lower floor. e FALSE: the floor is separating real cases and the
near-miss question is closed.
SCOPE. One family, one spendable pair plus two provisional ones, rank 1, one partner each.
Smoke: V565_SMOKE=<out.json> (CPU, V565_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_particle_battery_nearmiss_v565_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "verb_particle_"
PAIRS = {"down_in": "on_away", "away_up": "on_out", "out_down": "away_up"}
PRIMARY = "down_in"                       # the only v557 partner at or above the floor
PROVISIONAL = ("away_up", "out_down")     # 0.772 and 0.779 -- reported, never counted
V557_REACH = {"down_in": 0.815, "away_up": 0.772, "out_down": 0.779}
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v557_reach": V557_REACH, "v511_anchor": 0.677, "instrument_range": "0.54-0.82 moved vs 0.003-0.028 inert"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_particle_battery_nearmiss_v565", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    reach = {k: ps(k).get(VP + PAIRS[k], {}).get("units_recovery") for k in ks}
    pr = PRIMARY
    ok = pr in have
    a = bool(ok) and vals(pr)[0] >= B["joint_min"]
    b = bool(ok) and vals(pr)[1] >= B["joint_min"]
    c = bool(ok) and abs(vals(pr)[2]) <= B["p_max"] and abs(vals(pr)[3]) <= B["p_max"]
    d = bool(ok) and reach[pr] is not None and reach[pr] >= B["unit_min"] \
        and abs(reach[pr] - V557_REACH[pr]) <= B["tol_pool"]
    prov = [k for k in PROVISIONAL if k in have and vals(k)[3] is not None]
    e = len(prov) == len(PROVISIONAL) and all(abs(vals(k)[3]) <= B["p_max"] for k in prov)
    R["per_target"] = {k: {"measured": k in have, "A1": vals(k)[0], "A2": vals(k)[1],
                           "P": vals(k)[2], "C": vals(k)[3], "reach": reach.get(k),
                           "provisional": k in PROVISIONAL,
                           "passes": bool(k == pr and k in have and (reach.get(k) or 0) >= B["unit_min"]
                                          and vals(k)[0] >= B["joint_min"]
                                          and vals(k)[1] >= B["joint_min"]
                                          and abs(vals(k)[2]) <= B["p_max"]
                                          and abs(vals(k)[3]) <= B["p_max"])}
                       for k in ks}
    R["errored_groups"] = errored
    R["note"] = "provisional pairs are reported and never counted: reach below unit_min leaves part of the interchange unaccounted for"
    return {"pred_a_primary_A1": bool(a), "pred_b_primary_A2": bool(b),
            "pred_c_primary_P_and_C": bool(c), "pred_d_primary_reach": bool(d),
            "pred_e_nearmiss_like_primary": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V565_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V565_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
              "errored_groups": R.get("errored_groups"), "schema": "unit_particle_battery_nearmiss_v565",
              "candidate_id": "corpus.unit_particle_battery_nearmiss_v565", "bars": BARS,
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
