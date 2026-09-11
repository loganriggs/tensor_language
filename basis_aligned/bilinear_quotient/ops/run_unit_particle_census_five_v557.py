#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v557: does v511's per-family classification fail in BOTH directions, or only the one I caught?

WHAT v555 EXPOSED. v511 classified six constructions by whether a fitted head set reaches a vocabulary-disjoint task
in its own construction -- the precondition for an informative C row -- and got three of six. Every one of those six
readings came from ONE pair. v555 then measured five NEW pairs in noun_preposition, which v511 had called
construction-general on a single 0.913, and found reach of 0.500, 0.601, 0.757, 0.793 and 0.841: one of five clears
the floor. So a single pair did not characterise that family, and I said so.
THAT LEAVES THE OBVIOUS SYMMETRIC QUESTION UNANSWERED, WHICH IS WHY THIS RUNG EXISTS. If one pair can make a family
look better than it is, it can make one look worse. verb_particle was labelled NOT construction-general on a single
reading of 0.677 -- up_down against out_away -- and on that basis I have been treating row 4 as unavailable for the
whole family, including in v517, v519 and v521 which spent three rungs asking WHY its nodes do not reach. If other
pairs in it reach, that premise was never established.
Five new pairs, five DISTINCT partners, every pair vocabulary-disjoint, none of the five fitted tasks fitted before:
    out_down -> away_up   {out,down} vs {away,up}        down_in  -> on_away  {down,in} vs {on,away}
    on_out   -> down_in   {on,out}   vs {down,in}        up_on    -> out_away {up,on}   vs {out,away}
    away_up  -> on_out    {away,up}  vs {on,out}
The v511 pair is re-measured as the anchor. Standard budget; rank 1; nothing counted.
A CAVEAT THAT TRAVELS WITH THIS FAMILY: v477 established that its stimuli put the particle where English licenses
none, and that the direction transfers between that slot and a grammatical one, so the family is measurable -- but
any reach figure here is about tasks in that off-distribution slot.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold      all five fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80. prior 85%
  pred_b_some_reachable at least ONE of the five partners reports units_recovery >= unit_min = 0.80. If TRUE, the
                        family's "not construction-general" label was one-pair-based and wrong in the same way
                        noun_preposition's was, and three rungs spent asking why its nodes do not reach were asking
                        about one pair rather than a family.                                          prior 55%
  pred_c_reached_are_inert every partner clearing the floor has ABSOLUTE recovery at or below cross_max = 0.30,
                        quantified over the reached ones and vacuously false if none is.              prior 80%
  pred_d_majority_unreachable at most TWO of five reach the floor, i.e. v511's low reading broadly holds even if it
                        was not exact. Worked example: 0 to 2 of 5, TRUE and the label survives as a tendency;
                        3 or more, FALSE and the label is simply wrong.                               prior 60%
  pred_e_anchor_reproduces up_down -> out_away lands within tol_pool = 0.05 of v511's 0.677. Deterministic fit;
                        cross-receipt consistency, and if it drifts nothing else here compares.       prior 85%
HOW IT READS. b FALSE with d TRUE: the label holds, verb_particle really is a family whose node sets stay home, and
v511's reading was lucky rather than representative-but-right. b TRUE with d TRUE: mixed, one to two of five reach,
and the honest description is a RATE rather than a label -- which would mean v511's six one-pair readings should all
be reported as rates with a sample size of one. d FALSE: three or more reach, the label is wrong, and the three rungs
that asked why this family's nodes do not reach were built on a premise that was never established -- the most
consequential outcome and the reason this is worth 200 seconds.
SCOPE. Five tasks of nine in this family, rank 1, one partner each.
Smoke: V557_SMOKE=<out.json> (CPU, V557_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_particle_census_five_v557_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_particle_"
PAIRS = {"out_down": "away_up", "down_in": "on_away", "on_out": "down_in",
         "up_on": "out_away", "away_up": "on_out", "up_down": "out_away"}   # five new + the v511 anchor
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
ANCHOR_KEY = "up_down"
PRIOR = {"v511_anchor": 0.677, "v555_nounprep": "1 of 5 reached"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_particle_census_five_v557", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    ks = list(PAIRS)
    fit_v = [fld(k, VP + k, "joint_extraction") for k in ks]
    reach = {k: fld(k, VP + PAIRS[k], "units_recovery") for k in ks}
    absr = {k: fld(k, VP + PAIRS[k], "cross_abs_recovery") for k in ks}
    have = ok and all(x is not None for x in fit_v) \
        and all(v is not None for v in reach.values()) and all(v is not None for v in absr.values())
    new = [k for k in ks if k != ANCHOR_KEY]
    a = have and min(fit_v) >= B["joint_min"]
    b = have and any(reach[k] >= B["unit_min"] for k in new)
    reached = [k for k in new if have and reach[k] >= B["unit_min"]]
    c = bool(reached) and all(abs(absr[k]) <= B["cross_max"] for k in reached)
    d = have and sum(1 for k in new if reach[k] >= B["unit_min"]) <= 2
    e = have and abs(reach[ANCHOR_KEY] - PRIOR["v511_anchor"]) <= B["tol_pool"]
    return {"pred_a_fits_hold": bool(a), "pred_b_some_reachable": bool(b),
            "pred_c_reached_are_inert": bool(c), "pred_d_majority_unreachable": bool(d),
            "pred_e_anchor_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V557_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V557_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            pooled_fit, pooled_c_fit, per_shape_rows = [], [], {}
            for cellname in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cellname}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cellname] = (cut(held_half(a1)), cut(held_half(cc)))
                if cellname in cells:                       # only this mapping's own frames enter the fit
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
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
                controls=(P_cfit,), control_weight=LAM, mu=mu_joint)
            per_shape = {}
            for cellname in EVAL[gname]:
                held_rows, c_rows = per_shape_rows[cellname]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cellname] = {
                    "units_recovery": e_exact,
                    "cross_abs_recovery": round(ext(p_held, units, q=q_joint), 3),
                    "joint_extraction": round(ext(p_held, units, q=q_joint) / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                    "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                    "C_damage": cdmg, "C_ub975": cdmg["ce_ub975"], "n_held": len(p_held.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_particle_census_five_v557",
              "candidate_id": "corpus.unit_particle_census_five_v557", "bars": BARS,
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
