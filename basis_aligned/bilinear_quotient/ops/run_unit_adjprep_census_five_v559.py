#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v559: the third single-pair family, sampled properly, and chosen to avoid its own confound.

WHERE THIS SITS. v511 classified six constructions by whether a fitted head set reaches a vocabulary-disjoint task in
its own construction -- the precondition for an informative C row -- from ONE pair each. Two of those labels have now
been replaced by rates: noun_preposition went from a single 0.913 to one of five (v555), and verb_particle from a
single 0.677 to one of five (v557). They had sat on OPPOSITE sides of the floor and came out identical. Meanwhile
verb_preposition is genuinely different and well sampled: sixteen of seventeen distinct-partner pairs across v527,
v529 and v547.
adjective_preposition is the next family carrying a single-pair label -- 0.682, from of_at against in_beneath -- and
this rung replaces it with a rate.
THE TARGETS AVOID THIS FAMILY'S OWN CONFOUND. Reading the cue pairs across all thirty-eight tasks found about eight
that pit an adjective selecting a complement against a PASSIVE PARTICIPLE taking a locative adjunct -- hurried past,
hidden beneath, buried beneath, subsumed under, enclosed within, concealed beneath, glimpsed past -- so their
interchanges are confounded with adjective-versus-participle rather than isolating which preposition a word selects.
All five targets here are adjective-versus-adjective, and so are their partners.
    at_about  -> for_of        {at,about}  vs {for,of}        in_to    -> of_against {in,to}   vs {of,against}
    over_for  -> in_about      {over,for}  vs {in,about}      of_amid  -> in_with    {of,amid} vs {in,with}
    with_for  -> of_amid       {with,for}  vs {of,amid}
The v511 pair is re-measured as the anchor. Its partner IS one of the confounded tasks, which is fine for
reproducing a number and would not be fine for making a claim about that task -- the anchor is a consistency check,
not a result. Standard budget; rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold      all five fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80. prior 85%
  pred_b_some_reachable at least ONE of the five new partners reports units_recovery >= unit_min = 0.80. prior 55%
  pred_c_reached_are_inert every partner clearing the floor has ABSOLUTE recovery at or below cross_max = 0.30,
                        quantified over the reached ones and vacuously false if none is.              prior 80%
  pred_d_rate_low       at most TWO of five reach, matching the one-of-five rate the other two properly sampled
                        non-verb_preposition families produced. Worked example: 0 to 2, TRUE and three families now
                        agree; 3 or more, FALSE and this family is unlike them, which would make verb_preposition
                        less of an outlier than it currently looks.                                   prior 65%
  pred_e_anchor_reproduces of_at -> in_beneath lands within tol_pool = 0.05 of v511's 0.682.          prior 85%
HOW IT READS. b TRUE with d TRUE: a THIRD family at one to two of five, and the corpus-level picture becomes
verb_preposition at roughly sixteen of seventeen against everything else at roughly one of five -- a real structural
difference resting on three sampled families rather than six single readings. d FALSE: this family reaches more
often, verb_preposition is less exceptional, and the right description is a spread across families rather than one
outlier. b FALSE with d TRUE: zero of five, lower than the other two, and the honest report is that the rate varies
below the floor as well as above it.
SCOPE. Five tasks of thirty-eight, rank 1, one partner each, all adjective-versus-adjective.
Smoke: V559_SMOKE=<out.json> (CPU, V559_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_adjprep_census_five_v559_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "adjective_preposition_"
PAIRS = {"at_about": "for_of", "in_to": "of_against", "over_for": "in_about",
         "of_amid": "in_with", "with_for": "of_amid", "of_at": "in_beneath"}   # five new + the v511 anchor
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
ANCHOR_KEY = "of_at"
PRIOR = {"v511_anchor": 0.682, "v555_nounprep": "1 of 5", "v557_particle": "1 of 5"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_adjprep_census_five_v559", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
            "pred_c_reached_are_inert": bool(c), "pred_d_rate_low": bool(d),
            "pred_e_anchor_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V559_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V559_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_adjprep_census_five_v559",
              "candidate_id": "corpus.unit_adjprep_census_five_v559", "bars": BARS,
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
