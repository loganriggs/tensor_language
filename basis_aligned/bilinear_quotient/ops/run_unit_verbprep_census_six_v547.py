#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v547: refill the input. Six NEW tasks, six distinct partners, so the next battery has pairs to spend.

WHY A CENSUS IS THE PRODUCTIVE THING TO RUN. v545 put five verb_preposition targets through all four hypotheses in
156 seconds and every one passed. It was cheap because the expensive part -- establishing that each target has a
vocabulary-disjoint partner its own nodes REACH -- had already been paid in v487, v527 and v529. That is the
pipeline: census produces checked pairs, the battery spends them. v545 spent the five that existed, so the input is
empty and this rung refills it.
SIX NEW FITTED TASKS, SIX DISTINCT PARTNERS, every pair sharing no answer token -- distinct partners because v529
showed the rate drops when they are not (five of five on two partners became four of five on five) and v531 showed
why: availability is per-PAIR, since by_from reaches at_to at 0.932 and at_over at only 0.688.
    on_about     -> by_with      {on,about}     vs {by,with}
    into_with    -> at_to        {into,with}    vs {at,to}
    for_toward   -> at_into      {for,toward}   vs {at,into}
    with_against -> at_over      {with,against} vs {at,over}
    from_about   -> in_to        {from,about}   vs {in,to}
    of_into      -> by_from      {of,into}      vs {by,from}
None of the six fitted tasks has been fitted in any earlier rung, so this is a fresh sample rather than a re-read.
Standard budget; rank 1; nothing counted -- this measures control availability.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold        all six fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80. Instrument
                          check; a fit that does not hold says nothing about its partner.             prior 85%
  pred_b_majority_reachable at least THREE of six partners report units_recovery >= unit_min = 0.80.  prior 75%
  pred_c_reached_are_inert EVERY partner clearing the units floor has ABSOLUTE recovery at or below cross_max = 0.30,
                          quantified over the reached ones and vacuously false if none is.            prior 80%
  pred_d_rate_holds       at least FIVE of six are reachable, matching the four-of-five rate v529 measured on
                          distinct partners. Worked example: 5 or 6 of 6, TRUE and the rate is a property of the
                          family; 4 of 6, FALSE and the earlier rate was optimistic, which would mean the battery
                          input costs more per usable pair than I have been assuming.                 prior 55%
  pred_e_reached_strongly_inert every reached partner's absolute recovery is at or below strong_inert = 0.10, not
                          merely under 0.30. v545's five partners all came in between 0.004 and 0.014, so a value
                          between 0.10 and 0.30 would be a partner that is reachable but not cleanly inert -- usable
                          for a census and a poor choice for a battery.                               prior 70%
HOW IT READS. d TRUE with c and e TRUE: six fresh pairs are ready to spend and the pipeline sustains itself at
roughly one battery per census. d FALSE: the usable-pair rate is lower than v529 suggested and each battery target
costs more census than I have been budgeting -- worth knowing before I plan around it. e FALSE with c TRUE: some
partners are reachable but only marginally inert, and those should be censused as usable while being kept OUT of a
battery, which is a distinction the earlier rungs did not draw.
SCOPE. Six tasks of 119 and six partners, chosen for disjoint vocabularies rather than sampled at random.
Smoke: V547_SMOKE=<out.json> (CPU, V547_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_verbprep_census_six_v547_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
PAIRS = {"on_about": "by_with", "into_with": "at_to", "for_toward": "at_into",
         "with_against": "at_over", "from_about": "in_to", "of_into": "by_from"}   # six DISTINCT partners
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v529_rate": "4 of 5 on distinct partners", "v545_inertness": (0.004, 0.014)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_verbprep_census_six_v547", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    a = have and min(fit_v) >= B["joint_min"]
    b = have and sum(1 for k in ks if reach[k] >= B["unit_min"]) >= 3
    reached = [k for k in ks if have and reach[k] >= B["unit_min"]]
    c = bool(reached) and all(abs(absr[k]) <= B["cross_max"] for k in reached)
    d = have and sum(1 for k in ks if reach[k] >= B["unit_min"]) >= 5
    e = bool(reached) and all(abs(absr[k]) <= B["strong_inert"] for k in reached)
    return {"pred_a_fits_hold": bool(a), "pred_b_majority_reachable": bool(b),
            "pred_c_reached_are_inert": bool(c), "pred_d_rate_holds": bool(d),
            "pred_e_reached_strongly_inert": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V547_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V547_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_verbprep_census_six_v547",
              "candidate_id": "corpus.unit_verbprep_census_six_v547", "bars": BARS,
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
