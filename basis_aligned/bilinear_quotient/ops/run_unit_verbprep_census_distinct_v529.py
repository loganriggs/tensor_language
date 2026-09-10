#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v529: the census again, with FIVE DISTINCT partners -- the version v527's own receipt said was the honest one.

WHAT v527 SHOWED AND WHAT IT DID NOT. Five tasks in the corpus's largest family each had a vocabulary-disjoint
sibling their own nodes reached: 0.924, 0.859, 0.932, 1.009, 0.911, all inert, reachability symmetric on the one
pair run both ways. But those five fits used only TWO distinct partners -- about_for, by_from and from_for all
evaluated at_to, while at_to and in_to evaluated about_for. So the count established reachability against two
partner tasks, not against the family, and I flagged that in the receipt rather than letting the number carry it.
THIS IS THAT VERSION. Five fits, five DISTINCT partners, every pair sharing no answer token:
    at_to      -> about_for      {at,to}     vs {about,for}
    by_from    -> at_over        {by,from}   vs {at,over}
    in_to      -> by_with        {in,to}     vs {by,with}
    from_for   -> at_into        {from,for}  vs {at,into}
    about_into -> by_from        {about,into} vs {by,from}
Same bars as v527 so the two rates are directly comparable, and at_to -> about_for is carried over as the anchor so
a drift in the shared fit is visible. Standard budget; rank 1; nothing counted -- this measures control availability.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold      all five fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80. Instrument
                        check.                                                                        prior 85%
  pred_b_majority_reachable at least THREE of the five siblings report units_recovery >= unit_min = 0.80. Same bar
                        as v527, kept deliberately so the counts sit on one scale.                    prior 70%
  pred_c_reached_are_inert EVERY sibling clearing the units floor has ABSOLUTE recovery at or below cross_max = 0.30
                        -- quantified over the reached ones, vacuously false if none is.              prior 80%
  pred_d_rate_matches_v527 at least FOUR of five are reachable, i.e. the rate does not drop materially once the
                        partners are distinct. Worked example: 5 of 5 or 4 of 5, TRUE and v527's count was about
                        the family rather than about its two partners; 3 of 5, FALSE and partner choice was doing
                        part of the work in v527, which I would then say plainly.                     prior 60%
  pred_e_anchor_reproduces at_to -> about_for lands within tol_pool = 0.05 of 0.924. Deterministic fit; a
                        cross-receipt consistency check, not a question.                              prior 90%
HOW IT READS. d is the point. TRUE means control availability in this family is a property of the family and the
quality ceiling here really is authoring time, which is the claim v527 could not support on two partners. FALSE
means reachability depends on which partner is chosen, every four-hypothesis claim in this family needs its own
control check rather than inheriting one, and v527's five-of-five was partly an artefact of at_to and about_for
being unusually central. Either way the honest unit is a rate over distinct partners, which is what this rung
finally measures.
SCOPE. Five tasks of 119 and five partners of 119, chosen for disjoint vocabularies rather than sampled at random.
Smoke: V529_SMOKE=<out.json> (CPU, V529_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_verbprep_census_distinct_v529_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
PAIRS = {"at_to": "about_for", "by_from": "at_over", "in_to": "by_with",
         "from_for": "at_into", "about_into": "by_from"}          # five DISTINCT partners
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"anchor": 0.924, "v527_rate": "5 of 5 on two distinct partners"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_verbprep_census_distinct_v529", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    d = have and sum(1 for k in ks if reach[k] >= B["unit_min"]) >= 4
    e = have and abs(reach["at_to"] - PRIOR["anchor"]) <= B["tol_pool"]
    return {"pred_a_fits_hold": bool(a), "pred_b_majority_reachable": bool(b),
            "pred_c_reached_are_inert": bool(c), "pred_d_rate_matches_v527": bool(d),
            "pred_e_anchor_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V529_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V529_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_verbprep_census_distinct_v529",
              "candidate_id": "corpus.unit_verbprep_census_distinct_v529", "bars": BARS,
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
