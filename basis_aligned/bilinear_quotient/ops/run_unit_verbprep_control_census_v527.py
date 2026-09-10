#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v527: how many tasks in the LARGEST family could carry a full four-hypothesis claim?

WHY COUNT THIS. v525 put verb_preposition_at_to through all four hypotheses at rank 1 and it passed cleanly, with a
control its own nodes reach at units 0.924. That is one task of 119. The controlling goal is hundreds of
high-quality circuits, and the binding constraint on QUALITY -- established across v499, v507 and v511 -- is not the
fit but the CONTROL: a specificity reading needs a vocabulary-disjoint task that the fitted nodes actually reach, and
only three constructions of six have one at all. Within the family that does, nobody has asked how OFTEN.
Five fits, each evaluating a sibling from the same family whose answer pair shares no token with the fitted task's:
    at_to      -> about_for      (known: units 0.924, v487 -- carried as the instrument check)
    about_for  -> at_to          (the REVERSE of that pair, and the reason it is here is v521: in
                                  adjective_preposition reachability was strongly ASYMMETRIC, 0.682 one way and
                                  0.976 the other, so symmetry cannot be assumed)
    by_from    -> at_to
    from_for   -> at_to
    in_to      -> about_for
Standard budget throughout, so every number is comparable to v511's and v525's. Rank 1; nothing counted -- this
rung measures control availability, not circuits.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold        all five fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80.
                          Instrument check; a fit that does not hold says nothing about its control. prior 85%
  pred_b_majority_reachable at least THREE of the five siblings report units_recovery >= unit_min = 0.80. This is
                          the headline count. Capable of failing: outside this family the rate was one in three
                          constructions, and within it only a single pair has ever been measured.     prior 60%
  pred_c_known_reproduces at_to -> about_for lands within tol_pool = 0.05 of v487's 0.924. Deterministic fit, so
                          this is a consistency check across receipts and not a question.             prior 90%
  pred_d_reached_are_inert EVERY sibling that clears the units floor has ABSOLUTE recovery at or below
                          cross_max = 0.30 -- quantified over the reached ones, vacuously false if none is reached,
                          the behaviour v499 and v507 established as correct.                         prior 80%
  pred_e_reverse_symmetric about_for -> at_to differs from at_to -> about_for by at most tol_pool_wide = 0.15.
                          Worked example: 0.92 against 0.924 is 0.004, TRUE and reachability is symmetric in this
                          family; 0.65 against 0.924 is 0.274, FALSE and it is directional, as it was in
                          adjective_preposition.                                                      prior 60%
HOW IT READS. b is a count and is reported as one: five of five means a four-hypothesis claim is available for
essentially any task here and the corpus's quality ceiling is set by authoring time rather than by controls; three
or four of five means it is common but must be checked per task; two or fewer means v525's clean pass was lucky in
its partner and the family is no better off than the others. e is separate because a family can be reachable on
average and still directional pair by pair, which would mean the control has to be chosen with the direction in
mind -- exactly the distinction v521 forced.
SCOPE. Five tasks of 119, one partner each, rank 1, standard budget.
Smoke: V527_SMOKE=<out.json> (CPU, V527_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_verbprep_control_census_v527_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
PAIRS = {"at_to": "about_for", "about_for": "at_to", "by_from": "at_to",
         "from_for": "at_to", "in_to": "about_for"}
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v487_at_to_to_about_for": 0.924, "v521_adjective_asymmetry": (0.682, 0.976)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_verbprep_control_census_v527", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    c = have and abs(reach["at_to"] - PRIOR["v487_at_to_to_about_for"]) <= B["tol_pool"]
    reached = [k for k in ks if have and reach[k] >= B["unit_min"]]
    d = bool(reached) and all(abs(absr[k]) <= B["cross_max"] for k in reached)
    e = have and abs(reach["about_for"] - reach["at_to"]) <= B["tol_pool_wide"]
    return {"pred_a_fits_hold": bool(a), "pred_b_majority_reachable": bool(b),
            "pred_c_known_reproduces": bool(c), "pred_d_reached_are_inert": bool(d),
            "pred_e_reverse_symmetric": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V527_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V527_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_verbprep_control_census_v527",
              "candidate_id": "corpus.unit_verbprep_control_census_v527", "bars": BARS,
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
