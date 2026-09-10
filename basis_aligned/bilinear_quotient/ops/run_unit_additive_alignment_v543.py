#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v543: price the misalignment on the SECOND-worst counted case, so one measurement does not stand for nine.

WHAT v541 SETTLED AND WHAT I REFUSED TO CONCLUDE FROM IT. The corpus sweep found 102 of 465 tasks with a
clean/corrupted token-length mismatch and NINE counted. v541 priced the worst, reciprocal at 32 of 32 rows, and found
the misalignment cosmetic: own extractions 0.995 and 0.984, mutual transfer 0.996 and 0.932, unit overlap 0.812 with
the repaired set contained in the original. I registered in that receipt that "at most as affected" for the other
eight was an INFERENCE from one measurement, not a result. This is the second measurement.
additive_scope is the second-worst counted case at 28 of 32 rows. Its cause is simple: the cue ` not only` is two
GPT-2 tokens while most of the adverbs opposite it are one, so the four rows that happen to use a two-token adverb
are already matched. The repair keeps the design and draws the adverb only from two-token options -- warmly, bravely,
sternly, coolly, deftly, primly, flatly -- giving 0 of 32 mismatched, a CLEANER repair than reciprocal's, which left
one row. It screens at 32/32 with margins +/-2.0 and +/-2.4, lower than this corpus's usual because the shared object
lexicon puts an odd noun after `wrote`; that is reported, not hidden, and pred_a is what guards against reading a
weak fit.
Both tasks fitted, each evaluated on the other, exactly as v541. Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_both_fits_hold  BOTH fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80 on their own
                         rows. The margins here are lower than usual, so this guard matters more than it did in
                         v541.                                                                        prior 75%
  pred_b_orig_to_repaired the direction fitted on the ORIGINAL misaligned task reaches unit_min = 0.80 on the
                         repaired one.                                                                prior 65%
  pred_c_repaired_to_orig the direction fitted on the REPAIRED task reaches 0.80 on the original.      prior 65%
  pred_d_unit_overlap    the two node sets share Jaccard overlap of at least jac_min = 0.50.          prior 60%
  pred_e_extractions_agree the two tasks' own held-out extractions agree within tol_pool = 0.05.      prior 65%
HOW IT READS, one outcome per branch. All true: a SECOND counted case shows the misalignment costs nothing
measurable, and "the remaining seven are milder" stops being a bare inference and becomes an extrapolation from two
measurements at the top of the severity range -- still an extrapolation, and I will say so. b or c FALSE: the
aligned and misaligned versions give directions that do not transfer, so this counted result needs re-reading AND
v541's pass does not generalise, which would be the more useful outcome because it would tell me severity is not the
right axis. d FALSE with b and c TRUE: same direction, different nodes, reported as the overlap rather than a verdict.
SCOPE. One task pair, rank 1. Two of nine counted cases will have been priced; seven will not.
Smoke: V543_SMOKE=<out.json> (CPU, V543_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_additive_alignment_v543_result.json"
# the misaligned original and its length-matched repair, each fitted and each evaluated on the other
MAPS = {"orig": {"fit": ("additive_scope",), "held": ("additive_scope_lenmatched",)},
        "repaired": {"fit": ("additive_scope_lenmatched",), "held": ("additive_scope",)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
FORWARD = {}
PRIOR = {"orig_mismatch_rows": 28, "repaired_mismatch_rows": 0, "v541_overlap": 0.812}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"jac_min": 0.5, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_additive_alignment_v543", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    own = {k: fld(k, MAPS[k]["fit"][0], "joint_extraction") for k in GROUPS}
    cross = {k: fld(k, MAPS[k]["held"][0], "units_recovery") for k in GROUPS}
    us = {k: set(G.get(k, {}).get("units") or []) for k in GROUPS}
    have = ok and all(v is not None for v in own.values()) and all(v is not None for v in cross.values())
    a = have and min(own.values()) >= B["joint_min"]
    b = have and cross["orig"] >= B["unit_min"]
    c = have and cross["repaired"] >= B["unit_min"]
    both = us["orig"] | us["repaired"]
    d = bool(both) and (len(us["orig"] & us["repaired"]) / len(both)) >= B["jac_min"]
    e = have and abs(own["orig"] - own["repaired"]) <= B["tol_pool"]
    return {"pred_a_both_fits_hold": bool(a), "pred_b_orig_to_repaired": bool(b),
            "pred_c_repaired_to_orig": bool(c), "pred_d_unit_overlap": bool(d),
            "pred_e_extractions_agree": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V543_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V543_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "units": [str(u) for u in units]}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_additive_alignment_v543",
              "candidate_id": "corpus.unit_additive_alignment_v543", "bars": BARS,
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
