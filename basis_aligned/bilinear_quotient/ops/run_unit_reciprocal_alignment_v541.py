#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v541: what does the token-length misalignment actually COST? Measure it on the worst counted case.

THE FINDING THIS PRICES. A corpus-wide sweep found 102 of 465 tasks with a clean/corrupted token-length mismatch and
NINE of them counted. reciprocal is the most severe: 32 of 32 rows, and by CONSTRUCTION -- its cue is number, so the
plural side spends an extra token (`The two leaders` against `The leader`). The patched position is the final token
and is aligned; every position before it is shifted by one, on every row. Position alignment is standard practice
and an older lane here already carries the `equal_token_length` flag the fast-screen builder dropped, so this is a
regression against a known requirement rather than a discovery -- but nobody has measured what it costs.
THE REPAIR AND ITS RESIDUAL, STATED HONESTLY. reciprocal_lenmatched gives the singular side ` lone`, which costs
exactly the token the plural side spends on ` two`. That takes the mismatch from 32 of 32 rows to ONE of 32 -- the
single agent noun in the shared lexicon whose plural is not a single token -- not to zero, and I am not calling it a
clean repair. It screens at 32/32 rows with margins +/-3.4 and +/-3.9.
Both tasks are fitted and each is evaluated against the other, so the question is symmetric: do the misaligned and
aligned versions yield the SAME circuit? Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_both_fits_hold  BOTH fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80 on their own
                         rows. If either does not fit, its cross number says nothing.                 prior 85%
  pred_b_orig_to_repaired the direction fitted on the ORIGINAL misaligned task reaches unit_min = 0.80 on the
                         repaired one.                                                                prior 65%
  pred_c_repaired_to_orig the direction fitted on the REPAIRED task reaches 0.80 on the original.      prior 65%
  pred_d_unit_overlap    the two fits' node sets share Jaccard overlap of at least jac_min = 0.50, i.e. the search
                         lands on the same machinery either way.                                      prior 60%
  pred_e_extractions_agree the two tasks' own held-out extractions agree within tol_pool = 0.05, so any difference
                         above is about transfer rather than about one task being easier.             prior 70%
HOW IT READS, one outcome per branch. All true: the misalignment costs nothing measurable for this circuit -- the
same nodes, the same direction, mutual transfer -- so the defect is real as practice and COSMETIC here, and
reciprocal's counted result stands. b or c FALSE: the aligned and misaligned versions give directions that do not
transfer, so the counted result rests on a task whose positions were shifted and it needs re-reading -- which would
extend to the other eight counted mismatched tasks in proportion to their severity. d FALSE with b and c TRUE: the
directions transfer but the searches land on different nodes, which is a weaker version of the same worry and I
would report the overlap rather than a verdict.
SCOPE. One task pair, rank 1. It prices the worst counted case; the other eight are milder (28 of 32 down to 1 of 32)
and are not tested here.
Smoke: V541_SMOKE=<out.json> (CPU, V541_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_reciprocal_alignment_v541_result.json"
# the misaligned original and its length-matched repair, each fitted and each evaluated on the other
MAPS = {"orig": {"fit": ("reciprocal",), "held": ("reciprocal_lenmatched",)},
        "repaired": {"fit": ("reciprocal_lenmatched",), "held": ("reciprocal",)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
FORWARD = {}
PRIOR = {"orig_mismatch_rows": 32, "repaired_mismatch_rows": 1}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"jac_min": 0.5, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_reciprocal_alignment_v541", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    smoke = os.environ.get("V541_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V541_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_reciprocal_alignment_v541",
              "candidate_id": "corpus.unit_reciprocal_alignment_v541", "bars": BARS,
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
