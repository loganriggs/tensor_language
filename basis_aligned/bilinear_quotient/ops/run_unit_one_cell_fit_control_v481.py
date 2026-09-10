#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v481: is v479's reverse-transfer asymmetry about GOVERNANCE, or just about fitting one cell?

WHAT v479 LEFT OPEN. The coordinated-verb cell established that a governance direction exists at rank 1 (its own
held-out rows at 0.953, where a lexeme-presence axis could not separate identical token multisets at all), and the
family's ordinary five-cell direction reached it at 1.157. But the reverse failed: the coord-fitted direction
reached only 0.542-0.651 on the five single-verb cells. I DID NOT READ THAT, because the coord fit is ONE cell
against five and six units against eight, so the asymmetry is confounded with fit size by construction.
THIS RUNG REMOVES THE CONFOUND WITHOUT TOUCHING GOVERNANCE AT ALL. Fit at_to ALONE -- one ordinary single-verb cell,
same construction, same mapping, no coordination anywhere -- and ask whether it transfers to its four frame
siblings. Those siblings are the same cells the five-cell fit covered at 0.973-1.026, so the only thing that has
changed is how many cells the fit saw.
    fit_one   at_to alone; evaluate fb, fd, fj, fm held out, plus the coord cell, plus the control
    control   about_for, a different mapping entirely, on ABSOLUTE recovery
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_own_cell_in_distribution  held-out rows of the FITTED cell reach MINIMUM joint extraction
                                   joint_min = 0.80. If this fails nothing else is readable.          prior 90%
  pred_b_one_cell_transfers        the one-cell fit reaches 0.80 on the MINIMUM of its four frame siblings -- the
                                   same cells a five-cell fit covered at 0.973-1.026.                 prior 50%
  pred_c_fit_size_explains         the MEAN sibling transfer under this one-cell fit is at or below
                                   one_cell_max = 0.75, i.e. a one-cell fit does NOT reach what the five-cell fit
                                   reached. Worked example: 0.62 is at or below 0.75, TRUE; 0.95 is not, FALSE.
                                                                                                      prior 45%
  pred_d_units_available           every evaluated cell except the control reports units_recovery >= unit_min = 0.80,
                                   so a shortfall is the DIRECTION and not units that miss the shape.  prior 70%
  pred_e_different_mapping         the control's ABSOLUTE recovery stays at or below cross_max = 0.30. prior 85%
HOW IT READS, fixed in advance, and each branch names ONE outcome rather than a disjunction -- v479's reading table
used a branch covering two different combinations and I could not use it when it fired, which is the error this
table is written to avoid. b FALSE with c TRUE: a one-cell fit does not reach siblings even with no coordination
involved, so v479's reverse asymmetry is FIT SIZE and carries no information about governance; v479's pred_d is
void as evidence and I will say so. b TRUE: a one-cell fit does transfer, so the coord fit's failure was about the
COORDINATION FRAME and v479's asymmetry is a real content difference worth naming. b FALSE with c FALSE: the
siblings sit between the two bars, the one-cell fit is degraded but not to the coord range, and fit size explains
PART of the asymmetry -- in which case neither v479's pred_d nor this rung licenses a governance claim, and the
matched comparison needs equal fit sizes on both sides instead.
SCOPE. One mapping, one construction, rank 1, this budget. This rung says nothing about governance directly; it only
decides whether v479's asymmetry is readable.
Smoke: V481_SMOKE=<out.json> (CPU, V481_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_one_cell_fit_control_v481_result.json"
# one ordinary cell fitted alone, to see whether fit SIZE alone explains a failure to reach siblings
VPP = "verb_preposition_"
SIBS = ("fb_at_to", "fd_at_to", "fj_at_to", "fm_at_to")
OWN = "at_to"
COORD = "coord_at_to"
CTRL = "about_for"
MAPS = {"fit_one": {"fit": (OWN,), "held": SIBS + (COORD,), "control": CTRL}}
GROUPS = {k: tuple(VPP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VPP + c for c in v["held"]) + (VPP + v["control"],) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"five_cell_to_sibs": (0.975, 0.973, 1.012, 1.026), "coord_to_single_mean": 0.607}   # v479
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3, "one_cell_max": 0.75}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_one_cell_fit_control_v481", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(VPP + MAPS[k]["control"], {}).get("cross_abs_recovery")
    own_v = je("fit_one", OWN)
    sib_v = [je("fit_one", c) for c in SIBS]
    have = ok and own_v is not None and all(x is not None for x in sib_v)
    a = have and own_v >= B["joint_min"]
    b = have and min(sib_v) >= B["joint_min"]
    c = have and (sum(sib_v) / len(sib_v)) <= B["one_cell_max"]
    d = ok and all(ur("fit_one", cc) is not None and ur("fit_one", cc) >= B["unit_min"]
                   for cc in (OWN,) + SIBS + (COORD,))
    cross = G.get("fit_one", {}).get("per_shape", {}).get(VPP + CTRL, {}).get("cross_abs_recovery")
    e = ok and cross is not None and abs(cross) <= B["cross_max"]
    return {"pred_a_own_cell_in_distribution": bool(a), "pred_b_one_cell_transfers": bool(b),
            "pred_c_fit_size_explains": bool(c), "pred_d_units_available": bool(d),
            "pred_e_different_mapping": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V481_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V481_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
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
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
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
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_one_cell_fit_control_v481",
              "candidate_id": "corpus.unit_one_cell_fit_control_v481", "bars": BARS,
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
