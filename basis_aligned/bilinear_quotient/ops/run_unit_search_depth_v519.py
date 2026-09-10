#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v519: do these behaviours share nodes at ANY search depth, or is the machinery genuinely different?

WHAT v517 GOT WRONG, MINE. It doubled the unit CAP to ask whether the unreachable half of the corpus was a size
limit. The cap was never binding: both fits chose exactly the same units at 30 and at 60 (15 and 18), because
greedy stops on its own criterion long before any cap. Across 67 fitted groups in today's receipts only three ever
reached 30, so the cap I register in every docstring has been inert. The knob that actually sets node-set size is
the STOPPING CRITERION -- target and min_gain -- and that is what this rung varies.
HOW THIS IS REGISTERED, BECAUSE THE HAZARD IS OBVIOUS. This is NOT a re-reading of v511's verdicts and does not
revise them: those stand at the budget they were registered under, and a control that only becomes reachable by
searching deeper is not the control they were denied. The question here is mechanistic and stands on its own -- ARE
THE SIBLING'S MEDIATING NODES SIMPLY FURTHER DOWN THE RANKING, OR ABSENT FROM IT? A search that keeps adding nodes
and never picks up the sibling says these two behaviours are carried by different machinery, which is a fact about
the model rather than about my budget.
Two constructions, three depths each, with pool 80 and cap 60 throughout so the CRITERION binds rather than the cap:
    std      target 0.97   min_gain 0.001     the setting every earlier rung used
    deep     target 0.995  min_gain 0.0002
    deepest  target 0.999  min_gain 0.00005
    verb_particle          up_down -> out_away          (0.677 at std)
    adjective_preposition  of_at   -> in_beneath        (0.682 at std)
Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_sets_grow          at the DEEPEST setting both fits choose at least 1.5x as many units as at std. This is
                            the v517 guard applied to the right knob: without it, a null below is unreadable because
                            the manipulation may not have bitten.                                     prior 80%
  pred_b_particle_grows     the verb_particle sibling's units_recovery at the deepest setting exceeds its std value
                            by at least gain = 0.05.                                                  prior 55%
  pred_c_adjective_grows    the adjective_preposition sibling's does likewise.                        prior 55%
  pred_d_either_crosses     at least one sibling reaches unit_min = 0.80 at SOME depth. Separate from b and c
                            because reach can grow without crossing, and those are different facts.   prior 40%
  pred_e_fits_hold          every fit reaches MINIMUM held-out joint extraction joint_min = 0.80 on its own rows.
                            Deeper searches can overfit the fitted task; if this fails at depth, the deeper numbers
                            are not comparable to the std ones.                                       prior 75%
HOW IT READS, one outcome per branch. a TRUE with b, c and d TRUE: the sibling's nodes are simply deeper in the
ranking, the unexamined half is examinable at a stated search cost, and I would report that cost. a TRUE with b and
c FALSE: the search took the extra nodes and still never picked up the sibling, so these behaviours are carried by
genuinely different machinery -- the stronger and more useful outcome, and the one that turns "unexamined" into a
structural fact about this model rather than a limitation of my settings. a TRUE, b and c TRUE, d FALSE: reach
improves but never crosses, so depth helps and does not solve it, and the honest report is a curve rather than a
verdict. a FALSE: the criterion did not bite either, and I will not interpret b, c or d -- exactly as in v517.
SCOPE. Two constructions, one task each, rank 1. Nothing here revises a registered verdict.
Smoke: V519_SMOKE=<out.json> (CPU, V519_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_search_depth_v519_result.json"
# the same two fits at the standard unit cap and at double it
# pool and cap held generous throughout so the STOPPING CRITERION is what binds
BUDGETS = {"std": (0.97, 0.001), "deep": (0.995, 0.0002), "deepest": (0.999, 0.00005)}
BASE = {"verb_particle": ("verb_particle_up_down", "verb_particle_out_away"),
        "adjective_preposition": ("adjective_preposition_of_at", "adjective_preposition_in_beneath")}
MAPS = {f"{fam}__{b}": {"fit": (pair[0],), "held": (pair[1],), "tgt": t, "mg": m}
        for fam, pair in BASE.items() for b, (t, m) in BUDGETS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v511_particle": 0.677, "v511_adjective": 0.682, "v517_units": (15, 18)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_search_depth_v519", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def sib(fam, b):
        return G.get(f"{fam}__{b}", {}).get("per_shape", {}).get(BASE[fam][1], {}).get("units_recovery")

    def fit_ext(fam, b):
        return G.get(f"{fam}__{b}", {}).get("per_shape", {}).get(BASE[fam][0], {}).get("joint_extraction")

    def nu(fam, b):
        return G.get(f"{fam}__{b}", {}).get("n_units")

    fams = list(BASE)
    reach = {(f, b): sib(f, b) for f in fams for b in BUDGETS}
    exts = [fit_ext(f, b) for f in fams for b in BUDGETS]
    counts = {(f, b): nu(f, b) for f in fams for b in BUDGETS}
    have = ok and all(x is not None for x in exts) \
        and all(v is not None for v in reach.values()) and all(v is not None for v in counts.values())
    a = have and all(counts[(f, "deepest")] >= 1.5 * counts[(f, "std")] for f in fams)
    b = have and (reach[("verb_particle", "deepest")] - reach[("verb_particle", "std")]) >= B["gain"]
    c = have and (reach[("adjective_preposition", "deepest")]
                  - reach[("adjective_preposition", "std")]) >= B["gain"]
    d = have and any(reach[(f, bb)] >= B["unit_min"] for f in fams for bb in BUDGETS)
    e = have and min(exts) >= B["joint_min"]
    return {"pred_a_sets_grow": bool(a), "pred_b_particle_grows": bool(b),
            "pred_c_adjective_grows": bool(c), "pred_d_either_crosses": bool(d),
            "pred_e_fits_hold": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V519_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V519_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            tgt, mg = (0.97, 0.001)          # relaxed budget; only the unit CAP varies here
            if not smoke:
                pool, max_units = 80, 60
                tgt, mg = MAPS[gname]["tgt"], MAPS[gname]["mg"]
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
    result = {"predictions": predictions, "schema": "unit_search_depth_v519",
              "candidate_id": "corpus.unit_search_depth_v519", "bars": BARS,
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
