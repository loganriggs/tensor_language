#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v517: is the unreachable half a NODE-SET SIZE limit, or genuinely different nodes?

WHAT IS OPEN. v511 found that only three constructions of six have a node set that reaches a vocabulary-disjoint
task in the same construction. v515 then showed that where such a control IS reachable, swapping it for the
degenerate shared control changes no verdict -- the differences were 0.0040 to 0.0195 against a 0.05 gain. So the
specificity claims in that half survive. THE OTHER HALF IS UNEXAMINED RATHER THAN FINE: verb_particle at 0.677,
adjective_preposition at 0.682 and possessive at 0.729 have no control that is both disjoint and reached, so their
specificity has not been tested with an instrument capable of failing.
THIS RUNG ASKS WHY, AND IT IS NOT A REPAIR OF A FAILED BAR. The earlier verdicts stand at the budget they were
registered under (pool 40, target 0.97, min_gain 0.001, at most 30 units). The question here is different and is
worth asking on its own: are the sibling's mediating nodes simply OUTSIDE the top thirty, or are they genuinely
different nodes that no budget will include? Doubling the cap answers that, and both answers are useful -- one says
the unexamined half becomes examinable at a stated cost, the other says these behaviours are carried by disjoint
machinery and no control drawn from the same construction will ever serve.
Two constructions, each fitted twice on the SAME task with only the unit cap changed:
    verb_particle          up_down    -> out_away        standard 30 units, then 60
    adjective_preposition  of_at      -> in_beneath      standard 30 units, then 60
possessive is left out deliberately: v509 already showed its node set is specific to the NUMBER variable rather than
the construction, so a size question there is answering a different one.
Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_standard_reproduces at the STANDARD budget both siblings land within tol_pool = 0.05 of v511's 0.677 and
                             0.682. The fit is deterministic; if this drifts, the doubled-budget numbers are not
                             comparable to anything.                                                  prior 85%
  pred_b_particle_crosses    at the DOUBLED budget the verb_particle sibling reports units_recovery >= 0.80.
                                                                                                      prior 45%
  pred_c_adjective_crosses   at the DOUBLED budget the adjective_preposition sibling reports >= 0.80. prior 45%
  pred_d_budget_actually_grew the doubled runs chose at least 1.5x as many units as the standard runs, so a null
                             above means the extra capacity was taken and did not help, rather than the cap never
                             binding in the first place. Without this, b and c failing would be unreadable. prior 75%
  pred_e_fits_hold           all four fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80.
                                                                                                      prior 85%
HOW IT READS, one outcome per branch. b and c TRUE with d TRUE: the limit was node-set SIZE, the unexamined half is
examinable at roughly double the units, and I would report that cost rather than treat the earlier numbers as final.
b and c FALSE with d TRUE: the extra capacity was taken and did not help, so these behaviours are carried by
genuinely different nodes and no same-construction control will serve -- "unexamined" becomes a structural fact
about this corpus rather than a budget choice, which is the more consequential outcome. d FALSE: the cap was not
binding, the doubling changed nothing mechanically, and b and c say nothing either way -- I will not interpret them.
SCOPE. Two constructions, one task each, rank 1. Nothing here revises a verdict registered at the standard budget;
it only says what kind of limit that verdict ran into.
Smoke: V517_SMOKE=<out.json> (CPU, V517_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_budget_reach_v517_result.json"
# the same two fits at the standard unit cap and at double it
# the candidate POOL bounds the greedy search, so doubling the cap alone could never bind: raise both
BUDGETS = {"std": (40, 30), "big": (80, 60)}
BASE = {"verb_particle": ("verb_particle_up_down", "verb_particle_out_away"),
        "adjective_preposition": ("adjective_preposition_of_at", "adjective_preposition_in_beneath")}
MAPS = {f"{fam}__{b}": {"fit": (pair[0],), "held": (pair[1],), "pool": pl, "cap": cap}
        for fam, pair in BASE.items() for b, (pl, cap) in BUDGETS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v511_particle": 0.677, "v511_adjective": 0.682}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_budget_reach_v517", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def sib(fam, b, field):
        k = f"{fam}__{b}"
        return G.get(k, {}).get("per_shape", {}).get(BASE[fam][1], {}).get(field)

    def fit_ext(fam, b):
        k = f"{fam}__{b}"
        return G.get(k, {}).get("per_shape", {}).get(BASE[fam][0], {}).get("joint_extraction")

    def nu(fam, b):
        return G.get(f"{fam}__{b}", {}).get("n_units")

    fams = list(BASE)
    std_u = {f: sib(f, "std", "units_recovery") for f in fams}
    big_u = {f: sib(f, "big", "units_recovery") for f in fams}
    exts = [fit_ext(f, b) for f in fams for b in BUDGETS]
    counts = {(f, b): nu(f, b) for f in fams for b in BUDGETS}
    have = ok and all(x is not None for x in exts) \
        and all(v is not None for v in list(std_u.values()) + list(big_u.values())) \
        and all(v is not None for v in counts.values())
    a = have and abs(std_u["verb_particle"] - PRIOR["v511_particle"]) <= B["tol_pool"] \
        and abs(std_u["adjective_preposition"] - PRIOR["v511_adjective"]) <= B["tol_pool"]
    b = have and big_u["verb_particle"] >= B["unit_min"]
    c = have and big_u["adjective_preposition"] >= B["unit_min"]
    d = have and all(counts[(f, "big")] >= 1.5 * counts[(f, "std")] for f in fams)
    e = have and min(exts) >= B["joint_min"]
    return {"pred_a_standard_reproduces": bool(a), "pred_b_particle_crosses": bool(b),
            "pred_c_adjective_crosses": bool(c), "pred_d_budget_actually_grew": bool(d),
            "pred_e_fits_hold": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V517_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V517_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                pool, max_units = MAPS[gname]["pool"], MAPS[gname]["cap"]
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
    result = {"predictions": predictions, "schema": "unit_budget_reach_v517",
              "candidate_id": "corpus.unit_budget_reach_v517", "bars": BARS,
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
