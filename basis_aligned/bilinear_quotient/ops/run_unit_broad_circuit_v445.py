#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v445: does a direction fail because of HOW MANY READOUT TOKENS it must serve?

WHAT IS ESTABLISHED AND WHAT IS UNEXPLAINED. Two constructions, one instrument, one budget, rank 1 throughout:
    verb_particle      3 cells 1.028   5 cells 0.929   6 cells 0.935   7 cells 0.821
    verb_preposition   2 cells 1.009   3 cells 1.001   4 cells 0.980   5 cells 0.858   6 cells 0.772   7 cells 0.718
Same baseline, steeper fall for prepositions -- v443 measured the small end precisely to rule out the alternative that
they are intrinsically more distributed. What makes the slope differ is unexplained, and I have declined twice tonight
to invent an account at this point in a result.
ONE CANDIDATE IS ALREADY DEAD, at zero cost, from the cell definitions: the particle pools SHARE readout tokens far
more heavily than the preposition pools -- ` up`, ` down` and ` out` repeat across the five particle cells while the
five preposition cells span eight distinct tokens -- and the particles pool BETTER. A token-COLLISION story predicts
the opposite of what was measured.
That leaves its converse, which is testable with cells already on disk: the number of DISTINCT readout tokens one
direction must serve. Two five-cell preposition pools, chosen by exhaustive search over the 24 separable frame-A
members to minimise and maximise that count:
    low_diversity   about_against, about_into, from_about, into_with, with_against   -- FIVE distinct tokens
    high_diversity  about_against, at_over, by_from, for_toward, in_onto             -- TEN distinct tokens
Same construction, same size, same budget, same rank, one shared member (about_against) so a difference cannot come
from one pool happening to hold a strong cell. v441's arbitrary five-cell pool spanned eight tokens and gave mean
0.858, min 0.795, which sits between the two pools this rung compares.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_low_covers      the LOW-diversity pool's MINIMUM joint extraction reaches joint_min = 0.80, where the
                         arbitrary eight-token pool of the same size and construction gave 0.795.        prior 60%
  pred_b_high_resists    the HIGH-diversity pool's MINIMUM falls BELOW 0.80. Registered as the direct alternative so
                         the rung cannot be read as confirming whichever way it lands.                   prior 55%
  pred_c_gap             the low pool's MEAN exceeds the high pool's by at least gain = 0.05. Worked example: 0.93
                         against 0.84 gives 0.09, TRUE and token diversity is the variable; 0.87 against 0.85 gives
                         0.02, FALSE and it is not, which would send me back to the slope with one fewer candidate
                         and no story.                                                                   prior 50%
  pred_d_units_available both pools reach minimum pooled-unit recovery >= unit_min = 0.80, so a difference is the
                         direction and not the selection -- the premise v437 had to establish and every rung since
                         has carried.                                                                    prior 85%
  pred_e_control_clean   own-C damage UB975 <= 0.01 on every cell in both pools.                         prior 60%
COUNTING. Nothing here is counted; all nine cells already count separately. What is at stake is whether the slope has
a name or stays an observation.
Smoke: V445_SMOKE=<out.json> (CPU, V445_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v445_result.json"
LOW = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_from_about",
       "verb_preposition_into_with", "verb_preposition_with_against")        # 5 distinct readout tokens
HIGH = ("verb_preposition_about_against", "verb_preposition_at_over", "verb_preposition_by_from",
        "verb_preposition_for_toward", "verb_preposition_in_onto")           # 10 distinct readout tokens
GROUPS = {"low_diversity": LOW, "high_diversity": HIGH}
RELAXED = "ALL"
PREP_SIZE5_MIXED = (0.858, 0.795)     # the arbitrary five-cell preposition pool from v441, 8 distinct tokens
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v445", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values()) and set(G) == set(GROUPS)

    def stats(k):
        g = G.get(k, {})
        if not g or "error" in g:
            return None, None, None
        v = [s["joint_extraction"] for s in g["per_shape"].values() if s.get("joint_extraction") is not None]
        u = [s["units_recovery"] for s in g["per_shape"].values() if s.get("units_recovery") is not None]
        return (sum(v) / len(v) if v else None, min(v) if v else None, min(u) if u else None)

    ml, nl, ul = stats("low_diversity")
    mh, nh, uh = stats("high_diversity")
    a = ok and nl is not None and nl >= B["joint_min"]
    b = ok and nh is not None and nh < B["joint_min"]
    c = ok and ml is not None and mh is not None and (ml - mh) >= B["gain"]
    d = ok and all(u is not None and u >= B["unit_min"] for u in (ul, uh))
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_low_covers": bool(a), "pred_b_high_resists": bool(b), "pred_c_gap": bool(c),
            "pred_d_units_available": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V445_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V445_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in cells:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
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
            for cell in cells:
                held_rows, c_rows = per_shape_rows[cell]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cell] = {
                    "units_recovery": e_exact,
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v445",
              "candidate_id": "corpus.unit_broad_circuit_v445", "bars": BARS,
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
