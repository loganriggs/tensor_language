#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v443: do prepositions START lower, or FALL faster? Completing the curve at two, three and four cells.

THE CONTRAST THIS FINISHES. At the identical relaxed budget, with sites available in every pool, the two
constructions gave:
    verb_particle      3 cells 1.028/1.000    5 cells 0.929/0.893   6 cells 0.935/0.898   7 cells 0.821/0.777
    verb_preposition                          5 cells 0.858/0.795   6 cells 0.772/0.670   7 cells 0.718/0.576
(mean/min). Prepositions break earlier and harder, on MORE units (28 against 22-24), so it is not site scarcity. But
the preposition curve has no small end: it starts at five, where it is already under the bar, so two readings fit
every number measured so far.
    LOWER BASELINE -- prepositions cover less at EVERY size, including two and three, and the "boundary" is just where
                      a curve that was always lower crosses 0.80. Nothing special happens at five.
    STEEPER FALL   -- prepositions cover as well as particles in small pools and degrade faster with each added
                      behaviour, which would make the difference about how the construction ACCUMULATES rather than
                      about its baseline.
Three pools -- two, three and four cells, nested, same five members the size-five pool starts from -- separate them.
Rank stays at 1; the budget stays relaxed; nothing else changes.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_small_pools_cover  every one of the three pools has MINIMUM joint extraction >= joint_min = 0.80. Under the
                            steeper-fall reading this holds comfortably; under the lower-baseline reading the
                            four-cell pool and possibly the three-cell one do not.                       prior 55%
  pred_b_lower_at_three     the three-cell preposition MEAN is at least gain = 0.05 BELOW verb_particle's 1.028 at
                            the same size. This is the lower-baseline claim stated as a number. Worked example: a
                            preposition three-cell mean of 0.95 gives a gap of 0.078, TRUE and the baseline differs;
                            0.99 gives 0.038, FALSE and the constructions start together.                prior 60%
  pred_c_units_available    every pool's minimum pooled-unit recovery reaches unit_min = 0.80, so a shortfall is the
                            direction and not the selection -- the same premise v437 had to establish.  prior 85%
  pred_d_falls_across_two_to_four  the two-cell MEAN exceeds the four-cell mean by at least 0.05, i.e. the decline is
                            already visible inside the small end rather than beginning at five.         prior 50%
  pred_e_control_clean      own-C damage UB975 <= 0.01 on every cell in every pool. It failed at seven cells in this
                            construction (0.0166) and passed at five and six (0.0026, 0.0046).          prior 65%
COUNTING. Nothing here is counted; all four cells already count separately. What is at stake is whether "prepositions
are harder to pool" is a fact about their baseline or about their accumulation, which is the difference between a
construction being intrinsically more distributed and one whose behaviours interfere more with each other.
Smoke: V443_SMOKE=<out.json> (CPU, V443_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v443_result.json"
BASE = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_at_over",
        "verb_preposition_at_to", "verb_preposition_by_from", "verb_preposition_for_at",
        "verb_preposition_from_about")
GROUPS = {"prep_size2": BASE[:2], "prep_size3": BASE[:3], "prep_size4": BASE[:4]}
RELAXED = "ALL"
PARTICLE = {"size3": (1.028, 1.000), "size5": (0.929, 0.893), "size6": (0.935, 0.898), "size7": (0.821, 0.777)}
PREP_KNOWN = {"size5": (0.858, 0.795), "size6": (0.772, 0.670), "size7": (0.718, 0.576)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v443", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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

    m2, n2, u2 = stats("prep_size2")
    m3, n3, u3 = stats("prep_size3")
    m4, n4, u4 = stats("prep_size4")
    a = ok and all(n is not None and n >= B["joint_min"] for n in (n2, n3, n4))
    b = ok and m3 is not None and (PARTICLE["size3"][0] - m3) >= B["gain"]
    c = ok and all(u is not None and u >= B["unit_min"] for u in (u2, u3, u4))
    d = ok and all(m is not None for m in (m2, m3, m4)) and (m2 - m4) >= B["gain"]
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_small_pools_cover": bool(a), "pred_b_lower_at_three": bool(b), "pred_c_units_available": bool(c),
            "pred_d_falls_across_two_to_four": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V443_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V443_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v443",
              "candidate_id": "corpus.unit_broad_circuit_v443", "bars": BARS,
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
