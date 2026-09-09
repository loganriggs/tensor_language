#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v437: is the size limit in the SITE SELECTION or in the rank-1 DIRECTION over those sites?

WHAT v435 LEFT OPEN. Nested pools of three, five, seven and nine verb_particle cells gave mean joint extraction
1.028, 0.829, 0.777 and 0.756 -- monotone, a drop of 0.272 -- and pooled-unit recovery fell alongside it from 0.830
to 0.751. Two quantities fell together, so the finding "pooled coverage is size-limited" does not yet say WHERE the
limit lives:
    SELECTION -- the greedy set is the bottleneck. It stopped at FOURTEEN units in the size-nine pool, and not because
                 of the thirty-unit cap: it stopped because it reached its target of 0.88 with a min_gain of 0.005.
                 Give it a higher target and a smaller minimum gain and it may find the sites the ninth cell needs.
    DIRECTION -- the sites are there and one rank-1 direction cannot serve nine behaviours at once, in which case a
                 larger unit set buys nothing.
This rung runs the SAME nine-cell pool twice. size9_standard uses the corpus recipe (target 0.88, min_gain 0.005) and
exists to reproduce v435. size9_relaxed raises the target to 0.97 and drops min_gain to 0.001, so the selection may
spend up to the thirty-unit cap. Everything else -- cells, rank, steps, control weight, rows -- is identical.
RANK STAYS AT 1. Relaxing the UNIT budget is not raising the rank, and the standing protocol's rule that a null is
not permission to raise rank is untouched: this asks whether more SITES help, which is the other axis.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_more_units         the relaxed greedy selects strictly more than V435_UNITS = 14 units. If it does not, the
                            relaxation did nothing and the rest of the rung is uninterpretable -- this is the
                            manipulation check.                                                          prior 85%
  pred_b_units_recover      with the larger set, the MINIMUM pooled-unit recovery across the nine cells reaches
                            unit_min = 0.80, where v435 measured 0.751.                                  prior 55%
  pred_c_coverage_gains     the relaxed pool's MEAN joint extraction exceeds v435's 0.756 by at least gain = 0.05.
                            Worked example: 0.83 gives a gain of 0.074, TRUE; 0.78 gives 0.024, FALSE and the extra
                            sites bought nothing, which points at the direction rather than the selection.
                                                                                                          prior 45%
  pred_d_every_cell_covered the relaxed pool's MINIMUM joint extraction reaches joint_min = 0.80, i.e. the size-nine
                            merge would actually be supportable. This is stricter than pred_c and is the outcome that
                            would change what the corpus may merge.                                      prior 30%
  pred_e_standard_reproduces  the standard pool reproduces v435's mean of 0.756 within tol = 0.03. Without this the
                            comparison has no baseline; it is the same instrument check that made v421 and v423
                            readable.                                                                    prior 85%
COUNTING. Nothing here is counted. What is at stake is whether the scope limit I recorded an hour ago -- no merge
above five members without its own receipt -- is a limit of the METHOD (fixable with more sites) or of the model.
Smoke: V437_SMOKE=<out.json> (CPU, V437_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v437_result.json"
BASE = ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_away_up",
        "verb_particle_down_in", "verb_particle_out_up", "verb_particle_up_down_b",
        "verb_particle_away_up_b", "verb_particle_out_down_b", "verb_particle_away_up_c")
GROUPS = {"size9_standard": BASE, "size9_relaxed": BASE}
RELAXED = "size9_relaxed"                 # this pool gets the larger unit budget
V435_MEAN, V435_MIN, V435_UNITS = 0.756, 0.671, 14
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v437", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
            return None, None, None, None
        v = [s["joint_extraction"] for s in g["per_shape"].values() if s.get("joint_extraction") is not None]
        u = [s["units_recovery"] for s in g["per_shape"].values() if s.get("units_recovery") is not None]
        c = [s["C_ub975"] for s in g["per_shape"].values() if s.get("C_ub975") is not None]
        return (sum(v) / len(v) if v else None, min(v) if v else None,
                min(u) if u else None, max(c) if c else None)

    std_mean, std_min, std_u, _std_c = stats("size9_standard")
    rel_mean, rel_min, rel_u, rel_c = stats(RELAXED)
    n_rel = G.get(RELAXED, {}).get("n_units")
    a = ok and n_rel is not None and n_rel > V435_UNITS
    b = ok and rel_u is not None and rel_u >= B["unit_min"]
    c = ok and rel_mean is not None and (rel_mean - V435_MEAN) >= B["gain"]
    d = ok and rel_min is not None and rel_min >= B["joint_min"]
    e = ok and std_mean is not None and abs(std_mean - V435_MEAN) <= B["tol"]
    return {"pred_a_more_units": bool(a), "pred_b_units_recover": bool(b), "pred_c_coverage_gains": bool(c),
            "pred_d_every_cell_covered": bool(d), "pred_e_standard_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V437_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V437_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            tgt, mg = (0.97, 0.001) if gname == RELAXED else (TARGET, MIN_GAIN)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v437",
              "candidate_id": "corpus.unit_broad_circuit_v437", "bars": BARS,
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
