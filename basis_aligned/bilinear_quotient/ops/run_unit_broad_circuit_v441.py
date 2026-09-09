#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v441: is SIX a property of verb_particle, or of this model? The same relaxed series in the preposition class.

WHAT v439 ESTABLISHED, AND ITS ONE UNTESTED ASSUMPTION. Running five, six and seven cell pools at the relaxed greedy
budget -- with minimum pooled-unit recovery 0.921 to 0.929, so every site was demonstrably reachable -- gave five
cells mean 0.929 min 0.893, SIX cells mean 0.935 min 0.898, and SEVEN cells mean 0.821 min 0.777. One rank-1
direction carries up to six behaviours and fails at seven. That number is now the scope condition on every merge rule
the corpus applies, and it was measured in ONE construction.
This rung repeats it exactly in the preposition class, which has 23 separable frame-A members -- more than enough for
a nested five/six/seven series:
    size5  about_against, about_into, at_over, at_to, by_from
    size6  + for_at
    size7  + from_about
Same relaxed budget (target 0.97, min_gain 0.001, thirty-unit cap), same rank 1, same bars, nested so size is the only
thing that varies. The particle numbers are carried in the runner as PARTICLE for direct comparison.
WHY IT MATTERS. If the preposition boundary is also six, "six behaviours per rank-1 direction" is a fact about this
model and the corpus can state one scope condition for all merges. If it is four, or eight, the condition has to be
stated per family, and every merge rule needs its own boundary measurement rather than inheriting one -- which is a
materially larger obligation and worth knowing before anyone leans on the number.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_size5_covers    the five-cell pool's MINIMUM joint extraction reaches joint_min = 0.80.         prior 70%
  pred_b_size7_covers    the seven-cell pool's MINIMUM reaches 0.80. In verb_particle this FAILED at 0.777, and it is
                         the clause that decides whether the boundary sits in the same place. Worked example: a
                         minimum of 0.85 puts the preposition boundary above seven and makes "six" particle-specific;
                         0.74 reproduces the particle result.                                            prior 40%
  pred_c_units_available every pool's minimum pooled-unit recovery reaches unit_min = 0.80. Without this the rung
                         measures selection rather than the direction, which is the confound v437 had to remove.
                                                                                                        prior 80%
  pred_d_matches_particle  the seven-cell minimum lands within tol = 0.03 of verb_particle's 0.777 -- the strong form
                         of "same boundary", requiring the NUMBER to agree and not merely the verdict. It can fail
                         while pred_b also fails, which would mean both constructions break at seven but by different
                         amounts, and that is a third outcome worth distinguishing.                      prior 30%
  pred_e_control_clean   own-C damage UB975 <= 0.01 on every cell in every pool.                         prior 45%
COUNTING. Nothing here is counted; all seven cells already count separately. What is at stake is whether the scope
condition on merges is one number or one per family.
Smoke: V441_SMOKE=<out.json> (CPU, V441_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v441_result.json"
BASE = ("verb_preposition_about_against", "verb_preposition_about_into", "verb_preposition_at_over",
        "verb_preposition_at_to", "verb_preposition_by_from", "verb_preposition_for_at",
        "verb_preposition_from_about")
GROUPS = {"size5_relaxed": BASE[:5], "size6_relaxed": BASE[:6], "size7_relaxed": BASE[:7]}
RELAXED = "ALL"
PARTICLE = {"size5_relaxed": (0.929, 0.893), "size6_relaxed": (0.935, 0.898), "size7_relaxed": (0.821, 0.777)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v441", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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

    m5, n5, u5 = stats("size5_relaxed")
    m6, n6, u6 = stats("size6_relaxed")
    m7, n7, u7 = stats("size7_relaxed")
    a = ok and n5 is not None and n5 >= B["joint_min"]
    b = ok and n7 is not None and n7 >= B["joint_min"]
    c = ok and all(u is not None and u >= B["unit_min"] for u in (u5, u6, u7))
    d = ok and n7 is not None and abs(n7 - PARTICLE["size7_relaxed"][1]) <= B["tol"]
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_size5_covers": bool(a), "pred_b_size7_covers": bool(b), "pred_c_units_available": bool(c),
            "pred_d_matches_particle": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V441_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V441_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v441",
              "candidate_id": "corpus.unit_broad_circuit_v441", "bars": BARS,
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
