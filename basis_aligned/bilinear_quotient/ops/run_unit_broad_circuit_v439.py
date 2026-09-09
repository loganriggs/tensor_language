#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v439: WHERE exactly does a single direction stop covering? Five, six and seven cells at the relaxed budget.

WHAT IS BRACKETED AND WHAT IS NOT. v435 measured the standard-budget size series at 1.028, 0.829, 0.777 and 0.756 for
three, five, seven and nine cells. v437 then showed that at NINE the shortfall is part selection and part direction:
relaxing the greedy budget took the set from 14 to 23 units and repaired the SITES (pooled-unit recovery 0.751-0.983
becomes 0.923-1.024), lifting the mean from 0.756 to 0.822, but the weakest cell stayed at 0.744. So at nine, with
every site reachable, one rank-1 direction still cannot cover the pool.
That leaves the boundary bracketed between FIVE, where the corpus's actual merges live and where coverage has held
(the possessive five at 0.995-1.073, the determiner five at 0.965-1.011), and NINE, where it does not. This rung
locates it. Three pools -- five, six and seven cells, nested, drawn from the same nine verb_particle members -- each
fitted with the RELAXED budget that v437 showed makes the sites available, so any failure here is the direction and
not the selection.
    size5_relaxed  up_down, out_down, away_up, down_in, out_up
    size6_relaxed  + up_down_b
    size7_relaxed  + away_up_b
RANK STAYS AT 1 throughout. This rung varies pool size with the unit budget held generous, which is the one axis left
after v437 separated the two.
WHY THE ANSWER MATTERS BEYOND THIS FAMILY. Every merge rule the corpus applies asserts that one direction covers a
set: R7's twelve frame groups are two to four members, the correlative clause is three, the possessive and determiner
confirmations are five. A boundary at six or seven would mean the rules are comfortably inside the safe range and a
future six-member group needs a receipt; a boundary at five would mean the possessive and determiner fives are AT the
edge and should be re-read with that in mind.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_size5_covers    the five-cell pool's MINIMUM joint extraction reaches joint_min = 0.80. At the standard
                         budget v435 measured its minimum at 0.795 -- four thousandths under the bar -- so the
                         relaxed budget deciding this one way or the other is itself informative.        prior 70%
  pred_b_size7_covers    the seven-cell pool's MINIMUM reaches 0.80, where the standard budget gave 0.665. Worked
                         example: a minimum of 0.83 makes seven safe and moves the boundary to eight or nine; 0.71
                         leaves it between five and seven.                                               prior 40%
  pred_c_units_available every pool's minimum pooled-unit recovery reaches unit_min = 0.80, confirming that the
                         relaxed budget did for these pools what it did at nine. Without this the rung measures
                         selection again rather than the direction.                                      prior 80%
  pred_d_monotone        the means fall or hold across five, six and seven: mean5 >= mean6 >= mean7. The standard
                         series was monotone; if the relaxed one is not, the effect is noisier than the four points
                         at v435 suggested and the boundary talk is premature.                           prior 65%
  pred_e_control_clean   own-C damage UB975 <= 0.01 on every cell in every pool. It failed at v429, v431 and v435 and
                         passed at v433 and v437 (0.0079), so it is genuinely uncertain and is on the record either
                         way.                                                                            prior 45%
COUNTING. Nothing here is counted; all seven cells already count separately. What is at stake is the size at which a
merge stops being supportable, which is the scope condition on every merge rule in the corpus.
Smoke: V439_SMOKE=<out.json> (CPU, V439_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v439_result.json"
BASE = ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_away_up",
        "verb_particle_down_in", "verb_particle_out_up", "verb_particle_up_down_b",
        "verb_particle_away_up_b", "verb_particle_out_down_b", "verb_particle_away_up_c")
GROUPS = {"size5_relaxed": BASE[:5], "size6_relaxed": BASE[:6], "size7_relaxed": BASE[:7]}
RELAXED = "ALL"                            # every pool here gets the larger unit budget
V435 = {"size5_relaxed": 0.829, "size6_relaxed": None, "size7_relaxed": 0.777}   # standard-budget means
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol": 0.03}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v439", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    d = ok and all(m is not None for m in (m5, m6, m7)) and m5 >= m6 >= m7
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_size5_covers": bool(a), "pred_b_size7_covers": bool(b), "pred_c_units_available": bool(c),
            "pred_d_monotone": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V439_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V439_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v439",
              "candidate_id": "corpus.unit_broad_circuit_v439", "bars": BARS,
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
