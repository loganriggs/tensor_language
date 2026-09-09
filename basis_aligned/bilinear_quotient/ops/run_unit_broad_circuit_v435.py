#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v435: the SIZE SERIES -- how many behaviours can one rank-1 direction carry before coverage breaks?

WHAT IS LEFT AFTER TODAY'S WITHDRAWAL. v433 refuted my own claim that pooled coverage degrades with VARIABLE
diversity: inside verb_particle at matched size, a three-cell pool of different variables covered at mean 1.028 and a
three-cell pool of one variable at 0.987, the gap running the wrong way. What survived is narrower and is the only
regularity left across four pools -- three particle cells give 0.99 to 1.03, FIVE particle cells give 0.829, and the
one cell that has ever fallen below 0.80 in fifteen pooled rungs was in that five-cell pool. So the candidate variable
is POOL SIZE, and nothing has varied it deliberately.
This rung does, with NESTED pools so that size is the only thing that changes:
    size3  up_down, out_down, away_up
    size5  + down_in, out_up
    size7  + up_down_b, away_up_b
    size9  + out_down_b, away_up_c
Every larger pool CONTAINS the smaller one, so a difference between them cannot come from which cells were drawn.
All nine are separable verb_particle members, all in frame A, so construction, frame, position and P control are held
fixed across the whole series. Four fits, about three minutes.
WHY IT IS WORTH A RUNG. Every merge rule in the corpus -- R7's twelve frame groups, the two R5 group clauses, and the
determiner and possessive results tonight -- rests on a pooled direction covering a set of behaviours. If coverage
falls off with the SIZE of the set, then every one of those rules is safe only at the sizes actually tested (two to
five members), and the corpus should say so rather than assume the property scales. If coverage does not fall off,
that is the strongest statement today's instrument can make and it makes the merges safe to extend.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_every_pool_covers  the MINIMUM joint extraction in every pool is >= joint_min = 0.80. If size does not bite
                            at nine, it does not bite in any pool this corpus has built.                 prior 40%
  pred_b_size9_resists      the nine-cell pool's minimum falls BELOW 0.80. Registered as the direct alternative to
                            pred_a so the rung cannot be read as confirming whichever way it lands.     prior 55%
  pred_c_monotone_drop      the three-cell pool's MEAN exceeds the nine-cell pool's by at least drop = 0.05. This is
                            the size effect as a magnitude rather than a threshold. Worked example: means of 1.01 and
                            0.93 give 0.08, TRUE; 1.01 and 0.99 give 0.02, FALSE and size is not the variable either,
                            which would leave v431's 0.829 unexplained and worth a rung of its own.     prior 50%
  pred_d_units_transfer     every cell in every pool reaches exact-set recovery >= unit_min = 0.80 under its pool's
                            greedy set, so a coverage difference cannot be blamed on the units.          prior 45%
  pred_e_control_clean      own-C damage UB975 <= 0.01 on every cell in every pool. This failed at v429 (0.0193) and
                            v431 (0.0144) and passed at v433 (0.0044, 0.0072); if it fails only in the larger pools
                            here, control damage is a second thing that scales with size.               prior 35%
COUNTING. Nothing here is counted; all nine already count separately. What is at stake is the SCOPE of every merge
rule the corpus applies.
Smoke: V435_SMOKE=<out.json> (CPU, V435_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v435_result.json"
BASE = ("verb_particle_up_down", "verb_particle_out_down", "verb_particle_away_up",
        "verb_particle_down_in", "verb_particle_out_up", "verb_particle_up_down_b",
        "verb_particle_away_up_b", "verb_particle_out_down_b", "verb_particle_away_up_c")
GROUPS = {"size3": BASE[:3], "size5": BASE[:5], "size7": BASE[:7], "size9": BASE[:9]}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "drop": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v435", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
            return None, None
        v = [s["joint_extraction"] for s in g["per_shape"].values() if s.get("joint_extraction") is not None]
        return (sum(v) / len(v), min(v)) if v else (None, None)

    means = {k: stats(k)[0] for k in GROUPS}
    mins = {k: stats(k)[1] for k in GROUPS}
    a = ok and all(m is not None and m >= B["joint_min"] for m in mins.values())
    b = ok and mins.get("size9") is not None and mins["size9"] < B["joint_min"]
    have = ok and all(means[k] is not None for k in ("size3", "size9"))
    c = have and (means["size3"] - means["size9"]) >= B["drop"]
    d = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    e = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"]
                   for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_every_pool_covers": bool(a), "pred_b_size9_resists": bool(b), "pred_c_monotone_drop": bool(c),
            "pred_d_units_transfer": bool(d), "pred_e_control_clean": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V435_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V435_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=TARGET,
                                                     min_gain=MIN_GAIN, max_units=max_units)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v435",
              "candidate_id": "corpus.unit_broad_circuit_v435", "bars": BARS,
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
