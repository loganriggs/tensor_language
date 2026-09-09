#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v381: BREADTH FIVE for three preposition entries -- widening existing entries instead of adding new ones.

WHY THIS SHAPE OF WORK. v375 showed every frame copy merges into its donor, so authoring copies adds nothing to the
count; v377 gave one entry a measured breadth of SEVEN shapes on a single direction; v379 showed that pooling even
covers shapes which FAILED their own standalone battery (both at 1.043, above the three that passed). Nine of the
corpus's merged entries currently sit at a breadth of 2 of 2 -- not because they stop there, but because that is where
I stopped adding shapes. This rung raises three of them to five.
    prep_at_to     aimed/appealed      A locative + B agreement + denial + temporal + interrogative
    prep_from_for  prevented/blamed    A locative + B agreement + denial + temporal + interrogative
    prep_by_from   swore/withdrew      A locative + C report    + denial + temporal + interrogative
The nine new cells have NO four-row battery and are not candidates to count: they exist to be pooled. Their per-shape
numbers below are a breadth measurement of an entry that already counts once, exactly as P_longparen was at v377.
All nine cleared the CPU capability floor with zero dropped rows (A1 1.92 to 6.12).
WHAT WOULD MAKE THIS FAIL, AND WHY THAT WOULD BE THE MORE USEFUL RESULT. Every pooled fit so far has covered every
shape it was given, so the honest position is that I have not yet found the edge of a circuit's breadth. Three
entries x five shapes, with an interrogative and a subordinate adjunct in each pool, is the widest spread attempted
in the preposition class. A shape that resists would be the first evidence of where one direction stops, and pred_e
is registered for exactly that.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_breadth_five      in all k_groups = 3 groups, the joint direction reaches extraction >= joint_min = 0.80 on
                           EVERY one of that group's five shapes. Worked example: a group scoring 0.99, 0.95, 0.88,
                           0.91, 0.84 counts; a group scoring 0.99, 0.95, 0.88, 0.91, 0.62 does not.    prior 60%
  pred_b_units_transfer    in at least k_units = 2 of the 3 groups, the pooled greedy unit set reaches exact-set
                           recovery >= unit_min = 0.80 on every shape. Set at two of three, not three, because v379
                           found one shape at 0.763 whose direction was still covered at 1.043 -- unit recovery and
                           direction coverage can come apart and I want the receipt to show which.       prior 55%
  pred_c_control_clean     in all three groups the joint direction's own-C damage UB975 stays <= 0.01 on every shape.
                                                                                                        prior 60%
  pred_d_no_error          every group returns a receipt; an error is recorded per group and fails this predicate
                           rather than being dropped.                                                    prior 90%
  pred_e_some_shape_resists  at least ONE shape anywhere in the three groups falls below 0.80. Registered because
                           nothing has resisted yet and a uniform pass should not be read as confirmation of a
                           ceiling I have never located. pred_a and pred_e cannot both hold; if both fail, some group
                           errored and pred_d says so.                                                   prior 45%
COUNTING. Nothing here is counted. Success raises three entries from breadth 2 to breadth 5 and adds zero circuits,
which is the trade the user asked for: the same objects, described more generally.
Smoke: V381_SMOKE=<out.json> (CPU, V381_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v381_result.json"
GROUPS = {
    "prep_at_to": ("verb_preposition_at_to", "verb_preposition_fb_at_to", "verb_preposition_fd_at_to",
                   "verb_preposition_fj_at_to", "verb_preposition_fm_at_to"),
    "prep_from_for": ("verb_preposition_from_for", "verb_preposition_fb_from_for", "verb_preposition_fd_from_for",
                      "verb_preposition_fj_from_for", "verb_preposition_fm_from_for"),
    "prep_by_from": ("verb_preposition_by_from", "verb_preposition_fc_by_from", "verb_preposition_fd_by_from",
                     "verb_preposition_fj_by_from", "verb_preposition_fm_by_from"),
}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "k_groups": 3, "k_units": 2}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v381", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and all("error" not in g for g in G.values())
    covers = [k for k, g in G.items() if "error" not in g
              and all(s.get("joint_extraction") is not None and s["joint_extraction"] >= B["joint_min"]
                      for s in g["per_shape"].values())]
    units_ok = [k for k, g in G.items() if "error" not in g
                and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"]
                        for s in g["per_shape"].values())]
    clean = [k for k, g in G.items() if "error" not in g
             and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in g["per_shape"].values())]
    resists = any(s.get("joint_extraction") is not None and s["joint_extraction"] < B["joint_min"]
                  for g in G.values() if "error" not in g for s in g["per_shape"].values())
    return {"pred_a_breadth_five": bool(ok and len(covers) >= B["k_groups"]),
            "pred_b_units_transfer": bool(ok and len(units_ok) >= B["k_units"]),
            "pred_c_control_clean": bool(ok and len(clean) >= B["k_groups"]),
            "pred_d_no_error": bool(ok),
            "pred_e_some_shape_resists": bool(ok and resists)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V381_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V381_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v381",
              "candidate_id": "corpus.unit_broad_circuit_v381", "bars": BARS,
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
