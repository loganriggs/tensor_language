#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v569: a partner SEARCH for verb_particle's two near-misses -- and a correction to what v565 registered.

THE CORRECTION FIRST. v565 came out e TRUE: away_up at reach 0.772 and out_down at 0.779 have C effects of 0.018
and 0.004, indistinguishable from down_in, which cleared the floor at 0.815 and passed all four. I registered the
licensed follow-up as "re-fit those two at the relaxed budget to raise their reach". THAT FOLLOW-UP IS A NO-OP. Both
v557 and v565 already set RELAXED = "ALL" and fit every pool at tgt, mg = 0.97, 0.001, so the relaxed budget is what
produced 0.772 and 0.779 in the first place. There is no budget knob left: v517 showed the unit cap never binds, and
raising the rank is forbidden -- a near-miss is not permission to raise rank. So the registered follow-up was wrong,
and this rung replaces it rather than quietly running something else under the same heading.
WHAT ACTUALLY FOLLOWS. Reach is a property of the target-partner PAIR, not of the target alone, and v557 tested
exactly ONE partner per target. The real question is whether these two targets have a partner they do reach. So:
away_up, whose readouts are away/up, is evaluated against three untried disjoint partners drawn from down/in/on/out
-- down_in, out_down, down_out_c; out_down, whose readouts are down/out, is evaluated against three drawn from
away/in/on/up -- on_away, up_on, away_up_b. Reach is DIRECTIONAL, so away_up -> out_down is a new measurement even
though v557 measured out_down -> away_up.
WHAT THIS CANNOT DO. It cannot rescue the 0.772 and 0.779 readings: those pairs stay unspendable whatever happens
here. It can only find a DIFFERENT pair that clears the floor honestly.
RANK FIXED AT 1 AND REGISTERED. Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold     both targets hold joint extraction at or above joint_min = 0.80, so any low reach is about
                       the partner and not a failed fit.                                              prior 90%
  pred_b_away_up_found away_up reaches units_recovery >= unit_min = 0.80 on AT LEAST ONE of its three untried
                       partners.                                                                      prior 45%
  pred_c_out_down_found out_down reaches 0.80 on at least one of its three untried partners.          prior 45%
  pred_d_reached_are_inert every partner that reaches also has cross_abs_recovery within cross_max = 0.3, so a
                       reach is not bought by the direction simply reproducing the partner. IT IS VACUOUSLY TRUE
                       WHEN NOTHING REACHES, and carries no information in that case -- read it only alongside a
                       true b or c.                                                                   prior 85%
  pred_e_old_partners_reproduce the two partners v557 already measured -- away_up -> on_out and out_down -> away_up
                       -- come back within tol_pool = 0.05 of 0.772 and 0.779. Deterministic fits, so a drift means
                       this rung is not measuring what v557 measured.                                 prior 88%
HOW IT READS, one outcome per branch. b and c both true: each near-miss target has a partner it genuinely reaches,
the two tasks become spendable through a DIFFERENT pair, and the lesson is that one partner per target undersamples.
Both false: these two targets have no reachable partner among the disjoint particle cells, the 0.772/0.779 readings
are the best available, and verb_particle really is a family where most partners sit out of reach -- which is what
v557 concluded from one partner each and would now rest on four. One true and one false: reach is a pair property
and neither target nor family alone predicts it, which is the most useful of the three for choosing what to author.
SCOPE. Two targets, three candidate partners each, rank 1; a search, not an estimate of a family rate.
Smoke: V569_SMOKE=<out.json> (CPU, V569_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_particle_partner_search_v569_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_particle_"
SEARCH = {"away_up": ("down_in", "out_down", "down_out_c", "on_out"),
          "out_down": ("on_away", "up_on", "away_up_b", "away_up")}   # last entry each is the v557 partner
PAIRS = {k: v[0] for k, v in SEARCH.items()}          # kept so shared helpers still resolve a primary partner
V569_OLD = {"away_up": ("on_out", 0.772), "out_down": ("away_up", 0.779)}
MAPS = {k: {"fit": (VP + k,), "held": tuple(VP + q for q in v)} for k, v in SEARCH.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v557_old": V569_OLD, "v565_down_in": 0.815, "note": "relaxed budget was ALREADY in use in v557 and v565"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_particle_partner_search_v569", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    ks = list(SEARCH)
    fit_v = {k: fld(k, VP + k, "joint_extraction") for k in ks}
    reach = {k: {q: fld(k, VP + q, "units_recovery") for q in SEARCH[k]} for k in ks}
    absr = {k: {q: fld(k, VP + q, "cross_abs_recovery") for q in SEARCH[k]} for k in ks}
    have = ok and all(v is not None for v in fit_v.values()) \
        and all(x is not None for k in ks for x in reach[k].values())
    a = have and min(fit_v.values()) >= B["joint_min"]

    def found(k):
        new_q = [q for q in SEARCH[k] if q != V569_OLD[k][0]]
        return have and any(reach[k][q] >= B["unit_min"] for q in new_q)

    b = found("away_up")
    c = found("out_down")
    reached = [(k, q) for k in ks for q in SEARCH[k]
               if have and reach[k][q] >= B["unit_min"] and absr[k][q] is not None]
    d = have and all(abs(absr[k][q]) <= B["cross_max"] for k, q in reached)   # vacuously true if none reached
    e = have and all(abs(reach[k][V569_OLD[k][0]] - V569_OLD[k][1]) <= B["tol_pool"] for k in ks)
    R["per_partner"] = {k: {q: {"reach": reach[k][q], "leak": absr[k][q],
                                "reaches": bool(reach[k][q] is not None and reach[k][q] >= B["unit_min"]),
                                "v557_partner": q == V569_OLD[k][0]}
                            for q in SEARCH[k]} for k in ks}
    R["fits"] = fit_v
    return {"pred_a_fits_hold": bool(a), "pred_b_away_up_found": bool(b),
            "pred_c_out_down_found": bool(c), "pred_d_reached_are_inert": bool(d),
            "pred_e_old_partners_reproduce": bool(e)}


def main() -> None:
    effective_budget = {"target": TARGET, "min_gain": MIN_GAIN}   # recipe must record what was USED, not the constant
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V569_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V569_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            tgt, mg = (0.97, 0.001)          # every pool in this rung uses the relaxed budget
            effective_budget["target"], effective_budget["min_gain"] = tgt, mg
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
    result = {"predictions": predictions, "per_partner": R.get("per_partner"), "fits": R.get("fits"),
              "schema": "unit_particle_partner_search_v569",
              "candidate_id": "corpus.unit_particle_partner_search_v569", "bars": BARS,
              "recipe": {"pool": pool, "target": effective_budget["target"],
                         "min_gain": effective_budget["min_gain"],
                         "target_module_constant": TARGET, "min_gain_module_constant": MIN_GAIN,
                         "max_units": max_units,
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
