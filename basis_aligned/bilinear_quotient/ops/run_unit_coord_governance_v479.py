#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v479: is the direction lexeme PRESENCE or GOVERNANCE? Identical token multisets decide it.

THE GAP THIS CLOSES. In every one of the 119 verb_preposition cells the cue is the only verb in the window and is
always BOTH the nearest verb and the governing one, so "which verb lexeme is present nearby" and "which verb
selects the PP" are perfectly confounded -- an adversarial audit named this the family's second-ranked confound, and
this project's own note that these circuits are RECENCY readers is a governance-adjacent claim that has never been
tested against presence.
The new cell holds the token MULTISET fixed and permutes it: `... appealed and aimed, of course,` obliges ` at` and
`... aimed and appealed, of course,` obliges ` to`. A direction encoding which lexemes are present has NOTHING to
separate on these rows and must sit at complement level. Base and donor differ in TWO positions rather than the
family's usual one, because a permutation is the only way to hold the multiset fixed; that is the design, not an
oversight, and it is registered here.
The CPU screen already answered the behavioural half: 32/32 rows kept on both A1 and A2, 0 dropped, so the MODEL
tracks which verb governs rather than which lexemes occur. Margins are weaker than this family's usual (+/-1.2 and
+/-0.9 against +/-2 to +/-5), which is reported rather than argued from. What the DIRECTION encodes is the open
question and this rung asks it in both directions, as v477 did, so that neither is assumed:
    fit_single  fit on at_to + fb fd fj fm (one verb, nearest and governing); evaluate the coord cell held out
    fit_coord   fit on the coord cell alone; evaluate all five single-verb cells held out
    control     about_for, a different mapping entirely, on ABSOLUTE recovery, in both groups
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_single_in_distribution  held-out rows of the fitted single-verb cells reach MINIMUM joint extraction
                                 joint_min = 0.80. If this fails nothing else is readable.            prior 85%
  pred_b_single_to_coord         the single-verb-fitted direction reaches 0.80 on the coord cell.     prior 45%
  pred_c_coord_in_distribution   the coord-fitted direction reaches 0.80 on its own held-out rows. THIS IS THE
                                 LOAD-BEARING ONE: a lexeme-presence axis cannot separate identical multisets at
                                 all, so if this passes, a direction that encodes GOVERNANCE exists here.  prior 65%
  pred_d_coord_to_single         the coord-fitted direction reaches 0.80 on the MINIMUM of the five single-verb
                                 cells.                                                               prior 50%
  pred_e_units_available         every evaluated cell except the controls reports units_recovery >= unit_min = 0.80,
                                 so a transfer failure is the DIRECTION and not units that miss the shape. prior 70%
HOW IT READS, fixed in advance. c TRUE establishes that a governance direction exists in this model at rank 1. c TRUE
with b and d TRUE says the family's ORDINARY direction is that same object -- presence is ruled out as its content,
and "recency reader" earns the governance reading it has been carrying without one. c TRUE with b or d FALSE says a
governance direction exists but is NOT what the 119 cells have been measuring, which would be the more consequential
outcome and would put the burden on every cell in the family. c FALSE says nothing here separates the multisets at
rank 1, and then b and d are unreadable whatever they show -- I will not interpret them in that case.
SCOPE. One mapping, one construction, rank 1, this budget.
Smoke: V479_SMOKE=<out.json> (CPU, V479_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_coord_governance_v479_result.json"
# two fits: the family's usual one-verb window, and a coordinated window with an identical token multiset
VPP = "verb_preposition_"
SINGLE = ("at_to", "fb_at_to", "fd_at_to", "fj_at_to", "fm_at_to")
COORD = "coord_at_to"
CTRL = "about_for"
MAPS = {"fit_single": {"fit": SINGLE, "held": (COORD,), "control": CTRL},
        "fit_coord": {"fit": (COORD,), "held": SINGLE, "control": CTRL}}
GROUPS = {k: tuple(VPP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(VPP + c for c in v["held"]) + (VPP + v["control"],) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {}                                     # every number compared here is measured in this run
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_coord_governance_v479", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(VPP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(VPP + MAPS[k]["control"], {}).get("cross_abs_recovery")
    single_fit = [je("fit_single", c) for c in SINGLE]
    coord_from_single = je("fit_single", COORD)
    coord_own = je("fit_coord", COORD)
    single_from_coord = [je("fit_coord", c) for c in SINGLE]
    have = ok and all(x is not None for x in single_fit + single_from_coord + [coord_from_single, coord_own])
    a = have and min(single_fit) >= B["joint_min"]
    b = have and coord_from_single >= B["joint_min"]
    c = have and coord_own >= B["joint_min"]
    d = have and min(single_from_coord) >= B["joint_min"]
    e = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    return {"pred_a_single_in_distribution": bool(a), "pred_b_single_to_coord": bool(b),
            "pred_c_coord_in_distribution": bool(c), "pred_d_coord_to_single": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V479_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V479_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell in cells:                       # only this mapping's own frames enter the fit
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
            for cell in EVAL[gname]:
                held_rows, c_rows = per_shape_rows[cell]
                p_held = g.prepare(backend, held_rows, **V)
                p_c = g.prepare(backend, c_rows)
                e_exact = ext(p_held, units)
                cdmg = dmg(p_c, units, q_joint, mu_joint)
                per_shape[cell] = {
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
    result = {"predictions": predictions, "schema": "unit_coord_governance_v479",
              "candidate_id": "corpus.unit_coord_governance_v479", "bars": BARS,
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
