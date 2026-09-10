#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v483: the family's own suffix contains the token ` of`. Is a third of it riding an induction echo?

THE CHALLENGE. An adversarial audit of adjective_preposition reported that the shared suffix `, of course,`
CONTAINS the token ` of`, that the readout sits immediately after the closing comma, and that 12 of the family's 36
cells have ` of` in their answer vocabulary. A plain induction head therefore predicts ` of` at the readout for free
in a third of the family -- and it predicts the CORRECT arm there, so a direction fitted on those cells could be
modulating a copy signal rather than an adjective-to-preposition mapping. Six of the roughly eleven cells that ever
reached the board are of-cells, so this is not a corner of the family.
THE MANIPULATION HOLDS DISTANCE, WHICH IS WHY IT IS NOT THE OBVIOUS ONE. `, back then,` is four GPT-2 tokens,
exactly like `, of course,`; a three-token replacement such as `, naturally,` would have moved the cue-to-readout
distance from four to three, and v473 established in the sibling family that distance moves these directions on its
own. So the only thing changing here is the suffix's LEXIS, and specifically whether it contains the answer token.
Two fits, each on an ORIGINAL `, of course,` cell, each evaluating its of-free twin held out. The two cells are
byte-identical in frame and differ only in cue pair and vocabulary:
    fit_of   adjective_preposition_of_at  (proud -> ` of` / good -> ` at`)      -- ` of` IS in the vocabulary
    fit_ref  adjective_preposition_at_about (skilled -> ` at` / passionate -> ` about`) -- ` of` is NOT
    control  verb_preposition_at_to, a different construction entirely, on ABSOLUTE recovery
All four cells were capability-screened on CPU before this runner was written (A1 32/32 rows kept, 0 dropped, each).
Reported and not leaned on: the of-cell's margins fall slightly when ` of` leaves the suffix (+/-2.4 to +/-2.1) while
the reference cell's do not (+/-3.5 to +/-3.8). That is capability, not causality, and the deltas are small.
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_both_in_distribution  held-out rows of BOTH fitted cells reach MINIMUM joint extraction joint_min = 0.80.
                               If this fails nothing else is readable.                                prior 85%
  pred_b_of_cell_survives      the of-fitted direction reaches 0.80 on its of-free twin.              prior 55%
  pred_c_ref_cell_survives     the reference-fitted direction reaches 0.80 on ITS of-free twin. This is the matched
                               comparison: without it, a drop in the of-cell could be the suffix change rather than
                               the echo.                                                              prior 70%
  pred_d_of_drop_not_larger    the reference cell's transfer MINUS the of-cell's is at most gain = 0.05, i.e. the
                               of-cell does not degrade MORE than the matched non-of cell under the identical
                               manipulation. Worked example: 0.95 and 0.93 gives 0.02, TRUE and there is no echo
                               effect; 0.95 and 0.60 gives 0.35, FALSE and part of the of-cell's direction was the
                               echo.                                                                  prior 55%
  pred_e_units_available       every evaluated cell except the controls reports units_recovery >= unit_min = 0.80,
                               so any shortfall is the DIRECTION and not units that miss the shape.    prior 75%
HOW IT READS, one outcome per branch. b, c and d TRUE: the ` of` in the suffix is not doing hidden work and the
of-cells' directions stand as measured. d FALSE: the of-cell degrades more than its matched twin under the same
change, part of its direction was the echo, and all 12 of-cells in this family need re-measuring under an of-free
suffix before anything counted on them is trusted. b FALSE and c FALSE with d TRUE: BOTH cells lose the same amount,
so the suffix's lexis matters here in a way it did not in verb_preposition (v473: changing the parenthetical's class
at constant distance cost nothing, 0.843), which is a family difference and gets its own row rather than being
folded into the echo question.
SCOPE. Two cells of one family at rank 1 and this budget; the echo hypothesis is about the other ten of-cells too,
and a result here licenses re-measuring them, not concluding about them.
Smoke: V483_SMOKE=<out.json> (CPU, V483_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_of_echo_v483_result.json"
# two fits, matched frames: one cell whose answer token appears in the shared suffix, one whose does not
AP = "adjective_preposition_"
OF_CELL, OF_TWIN = "of_at", "nf_of_at"
REF_CELL, REF_TWIN = "at_about", "nf_at_about"
CTRL = "verb_preposition_at_to"
MAPS = {"fit_of": {"fit": (OF_CELL,), "held": (OF_TWIN,)},
        "fit_ref": {"fit": (REF_CELL,), "held": (REF_TWIN,)}}
GROUPS = {k: tuple(AP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(AP + c for c in v["held"]) + (CTRL,) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v473_class_change": 0.843}          # sibling family: parenthetical class at constant distance cost nothing
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_of_echo_v483", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(AP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(AP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery")
    of_own, ref_own = je("fit_of", OF_CELL), je("fit_ref", REF_CELL)
    of_tw, ref_tw = je("fit_of", OF_TWIN), je("fit_ref", REF_TWIN)
    have = ok and all(x is not None for x in (of_own, ref_own, of_tw, ref_tw))
    a = have and min(of_own, ref_own) >= B["joint_min"]
    b = have and of_tw >= B["joint_min"]
    c = have and ref_tw >= B["joint_min"]
    d = have and (ref_tw - of_tw) <= B["gain"]
    e = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    return {"pred_a_both_in_distribution": bool(a), "pred_b_of_cell_survives": bool(b),
            "pred_c_ref_cell_survives": bool(c), "pred_d_of_drop_not_larger": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V483_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V483_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_of_echo_v483",
              "candidate_id": "corpus.unit_of_echo_v483", "bars": BARS,
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
