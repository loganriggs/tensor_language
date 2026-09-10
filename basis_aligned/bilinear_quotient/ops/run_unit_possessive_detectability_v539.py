#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v539: calibrate detectability for the POSSESSIVE direction, and see whether row 4 opens for the second DAS target.

WHAT IS BLOCKED AND WHY. possessive_number.adjacent_antecedent is the second named DAS target and has sat at three
of four hypotheses all day: A1 0.988, A2 1.016, P 0.0814, and row 4 UNESTABLISHABLE because no control was both
vocabulary-disjoint and reached. v507 found all five cross-construction candidates between 0.322 and 0.474; v509
authored a same-construction disjoint cell -- person instead of number, ` my` against ` your` -- which reached 0.729,
better than everything else but under the 0.80 floor, so I booked it as a failure and refused to read its inert
effect.
WHAT CHANGED, AND ITS LIMIT. v537 showed for the ABOUT_FOR direction that the floor is not a detectability gate:
a positive case at reach 0.694 still registered an effect of 0.5433 against a 0.35 bar, so a near-zero reading at
that reach is meaningful. I wrote into that receipt that this calibrates detectability for THAT direction only and
the floor stays where it is for others until the same check is run on them. THIS IS THAT CHECK, run rather than
assumed, and it is the one thing that could open row 4 for this target.
The possessive family supplies four same-mapping positives -- same cue pair, same readout pair, different frame --
which v475 already showed the direction transfers to at 0.838 and 0.804:
    POSITIVE  possessive_attractor, possessive_argument, possessive_long_simple,
              possessive_number_attractor_conflict
    NEGATIVE  possessive_disjoint_my_your   (reach 0.729 in v509; ` my`/` your` shares no token with ` their`/` his`)
Rank 1; nothing counted -- this calibrates an instrument and reads one control.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fit_holds       the fitted task reaches MINIMUM held-out joint extraction joint_min = 0.80.   prior 85%
  pred_b_positives_move  ALL four same-mapping positives show same-answer effect at or above pos_min = 0.35.
                                                                                                      prior 70%
  pred_c_low_reach_calibrated at least one positive with reach at or below low_reach = 0.75 shows an effect at or
                         above 0.35. THIS IS THE CLAUSE THAT LICENSES ANYTHING: it can fail two ways -- no positive
                         lands that low, or one does and fails to move -- and either way the 0.729 control stays
                         unlicensed for this direction and v509's verdict stands untouched.           prior 55%
  pred_d_disjoint_inert  the disjoint control's same-answer effect is at or below p_max = 0.23.       prior 75%
  pred_e_reach_reproduces the disjoint control's reach lands within tol_pool = 0.05 of v509's 0.729.
                         Deterministic fit; cross-receipt consistency.                                prior 85%
HOW IT READS, one outcome per branch. b, c and d TRUE: the possessive instrument registers movement at low reach, so
the 0.729 control is informative for THIS direction, its inert reading counts, and row 4 is established -- which
would complete the four-hypothesis protocol on the second named DAS target and take it from three of four to four of
four. c FALSE: the calibration is absent and 0.729 stays unlicensed here; v509's verdict stands and I will say so
without reaching for the about_for calibration, which is about a different direction. d FALSE: the direction moves a
disjoint same-construction behaviour, which would be a genuine specificity failure and more informative than a pass.
SCOPE. One fitted task, five partners, rank 1. Detectability is being calibrated per direction, which is the whole
point of running this rather than importing v537's number.
Smoke: V539_SMOKE=<out.json> (CPU, V539_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_detectability_v539_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = ""                                        # these task names carry no shared prefix
FIT = "possessive_adjacent"
POSITIVE = ("possessive_attractor", "possessive_argument", "possessive_long_simple",
            "possessive_number_attractor_conflict")   # same mapping, different frame -- should MOVE
NEGATIVE = ("possessive_disjoint_my_your",)     # disjoint vocabulary -- should not
PARTNERS = POSITIVE + NEGATIVE
MAPS = {"about_for": {"fit": (VP + FIT,), "held": tuple(VP + p for p in PARTNERS)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"my_your_reach": 0.729, "v475_transfer": (0.838, 0.804)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"low_reach": 0.75, "pos_min": 0.35, "range_min": 0.20, "gain_range": 0.20, "p_max": 0.23, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_detectability_v539", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("about_for", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL["about_for"])
    ps = g0.get("per_shape", {}) if ok else {}
    eff = g0.get("effects", {}) if ok else {}
    fit_v = ps.get(VP + FIT, {}).get("joint_extraction")
    reach = {p: ps.get(VP + p, {}).get("units_recovery") for p in PARTNERS}
    have = ok and fit_v is not None and all(v is not None for v in reach.values()) \
        and all(eff.get(p) is not None for p in PARTNERS)
    a = have and fit_v >= B["joint_min"]
    b = have and all(abs(eff[p]) >= B["pos_min"] for p in POSITIVE)
    c = have and any(reach[p] <= B["low_reach"] and abs(eff[p]) >= B["pos_min"] for p in POSITIVE)
    neg = NEGATIVE[0]
    d = have and abs(eff[neg]) <= B["p_max"]
    e = have and abs(reach[neg] - PRIOR["my_your_reach"]) <= B["tol_pool"]
    return {"pred_a_fit_holds": bool(a), "pred_b_positives_move": bool(b),
            "pred_c_low_reach_calibrated": bool(c), "pred_d_disjoint_inert": bool(d),
            "pred_e_reach_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V539_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V539_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            p_a1 = g.prepare(backend, per_shape_rows[VP + FIT][0], **V)
            scale = g.target_scale(p_a1)
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            effects = {p: eff(g.prepare(backend, per_shape_rows[VP + p][0], **V)) for p in PARTNERS}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "scale": round(scale, 4), "effects": effects}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_possessive_detectability_v539",
              "candidate_id": "corpus.unit_possessive_detectability_v539", "bars": BARS,
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
