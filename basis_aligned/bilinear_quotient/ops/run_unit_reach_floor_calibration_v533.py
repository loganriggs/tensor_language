#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v533: does the specificity estimate actually depend on how well the control is REACHED?

THE CAVEAT THIS TESTS, AND IT IS ONE I RAISED AGAINST MYSELF. Every rung since v499 has used a 0.80 floor on
units_recovery to decide whether a control is informative, on the reasoning that a near-zero reading from a control
the fitted nodes do not reach says nothing. v531 measured the same partner from four different fits and got 0.656,
0.688, 0.745 and 0.873 -- a CONTINUUM, with my floor cutting through the middle of it. So 0.873 against 0.745 is a
threshold crossing rather than a difference in kind, and I said so in the receipt. The question that follows is
whether the floor is doing real work: IF THE SPECIFICITY NUMBER IS THE SAME WHETHER THE CONTROL IS REACHED AT 0.65
OR AT 0.95, the floor is over-conservative and roughly twice as many controls are usable; if the number drifts with
reach, the floor is earning its place and should perhaps be higher.
This is instrument calibration, which lesson 6 asks for directly: check an instrument by running it against known
cases and read what the number depends on. One fit -- about_for, chosen because v527 and v531 already place two of
its partners far apart at 0.859 and 0.656 -- against SIX vocabulary-disjoint partners spanning the range:
    at_to (0.859 known), at_over (0.656 known), by_from, in_to, at_into, by_with
Every partner's answer pair is disjoint from (` about`, ` for`). Both quantities come from the same fitted direction
in one run, so reach and effect are measured on the same footing. Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fit_holds        the fitted task reaches MINIMUM held-out joint extraction joint_min = 0.80. prior 90%
  pred_b_range_present    the six partners' reach values span at least gain_range = 0.20 between highest and
                          lowest. This is the guard v517 taught me to register: if the partners all cluster, the
                          question is unanswerable and nothing below is readable.                     prior 80%
  pred_c_all_effects_small every partner's same-answer effect is at or below p_max = 0.23 REGARDLESS of its reach.
                                                                                                      prior 70%
  pred_d_effect_flat_in_reach the effect at the LOWEST-reach partner differs from the effect at the HIGHEST-reach
                          partner by at most gain = 0.05. Worked example: 0.006 against 0.014 is 0.008, TRUE and the
                          estimate does not depend on reach across this range; 0.006 against 0.20 is 0.194, FALSE
                          and the floor is doing real work.                                           prior 55%
  pred_e_reach_reproduces at_to and at_over land within tol_pool = 0.05 of 0.859 and 0.656. Deterministic fit;
                          cross-receipt consistency check.                                            prior 85%
HOW IT READS, one outcome per branch. b, c and d TRUE: the specificity estimate is insensitive to reach across
0.65 to 0.95, so the 0.80 floor is CONSERVATIVE rather than necessary, controls down to about 0.65 give the same
answer, and the pool of usable controls is larger than every rung since v499 has assumed -- which would widen how
many tasks can carry a four-hypothesis claim. d FALSE: the estimate drifts with reach, the floor is earning its
place, and the honest response is to report the drift and consider whether 0.80 is high enough rather than lower it.
b FALSE: the partners cluster and the question is unanswerable here; I will not read c or d.
SCOPE. One fitted task, six partners, rank 1, standard budget. A flat result would license widening the floor for
THIS family, not for the corpus.
Smoke: V533_SMOKE=<out.json> (CPU, V533_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_reach_floor_calibration_v533_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
FIT = "about_for"
PARTNERS = ("at_to", "at_over", "by_from", "in_to", "at_into", "by_with")   # all disjoint from {about, for}
MAPS = {"about_for": {"fit": (VP + FIT,), "held": tuple(VP + p for p in PARTNERS)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"at_to": 0.859, "at_over": 0.656}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"gain_range": 0.20, "p_max": 0.23, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_reach_floor_calibration_v533", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    b = have and (max(reach.values()) - min(reach.values())) >= B["gain_range"]
    c = have and all(abs(eff[p]) <= B["p_max"] for p in PARTNERS)
    lo = min(PARTNERS, key=lambda p: reach[p]) if have else None
    hi = max(PARTNERS, key=lambda p: reach[p]) if have else None
    d = have and abs(abs(eff[lo]) - abs(eff[hi])) <= B["gain"]
    e = have and abs(reach["at_to"] - PRIOR["at_to"]) <= B["tol_pool"] \
        and abs(reach["at_over"] - PRIOR["at_over"]) <= B["tol_pool"]
    return {"pred_a_fit_holds": bool(a), "pred_b_range_present": bool(b),
            "pred_c_all_effects_small": bool(c), "pred_d_effect_flat_in_reach": bool(d),
            "pred_e_reach_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V533_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V533_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_reach_floor_calibration_v533",
              "candidate_id": "corpus.unit_reach_floor_calibration_v533", "bars": BARS,
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
