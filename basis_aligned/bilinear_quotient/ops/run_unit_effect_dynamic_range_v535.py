#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v535: v533's panel had no positive case. Give the instrument something it SHOULD move.

WHAT v533 COULD NOT DECIDE, AND I SAID SO IN ITS RECEIPT. Six vocabulary-disjoint partners of one fit spanned reach
0.656 to 0.859 and their specificity effects were flat -- 0.0279, 0.0432, 0.0183, 0.0295, 0.0440, 0.0030, none
related to reach. I registered that as evidence the estimate does not depend on reach. But ALL SIX WERE INERT, so
the flatness is equally consistent with "the estimate does not depend on reach" and with "this direction happens to
be specific to all six of these tasks", and a panel with no positive case cannot separate them. Lesson 6 asks for
exactly this: check an instrument by running it against known-GOOD and known-BAD cases.
THE POSITIVE CASES ARE ALREADY ESTABLISHED, WHICH IS WHY THIS IS CHEAP. v455 measured frame copies of about_for --
same cue pair, same readout pair, different sentence frame -- covered by a pooled direction at about 1.0. So the
about_for direction SHOULD move fb_about_for and fc_about_for hard, and if `same_answer_effect` cannot show that, the
instrument has no dynamic range and every small number it has produced today means less than I have been reading
into it.
    POSITIVE  fb_about_for, fc_about_for   same mapping, different frame -- the direction should move these
    NEGATIVE  at_to (reach 0.859, effect 0.0030), at_over (reach 0.656, effect 0.0279)  -- carried from v533
Same fit, same instrument, same scale, one run. Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fit_holds       the fitted task reaches MINIMUM held-out joint extraction joint_min = 0.80.  prior 90%
  pred_b_positives_move  BOTH positive partners show same-answer effect at or above pos_min = 0.35 -- comfortably
                         above the 0.23 noise bar, so a pass means real movement rather than drift. Capable of
                         failing, and its failure would be the more important result: it would mean the instrument
                         cannot register movement it ought to and today's small numbers are uninformative. prior 70%
  pred_c_negatives_small BOTH negative partners stay at or below p_max = 0.23, reproducing v533.       prior 85%
  pred_d_dynamic_range   the SMALLER positive effect exceeds the LARGER negative effect by at least
                         range_min = 0.20, i.e. the instrument separates cases it should move from cases it should
                         not. Worked example: 0.80 against 0.03 is 0.77, TRUE; 0.20 against 0.15 is 0.05, FALSE and
                         the scale is too compressed to carry the readings I have been making.        prior 70%
  pred_e_v533_reproduces at_to and at_over land within tol_pool = 0.05 of 0.0030 and 0.0279. Deterministic fit;
                         cross-receipt consistency.                                                   prior 85%
HOW IT READS, one outcome per branch. b, c and d TRUE: the instrument has dynamic range, the inert readings across
today's rungs are real inertness rather than a floor effect, and v533's flatness can be read as I registered it.
b FALSE: `same_answer_effect` does not register movement it should, and every specificity number I have quoted today
-- including the four-hypothesis passes -- rests on a measure that cannot distinguish inert from unmeasured, which I
would report as the finding and act on before running another specificity rung. d FALSE with b TRUE: it moves but
the separation is small, so the scale is compressed and the bars need rethinking rather than the readings.
SCOPE. One fitted task, four partners, rank 1. This calibrates the instrument on this direction, not on all of them.
Smoke: V535_SMOKE=<out.json> (CPU, V535_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_effect_dynamic_range_v535_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
FIT = "about_for"
POSITIVE = ("fb_about_for", "fc_about_for")     # same mapping, different frame -- should MOVE
NEGATIVE = ("at_to", "at_over")                 # disjoint vocabulary -- should not
PARTNERS = POSITIVE + NEGATIVE
MAPS = {"about_for": {"fit": (VP + FIT,), "held": tuple(VP + p for p in PARTNERS)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"at_to_effect": 0.0030, "at_over_effect": 0.0279}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"pos_min": 0.35, "range_min": 0.20, "gain_range": 0.20, "p_max": 0.23, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_effect_dynamic_range_v535", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    c = have and all(abs(eff[p]) <= B["p_max"] for p in NEGATIVE)
    d = have and (min(abs(eff[p]) for p in POSITIVE) - max(abs(eff[p]) for p in NEGATIVE)) >= B["range_min"]
    e = have and abs(abs(eff["at_to"]) - PRIOR["at_to_effect"]) <= B["tol_pool"] \
        and abs(abs(eff["at_over"]) - PRIOR["at_over_effect"]) <= B["tol_pool"]
    return {"pred_a_fit_holds": bool(a), "pred_b_positives_move": bool(b),
            "pred_c_negatives_small": bool(c), "pred_d_dynamic_range": bool(d),
            "pred_e_v533_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V535_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V535_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_effect_dynamic_range_v535",
              "candidate_id": "corpus.unit_effect_dynamic_range_v535", "bars": BARS,
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
