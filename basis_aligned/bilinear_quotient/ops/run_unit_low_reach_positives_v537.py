#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v537: does a control BELOW the reach floor still register movement? Extend the positive panel downward.

WHERE THIS STANDS. Since v499 every rung has used a 0.80 floor on how well a control is reached, on the reasoning
that a near-zero reading from a control the fitted nodes do not reach says nothing. v531 showed the floor cutting
through a continuum; v533 showed the specificity estimate flat across reach 0.66 to 0.86 but could not interpret
that, because ALL its partners were inert; v535 fixed the panel with positive cases and found the instrument has
real dynamic range -- 0.7144 and 0.7414 for tasks the direction should move against 0.0030 and 0.0279 for tasks it
should not.
THE PART OF v535 I DID NOT DESIGN FOR IS WHAT THIS RUNG PURSUES. One positive, fb_about_for, has reach 0.771 --
BELOW my own floor -- and still registered 0.7144. So the floor is not what makes a reading informative, at least at
0.771. But that is ONE point, and at_over's 0.656 still has no positive case, so I refused to move the floor on it.
This rung extends the positive panel to ALL SIX same-mapping frame copies of about_for -- same cue pair, same readout
pair, different sentence frame, all established by v455 as covered by a pooled direction at about 1.0 -- and reads
reach and effect for each. Whatever reach those six happen to span is the range over which the refutation gets
tested, and I am not choosing them to hit a number.
    POSITIVE  fb, fc, fd, fe, fg, fh about_for
    NEGATIVE  at_to, at_over                       carried from v533 and v535 as the low anchors
Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fit_holds        the fitted task reaches MINIMUM held-out joint extraction joint_min = 0.80.  prior 90%
  pred_b_low_reach_present at least one positive has reach at or below low_reach = 0.75, so the question is
                          answerable. The guard v517 taught me to register: if every copy sits above the floor the
                          panel cannot test anything below it and nothing else here is readable.      prior 60%
  pred_c_all_positives_move EVERY positive shows same-answer effect at or above pos_min = 0.35, regardless of its
                          reach.                                                                      prior 65%
  pred_d_effect_flat_across_reach the lowest-reach positive and the highest-reach positive differ in effect by at
                          most tol_pool_wide = 0.15, i.e. detectability does not degrade with reach across the
                          span these copies happen to cover.                                          prior 60%
  pred_e_anchors_reproduce fb and fc land within tol_pool = 0.05 of 0.7144 and 0.7414, and at_to within tol_pool of
                          0.0030. Deterministic fit; cross-receipt consistency.                       prior 85%
HOW IT READS, one outcome per branch. b, c and d TRUE: the instrument registers movement across the whole reach span
these copies cover, so reach does not gate DETECTABILITY there and the floor should be reported as conservative down
to the lowest reach observed -- with the number named, not rounded. c FALSE: the low-reach positives fail to
register, so reach DOES gate detection at the bottom of the range and the floor is earning its place -- which would
mean v535's single point at 0.771 was near the edge rather than deep inside the safe region. b FALSE: every copy
sits above 0.75, the panel cannot probe below the floor, and I will not read c or d as if it had.
SCOPE. One fitted task and its own frame copies, rank 1. This calibrates detectability for THIS direction; the
floor stays where it is for other directions until the same check is run on them.
Smoke: V537_SMOKE=<out.json> (CPU, V537_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_low_reach_positives_v537_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
FIT = "about_for"
POSITIVE = ("fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for",
            "fg_about_for", "fh_about_for")     # same mapping, different frame -- should MOVE
NEGATIVE = ("at_to", "at_over")                 # disjoint vocabulary -- should not
PARTNERS = POSITIVE + NEGATIVE
MAPS = {"about_for": {"fit": (VP + FIT,), "held": tuple(VP + p for p in PARTNERS)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"fb_effect": 0.7144, "fc_effect": 0.7414, "at_to_effect": 0.0030}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"low_reach": 0.75, "pos_min": 0.35, "range_min": 0.20, "gain_range": 0.20, "p_max": 0.23, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_low_reach_positives_v537", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    b = have and any(reach[p] <= B["low_reach"] for p in POSITIVE)
    c = have and all(abs(eff[p]) >= B["pos_min"] for p in POSITIVE)
    lo = min(POSITIVE, key=lambda p: reach[p]) if have else None
    hi = max(POSITIVE, key=lambda p: reach[p]) if have else None
    d = have and abs(abs(eff[lo]) - abs(eff[hi])) <= B["tol_pool_wide"]
    e = have and abs(abs(eff["fb_about_for"]) - PRIOR["fb_effect"]) <= B["tol_pool"] \
        and abs(abs(eff["fc_about_for"]) - PRIOR["fc_effect"]) <= B["tol_pool"] \
        and abs(abs(eff["at_to"]) - PRIOR["at_to_effect"]) <= B["tol_pool"]
    return {"pred_a_fit_holds": bool(a), "pred_b_low_reach_present": bool(b),
            "pred_c_all_positives_move": bool(c), "pred_d_effect_flat_across_reach": bool(d),
            "pred_e_anchors_reproduce": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V537_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V537_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_low_reach_positives_v537",
              "candidate_id": "corpus.unit_low_reach_positives_v537", "bars": BARS,
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
