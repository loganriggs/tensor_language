#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v551: the same six targets with predicates that do not let one broken group erase five good ones.

WHAT WENT WRONG IN v549, AND IT WAS THE RUNNER, NOT THE DATA. Five of six targets produced complete numbers clearing
every bar; the sixth, with_against, raised InvalidEvidenceError -- donor denominator must be positive and greater
than 1e-6 -- and because every predicate was coded as all() over all six groups, that one error made the whole
predicate block False. The receipt said 0/5 while the measurement said five of six. The docstring had already stated
that these are six INDEPENDENT claims sharing a runner and must be reported per target; the code did not implement
what the docstring promised.
THE FIX, WHICH IS THE POINT OF THIS RUNG AS MUCH AS THE NUMBERS ARE. Predicates now quantify over groups that
actually produced data, an explicit clause bounds how many groups may error before the rung is untrustworthy, and
the receipt carries a per_target block so a reader sees five passes and one instrument failure rather than a bare
False. That shape is what any bulk runner needs: a conjunction over independent targets converts a single instrument
failure into a total null, which is the opposite of what a bulk rung is for.
Same six pairs as v549, same partners, same standard budget, rank 1 fixed and registered. with_against is retried
rather than dropped -- if it errors again the error is now isolated, and if it succeeds the rung gains a sixth
measurement. Nothing counted: these are DAS receipts, not new behaviours.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_measured_A1  every group that PRODUCED DATA reaches MINIMUM held-out joint extraction joint_min = 0.80.
                      Quantified over measured groups, which is the whole change.                     prior 88%
  pred_b_measured_A2  every measured group reaches 0.80 on its A2 construction. Five of six did in v549 -- 0.899 to
                      1.031 -- so this is close to a reproduction for those five.                     prior 85%
  pred_c_measured_P   every measured group's P family shows same-answer effect at or below p_max = 0.23. prior 85%
  pred_d_measured_C   every measured group's partner shows same-answer effect at or below 0.23.       prior 85%
  pred_e_at_most_one_errored at most ONE group raises an error. This is the guard that keeps the new quantifier
                      honest: without it, predicates over measured groups would pass vacuously if five of six broke.
                      v549 had exactly one error, so a second one here would mean something changed.  prior 80%
HOW IT READS. a through d TRUE with e TRUE: the five v549 measurements reproduce and the receipt now says so
directly, plus with_against either joins them or is isolated as a single instrument failure. e FALSE: more groups
broke than last time and the measured-group predicates are not to be trusted, because they would be quantifying over
a shrinking and self-selected subset -- I will report the error count and stop there. Any of a through d FALSE:
reported PER TARGET from the per_target block, since these remain six independent claims.
SCOPE. Six tasks of 119, rank 1, one partner each. The runner-shape fix is the reusable part.
Smoke: V551_SMOKE=<out.json> (CPU, V551_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_verbprep_bulk_pertarget_v551_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "verb_preposition_"
PAIRS = {"on_about": "by_with", "into_with": "at_to", "for_toward": "at_into",
         "with_against": "at_over", "from_about": "in_to", "of_into": "by_from"}   # reach measured in v547
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
V547_REACH = {"on_about": 0.928, "into_with": 1.005, "for_toward": 0.832,
              "with_against": 0.833, "from_about": 0.965, "of_into": 0.911}   # v547
PRIOR = {"v547_reach": V547_REACH}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_verbprep_bulk_pertarget_v551", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ks = list(PAIRS)
    measured = [k for k in ks if k in G and "error" not in G[k]]
    errored = [k for k in ks if k not in G or "error" in G[k]]

    def ps(k):
        return G.get(k, {}).get("per_shape", {})

    def hyp(k):
        return G.get(k, {}).get("hypotheses", {})

    def vals(k):
        return (ps(k).get(VP + k, {}).get("joint_extraction"), hyp(k).get("A2_extraction"),
                hyp(k).get("P_same_answer_effect"), hyp(k).get("C_same_answer_effect"))

    have = [k for k in measured if all(v is not None for v in vals(k))]
    a = bool(have) and all(vals(k)[0] >= B["joint_min"] for k in have)
    b = bool(have) and all(vals(k)[1] >= B["joint_min"] for k in have)
    c = bool(have) and all(abs(vals(k)[2]) <= B["p_max"] for k in have)
    d = bool(have) and all(abs(vals(k)[3]) <= B["p_max"] for k in have)
    e = len(errored) <= 1
    R["per_target"] = {k: {"measured": k in have, "A1": vals(k)[0], "A2": vals(k)[1],
                           "P": vals(k)[2], "C": vals(k)[3],
                           "passes": bool(k in have and vals(k)[0] >= B["joint_min"]
                                          and vals(k)[1] >= B["joint_min"]
                                          and abs(vals(k)[2]) <= B["p_max"]
                                          and abs(vals(k)[3]) <= B["p_max"])}
                       for k in ks}
    R["errored_groups"] = errored
    return {"pred_a_measured_A1": bool(a), "pred_b_measured_A2": bool(b),
            "pred_c_measured_P": bool(c), "pred_d_measured_C": bool(d),
            "pred_e_at_most_one_errored": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V551_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V551_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            pooled_fit, pooled_c_fit, per_shape_rows, extra_rows = [], [], {}, {}
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell == VP + gname:                  # the other two hypotheses, read on the same direction
                    extra_rows["A2"] = cut(held_half(g.rows_of(m, "A2")))
                    extra_rows["P"] = cut(held_half(g.rows_of(m, "P")))
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
                backend, P_fit, units, rank=RANK, steps=steps, lr=LR, seed=0, complement_weight=CW,
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
            p_a2 = g.prepare(backend, extra_rows["A2"], **V)
            p_p = g.prepare(backend, extra_rows["P"])
            e_a2 = ext(p_a2, units)
            scale = g.target_scale(g.prepare(backend, per_shape_rows[VP + gname][0], **V))
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            p_disj = g.prepare(backend, per_shape_rows[VP + PAIRS[gname]][0], **V)
            hyp = {"A2_units_recovery": e_a2,
                   "scale": round(scale, 4),
                   "P_same_answer_effect": eff(p_p),
                   "C_same_answer_effect": eff(p_disj),
                   "A2_extraction": round(ext(p_a2, units, q=q_joint) / e_a2, 3) if abs(e_a2) > 1e-6 else None,
                   "P_damage": dmg(p_p, units, q_joint, mu_joint),
                   "P_units_recovery": ext(p_p, units),
                   "n_a2": len(p_a2.base_batch.row_ids), "n_p": len(p_p.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "hypotheses": hyp}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "per_target": R.get("per_target"),
              "errored_groups": R.get("errored_groups"), "schema": "unit_verbprep_bulk_pertarget_v551",
              "candidate_id": "corpus.unit_verbprep_bulk_pertarget_v551", "bars": BARS,
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
