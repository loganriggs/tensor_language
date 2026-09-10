#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v505: P and C read with the instrument the corpus already ships for them, and with bars that can fail.

WHAT v503 GOT WRONG, MINE. It registered P as REMOVAL damage <= 0.01. P rows carry the SAME answer, so under
mean-ablation P removal tracks the same-side A1 removal -- measured at 2.2625 against A1's 2.0035 -- and the bar
could not have been met by any direction that carries the variable. A clause that cannot pass.
WHAT READING THE CODE FOUND. `circuit_unit_greedy` already ships `same_answer_effect(prep, patched, scale)`,
documented in one line as "Mean |movement| in units of the target families' native separation (P and C)", with
`target_scale(prep)` giving that separation as the median |donor - base| of the A1 family. That is the instrument
for these two hypotheses under PATCHING, and it normalises the movement by the effect the direction is supposed to
have, so the numbers are interpretable rather than raw. I used removal instead of reading the definition; lesson 6
is exactly this and I did not apply it.
THE BARS ARE SET SO THEY CAN FAIL. This project's recorded range is that same-answer controls land at 0.07-0.23
against a 0.35 bar -- so 0.35 is a clause that cannot fail here, and registering it would repeat the v503 error with
the sign flipped. The bar used is p_max = 0.23, the TOP of the observed range: a direction that disturbs P or C more
than any same-answer control has been observed to move will fail it, and one inside the range will pass.
Same deterministic fit as v497, v499, v501 and v503; RANK FIXED AT 1 and registered, per the protocol.
    A1  held-out extraction, the instrument check -- measured at 1.001 four times on this fit
    A2  held-out extraction on the cell's second construction -- 0.935 in v503
    P   same_answer_effect under patching, scaled by the A1 family's native separation
    C   the v501 disjoint cell, same instrument and same scale, so P and C are directly comparable
Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_reproduces_fit  the A1 held-out extraction lands within tol_pool = 0.05 of 1.001. Deterministic fit;
                         instrument check.                                                            prior 90%
  pred_b_P_effect_small  the P family's same-answer effect is at or below p_max = 0.23. Worked example: 0.15 is
                         inside the observed range, TRUE; 0.40 is above anything recorded for a same-answer
                         control, FALSE and the direction disturbs an answer-preserving edit.         prior 65%
  pred_c_C_effect_small  the v501 disjoint cell's same-answer effect is at or below 0.23, on the same instrument
                         and the same scale.                                                          prior 75%
  pred_d_P_and_C_comparable  the two effects differ by at most tol_pool_wide = 0.15. P shares the cue and the
                         construction with the target while C shares neither, so a large gap would say the
                         direction is disturbed by the answer-preserving edit specifically rather than behaving the
                         same way toward every same-answer family. No baseline in this corpus predicts either
                         outcome, which is why it is worth a clause.                                  prior 55%
  pred_e_units_available every evaluated family reports units_recovery >= unit_min = 0.80.            prior 85%
HOW IT READS, one outcome per branch. b and c TRUE with d TRUE: P and C are both undisturbed and behave alike, so
rows 3 and 4 of the battery hold for this direction on the instrument built for them, and the only outstanding
caveat from v503 -- the 0.1236 removal collateral on C -- is an instrument difference rather than a specificity
failure. b FALSE: the direction is disturbed by the answer-preserving edit. c FALSE: it damages a same-site
behaviour with disjoint tokens and row 4 fails outright. d FALSE alone with b and c TRUE: both are small but P moves
materially more than C, which would locate the residual disturbance in the shared cue rather than in the site.
SCOPE. One mapping, rank 1, this budget.
Smoke: V505_SMOKE=<out.json> (CPU, V505_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_correlative_p_interchange_v505_result.json"
# one fit, four hypotheses read on the same rank-1 direction
CANON = "correlative_pair"
DISJOINT_C = "correlative_disjoint_either_not"      # v501: reached at 0.973, inert at -0.021
MAPS = {"fit_canonical": {"fit": (CANON,), "held": (DISJOINT_C,)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v495_A1": 1.001, "v501_disjoint_units": 0.973, "v501_disjoint_abs": -0.021}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_correlative_p_interchange_v505", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("fit_canonical", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL["fit_canonical"])
    ps = g0.get("per_shape", {}) if ok else {}
    hyp = g0.get("hypotheses", {}) if ok else {}
    a1_v = ps.get(CANON, {}).get("joint_extraction")
    p_eff = hyp.get("P_same_answer_effect")
    c_eff = hyp.get("C_same_answer_effect")
    have = ok and all(x is not None for x in (a1_v, p_eff, c_eff))
    a = have and abs(a1_v - PRIOR["v495_A1"]) <= B["tol_pool"]
    b = have and abs(p_eff) <= B["p_max"]
    c = have and abs(c_eff) <= B["p_max"]
    d = have and abs(abs(p_eff) - abs(c_eff)) <= B["tol_pool_wide"]
    e = ok and ps.get(CANON, {}).get("units_recovery", 0) >= B["unit_min"] \
        and ps.get(DISJOINT_C, {}).get("units_recovery", 0) >= B["unit_min"] \
        and (hyp.get("A2_units_recovery") or 0) >= B["unit_min"]
    return {"pred_a_reproduces_fit": bool(a), "pred_b_P_effect_small": bool(b),
            "pred_c_C_effect_small": bool(c), "pred_d_P_and_C_comparable": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V505_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V505_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                if cell == CANON:                       # the other two hypotheses, read on the same direction
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
            scale = g.target_scale(g.prepare(backend, per_shape_rows[CANON][0], **V))
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            p_disj = g.prepare(backend, per_shape_rows[DISJOINT_C][0], **V)
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
    result = {"predictions": predictions, "schema": "unit_correlative_p_interchange_v505",
              "candidate_id": "corpus.unit_correlative_p_interchange_v505", "bars": BARS,
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
