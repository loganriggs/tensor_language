#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v523: a THIRD four-hypothesis target, and this one comes with a caveat that changes what a pass means.

WHY THIS TARGET. A specificity reading needs a control the fitted nodes actually REACH, and v511 measured exactly
that for this task: a set fitted on noun_preposition_interest recovers 0.913 of the patching effect on
between_with, whose answer pair (` between`, ` with`) shares no token with (` in`, ` for`). So the C row here can
fail for a reason -- the property that took four rungs to obtain for correlative and that possessive still lacks.
WHY NOT THE OBVIOUS ALTERNATIVE. v521 opened a route to a row-4 claim for adjective_preposition_in_beneath, whose
larger node set reaches of_at at 0.976. I inspected that task before spending a screen on it and it is confounded:
its arms are `interested` -> ` in` against `buried` -> ` beneath`, an adjective selecting a complement against a
PASSIVE PARTICIPLE taking a locative adjunct. Reading the cue pairs across the family found the same shape in about
eight of thirty-eight tasks -- hurried past, hidden beneath, salvaged past, subsumed under, enclosed within,
concealed beneath, glimpsed past -- so those interchanges are confounded with adjective-versus-participle and
argument-versus-adjunct rather than isolating which preposition a word selects. None of them appears in any
canonical circuit record, so nothing counted rests on them, and that is why this rung goes elsewhere.
THE CAVEAT THAT BOUNDS A PASS, REGISTERED IN ADVANCE. v489 showed that in THIS family the model is a recency reader:
interpose the other member of the cue pair -- `The pilot showed interest, despite the ongoing respect, of course,` --
and all 32 rows fail, while a neutral distractor in the identical frame keeps all 32. So cue and nearest noun cannot
be separated behaviourally here. A four-hypothesis pass therefore says the rank-1 subspace carries the variable AS
THE MODEL COMPUTES IT, and the model computes it by recency. That is a real result about a real mechanism; it is not
a claim that the subspace encodes noun-governed complement selection, and I will not write it as one.
RANK IS FIXED AT 1 AND REGISTERED HERE, per the protocol. Instruments as the corpus ships them: extraction for A1
and A2, `same_answer_effect` scaled by `target_scale` for P and C. Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_A1_holdout    held-out A1 rows reach MINIMUM joint extraction joint_min = 0.80.              prior 85%
  pred_b_A2_carries    the same direction reaches 0.80 on the task's A2 construction.                 prior 60%
  pred_c_P_effect_small the P family's same-answer effect is at or below p_max = 0.23, the top of the recorded
                       saturation range. 0.35 would be a clause that cannot fail and is not used.     prior 70%
  pred_d_C_effect_small between_with's same-answer effect is at or below 0.23, on the same instrument and scale.
                                                                                                      prior 70%
  pred_e_units_available the fitted task, its A2 family and the control all report units_recovery >= unit_min = 0.80,
                       reproducing v511's 0.913 for the control.                                      prior 85%
HOW IT READS, one outcome per branch. All true: a third target passes all four hypotheses with a control that is
inert for an informative reason, and the protocol is satisfied on two of the three named targets plus this one.
b FALSE: the direction is construction-bound. c FALSE: it is disturbed by the answer-preserving edit. d FALSE: it
moves a genuinely unrelated behaviour whose nodes it demonstrably reaches, which would be the first real specificity
FAILURE in this corpus rather than an untestable one -- and far more informative than another pass.
SCOPE. One task, rank 1, this budget, with the recency caveat above attached to any pass.
Smoke: V523_SMOKE=<out.json> (CPU, V523_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_noun_four_hypotheses_v523_result.json"
# one fit, four hypotheses read on the same rank-1 direction
CANON = "noun_preposition_interest"
DISJOINT_C = "noun_preposition_between_with"        # v511: reached at 0.913, disjoint vocabulary
MAPS = {"fit_canonical": {"fit": (CANON,), "held": (DISJOINT_C,)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v511_control_units": 0.913, "v489_recency": "model follows the nearest noun in this family"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_noun_four_hypotheses_v523", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    a = have and a1_v >= B["joint_min"]
    a2_v = hyp.get("A2_extraction")
    b = have and a2_v is not None and a2_v >= B["joint_min"]
    c = have and abs(p_eff) <= B["p_max"]
    d = have and abs(c_eff) <= B["p_max"]
    e = ok and ps.get(CANON, {}).get("units_recovery", 0) >= B["unit_min"] \
        and ps.get(DISJOINT_C, {}).get("units_recovery", 0) >= B["unit_min"] \
        and (hyp.get("A2_units_recovery") or 0) >= B["unit_min"]
    return {"pred_a_A1_holdout": bool(a), "pred_b_A2_carries": bool(b),
            "pred_c_P_effect_small": bool(c), "pred_d_C_effect_small": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V523_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V523_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_noun_four_hypotheses_v523",
              "candidate_id": "corpus.unit_noun_four_hypotheses_v523", "bars": BARS,
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
