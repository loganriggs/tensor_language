#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v515: the shared control cannot fail. How much collateral has it been hiding?

THE EXPOSURE. Every task in this corpus is screened against the same control, and its clean and corrupted sides
carry the SAME answer -- `...finished the survey in the middle of the` -> ` night` on both. Patching cannot flip a
logit difference that does not exist, so that control is close to unfailable; this project's own lesson records
same-answer controls saturating at 0.07-0.23 against a 0.35 bar. Its final token is also ` the`, while the tasks it
guards mostly end in `,` -- and patching happens at the final position, so a direction that encodes "a preposition is
due after a comma-parenthetical" would score zero on it for free.
v501 and v505 replaced it for ONE task with an answer-changing, vocabulary-disjoint control that the fitted nodes
actually reach, and the specificity number became meaningful rather than automatic. v511 then found that such a
control is REACHABLE in three constructions of six. This rung asks what the swap is worth in those three: both
controls are scored for the SAME fitted direction, on the SAME instrument and the SAME scale, so the difference is
the collateral the degenerate control has been missing.
    verb_preposition   fit at_to      canonical control  vs  about_for      (reachable, 0.924 in v487)
    noun_preposition   fit interest   canonical control  vs  between_with   (reachable, 0.913 in v511)
    correlative        fit correlative_pair  canonical control  vs  disjoint_either_not (reachable, 0.973 in v501)
Instrument for both: `same_answer_effect` scaled by `target_scale` of the task's own A1 family -- movement in units
of the effect the direction is supposed to have. Rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold        all three fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80.
                          Instrument check.                                                           prior 85%
  pred_b_canonical_inert  the CANONICAL control's movement is at or below p_max = 0.23 in all three. Expected --
                          it is the saturating one -- and registered so that a failure here would be news about the
                          control rather than about the direction.                                    prior 90%
  pred_c_disjoint_inert   the ANSWER-CHANGING disjoint control's movement is at or below 0.23 in all three.
                                                                                                      prior 65%
  pred_d_disjoint_larger  in at least TWO of the three, the disjoint control moves MORE than the canonical one by
                          at least gain = 0.05. Worked example: 0.09 against 0.02 is 0.07, TRUE and the degenerate
                          control was missing collateral that a real one detects; 0.03 against 0.02 is 0.01, FALSE
                          and the two agree, which would mean the shared control -- for all its design faults --
                          was not actually hiding anything in these three cases.                      prior 60%
  pred_e_disjoint_reached every disjoint control reports units_recovery >= unit_min = 0.80, reproducing v487, v511
                          and v501. A consistency check across three receipts, not a question.        prior 90%
HOW IT READS, one outcome per branch. c TRUE and d TRUE: the swap matters in degree but not in verdict -- the
directions really are specific, and the shared control was understating collateral without changing any conclusion,
which is the good outcome and bounds the damage. c FALSE: at least one direction moves a genuinely unrelated
behaviour past the bar, so a specificity claim that the shared control passed does NOT survive a control capable of
failing, and every claim resting on it in that construction needs re-reading. d FALSE with c TRUE: the two controls
agree, the shared control was adequate here despite its design, and the exposure I described is smaller than the
design fault suggests -- which I would report as such rather than defending the framing.
SCOPE. Three constructions, one task each, rank 1. It says nothing about the three constructions where no reachable
disjoint control exists -- there the question remains open for want of an instrument, which is v511's finding.
Smoke: V515_SMOKE=<out.json> (CPU, V515_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_control_exposure_v515_result.json"
# one fit per construction; each scores the CANONICAL same-answer control and an answer-changing disjoint one
MAPS = {
    "verb_preposition": {"fit": ("verb_preposition_at_to",), "held": ("verb_preposition_about_for",)},
    "noun_preposition": {"fit": ("noun_preposition_interest",), "held": ("noun_preposition_between_with",)},
    "correlative": {"fit": ("correlative_pair",), "held": ("correlative_disjoint_either_not",)},
}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"reachable_units": {"verb_preposition": 0.924, "noun_preposition": 0.913, "correlative": 0.973}}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"p_max": 0.23, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_control_exposure_v515", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, f):
        return G.get(k, {}).get(f)

    def sib(k, f):
        return G.get(k, {}).get("per_shape", {}).get(MAPS[k]["held"][0], {}).get(f)

    fit_v = [G.get(k, {}).get("per_shape", {}).get(MAPS[k]["fit"][0], {}).get("joint_extraction") for k in GROUPS]
    canon = {k: fld(k, "canonical_effect") for k in GROUPS}
    disj = {k: fld(k, "disjoint_effect") for k in GROUPS}
    have = ok and all(x is not None for x in fit_v) \
        and all(v is not None for v in list(canon.values()) + list(disj.values()))
    a = have and min(fit_v) >= B["joint_min"]
    b = have and all(abs(canon[k]) <= B["p_max"] for k in GROUPS)
    c = have and all(abs(disj[k]) <= B["p_max"] for k in GROUPS)
    d = have and sum(1 for k in GROUPS if (abs(disj[k]) - abs(canon[k])) >= B["gain"]) >= 2
    e = ok and all(sib(k, "units_recovery") is not None and sib(k, "units_recovery") >= B["unit_min"]
                   for k in GROUPS)
    return {"pred_a_fits_hold": bool(a), "pred_b_canonical_inert": bool(b),
            "pred_c_disjoint_inert": bool(c), "pred_d_disjoint_larger": bool(d),
            "pred_e_disjoint_reached": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V515_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V515_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            fit_cell, held_cell = MAPS[gname]["fit"][0], MAPS[gname]["held"][0]
            p_a1 = g.prepare(backend, per_shape_rows[fit_cell][0], **V)
            scale = g.target_scale(p_a1)
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            p_canon = g.prepare(backend, per_shape_rows[fit_cell][1])          # the shared same-answer control
            p_disj = g.prepare(backend, per_shape_rows[held_cell][0], **V)
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "scale": round(scale, 4),
                             "canonical_effect": eff(p_canon), "disjoint_effect": eff(p_disj)}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_control_exposure_v515",
              "candidate_id": "corpus.unit_control_exposure_v515", "bars": BARS,
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
