#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v475: does the adjacent-antecedent direction read SUBJECT number, or the NEAREST noun's?

THE CHALLENGE THIS ANSWERS. An adversarial audit of possessive_number reported that in all seven existing cells the
intervener's number is HELD CONSTANT and never covaries with the answer, so "number of the subject", "number of the
nearest noun" and "number of the nearest human noun" have never been separated -- and the one cell that would have
separated them, animate_attractor, is on the board as a donor-side capability failure. That matters here because
this project already records these circuits as RECENCY readers elsewhere, and because v419 found the rank-1
adjacent_antecedent direction damaging every matched sibling and concluded it "encodes the construction SLOT rather
than the variable". The stimulus design is a candidate explanation for both.
The new cell makes the separation possible and was capability-screened first: `Near the desk the leaders near the
crates lost` -> ` their` against `... the leader near the crates lost` -> ` his`, attractor held PLURAL on both sides,
base and donor differing in exactly ONE word which is a single token on both sides. I registered before that screen
that a recency reader would collapse the donor side; it did not -- 32/32 rows kept, margins +/-4.0. So the MODEL
resists the attractor. What the SITE encodes is the open question and this rung is the first that can ask it.
    FIT      possessive_adjacent            `The leaders checked` -- the named DAS target, no intervener at all
    HELD OUT possessive_attractor           intervener SINGULAR on both sides, so the conflict falls on the BASE
                                            side (plural subject, singular nearest noun)
             possessive_number_attractor_conflict  intervener PLURAL on both sides, so the conflict falls on the
                                            DONOR side. The two cells put the conflict on OPPOSITE interchange
                                            sides, which is why both are here rather than either alone.
    CONTROL  verb_preposition_about_for, a different readout pair entirely, on ABSOLUTE recovery.
RANK IS FIXED AT 1 AND REGISTERED HERE IN ADVANCE, per the standing DAS protocol; a null is not permission to raise
it. Budget stays relaxed (0.97 / 0.001, 30 units). Nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution   held-out rows of the FITTED cell reach MINIMUM joint extraction joint_min = 0.80. If this
                           fails nothing else is readable.                                            prior 85%
  pred_b_singular_attractor the existing attractor cell reaches 0.80 -- an intervener that conflicts on the BASE
                           side costs the direction nothing.                                          prior 55%
  pred_c_plural_attractor  the new conflict cell reaches 0.80 -- an intervener that conflicts on the DONOR side
                           costs the direction nothing.                                               prior 45%
  pred_d_units_available   every evaluated cell except the control reports units_recovery >= unit_min = 0.80, so a
                           failure is attributable to the DIRECTION and not to units that miss the shape. prior 75%
  pred_e_different_readout the control's ABSOLUTE recovery stays at or below cross_max = 0.30 -- the direction is
                           the number axis and not a generic next-token axis.                         prior 85%
HOW IT READS, fixed in advance. b TRUE and c TRUE: the site carries SUBJECT number and survives an attractor on
either side; the audit's recency challenge is answered for this family and the name survives. c FALSE with b TRUE:
the direction fails exactly when the NEAREST noun conflicts on the donor side -- the recency signature, and
"adjacent_antecedent" would be a nearest-noun reader that the six existing cells could not distinguish. b FALSE and
c FALSE: consistent with v419's slot-encoding finding, and the failure is about the construction, not about number
or recency -- in which case neither this rung nor v419 licenses a number-circuit claim for the family.
SCOPE. One family, rank 1, this budget. A transfer failure here does not tell us what the site DOES encode, only
that it is not a subject-number reader that generalises across interveners.
Smoke: V475_SMOKE=<out.json> (CPU, V475_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_attractor_transfer_v475_result.json"
# fit on the no-intervener cell; evaluate two cells whose intervener conflicts on OPPOSITE interchange sides
FIT_CELLS = ("possessive_adjacent",)
SING_ATTR = "possessive_attractor"                       # intervener singular -> conflict on the BASE side
PLUR_ATTR = "possessive_number_attractor_conflict"       # intervener plural   -> conflict on the DONOR side
NEW_TYPE = (SING_ATTR, PLUR_ATTR)
CONTROL = "verb_preposition_about_for"                   # different readout pair entirely
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_adjacent": FIT_CELLS}
RELAXED = "ALL"
RANK = 1                                                 # fixed and registered in advance, per the DAS protocol
PRIOR = {"v419_sibling_damage": (0.295, 0.555, 0.591, 0.679)}   # attractor, medial, long_simple, their_vs_his
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_attractor_transfer_v475", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("fit_adjacent", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL_CELLS)
    ps = g0.get("per_shape", {}) if ok else {}
    je = lambda c: ps.get(c, {}).get("joint_extraction")
    ur = lambda c: ps.get(c, {}).get("units_recovery")
    fit_v = [je(c) for c in FIT_CELLS]
    sing_v, plur_v = je(SING_ATTR), je(PLUR_ATTR)
    have = ok and all(x is not None for x in fit_v + [sing_v, plur_v])
    a = have and min(fit_v) >= B["joint_min"]
    b = have and sing_v >= B["joint_min"]
    c = have and plur_v >= B["joint_min"]
    d = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in FIT_CELLS + NEW_TYPE)
    cross = ps.get(CONTROL, {}).get("cross_abs_recovery")
    e = ok and cross is not None and abs(cross) <= B["cross_max"]
    return {"pred_a_in_distribution": bool(a), "pred_b_singular_attractor": bool(b),
            "pred_c_plural_attractor": bool(c), "pred_d_units_available": bool(d),
            "pred_e_different_readout": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V475_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V475_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL_CELLS:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
                if cell in cells:                       # ONLY frames A-E enter the fit
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
            for cell in EVAL_CELLS:
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
    result = {"predictions": predictions, "schema": "unit_possessive_attractor_transfer_v475",
              "candidate_id": "corpus.unit_possessive_attractor_transfer_v475", "bars": BARS,
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
