#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v461: if the binding is the cue's surface FORM, does ONE bare-form cell in the fit repair the rest?

WHAT v459 SETTLED. A direction fitted on frames A-E (all -ed cues) reaches 0.931 on a wh-interrogative that keeps
`cared` and only 0.658 on a declarative that switches to bare `care` -- with units_recovery 0.898-1.012 throughout,
so the sites carry every shape and it is the direction that is bound. Across all eight cells the split is clean:
every -ed cell 0.79-1.05, both bare cells 0.575 and 0.658. The factor is the cue's SURFACE FORM, and the clause
type is not it.
THE USEFUL CONSEQUENCE IS A REPAIR, AND IT IS FALSIFIABLE. v455 showed a pooled fit absorbs an added shape at no
cost to the shapes already in it (seven frames of one mapping at 1.015, against 1.023 for five). So adding ONE
bare-form cell to the fit should lift the OTHER bare-form cell -- which stays out of the fit -- to the bar. If it
does, the recipe for breadth is one cell per surface form rather than more sentence templates, which is a cheaper
and more general instruction than anything the frame work has produced so far.
    FIT      frames A-E (-ed) + fi (bare, declarative)          one bare-form cell added, nothing else changed
    HELD OUT fg  bare verb under do-support, interrogative      was 0.575 fitted on -ed only
             fh  fronted adjunct, -ed cue                       was 0.790 fitted on -ed only
             fk  wh-interrogative, -ed cue                      was 0.931 fitted on -ed only
    CONTROL  at_to, a different mapping in a frame the fit saw, scored on ABSOLUTE recovery
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution  held-out rows of the fitted frames reach MINIMUM joint extraction joint_min = 0.80.
                                                                                                      prior 90%
  pred_b_fg_repaired      fg, still absent from the fit, reaches 0.80 -- up from 0.575. This is the repair.
                                                                                                      prior 65%
  pred_c_fh_unchanged     fh, which already had the -ed form and so has nothing to gain from a bare-form cell,
                          stays within tol_pool = 0.05 of its 0.790. This is the SPECIFICITY clause: a repair that
                          also moves a cell it should not touch is a change in the fit's size or seed, not the
                          surface-form account. Worked example: 0.81 differs by 0.020, TRUE; 0.95 differs by 0.160,
                          FALSE and the gain is not specific to the form.                             prior 60%
  pred_d_no_cost_to_fitted the mean over the held-out rows of frames A-E stays within tol_pool of v457's 1.023, so
                          the added cell does not buy fg at the expense of the frames already covered.  prior 80%
  pred_e_units_available  every evaluated cell except the control reports units_recovery >= unit_min = 0.80.
                                                                                                      prior 85%
HOW THE OUTCOMES READ, fixed in advance: b TRUE with c TRUE is the surface-form account confirmed and the breadth
recipe established. b TRUE with c FALSE means the fit got better at everything and the gain says nothing about
form. b FALSE means one cell is not enough, and the open question becomes how many -- not whether the account is
right, which v459 settled on its own evidence.
SCOPE. One mapping, one construction, this rank and budget.
Smoke: V461_SMOKE=<out.json> (CPU, V461_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v461_result.json"
# fit on frames A-E PLUS one bare-form cell; the other bare-form cell stays out
VP = "verb_preposition_"
BASE_AE = tuple(VP + c for c in ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"))
FORM_CELL = VP + "fi_about_for"                      # bare form, declarative -- ADDED TO THE FIT here
FIT_CELLS = BASE_AE + (FORM_CELL,)
FG, FH, FK = VP + "fg_about_for", VP + "fh_about_for", VP + "fk_about_for"
NEW_TYPE = (FG, FH, FK)
CONTROL = VP + "at_to"
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}
RELAXED = "ALL"
PRIOR = {"fg_ed_only": 0.575, "fh_ed_only": 0.790, "fk_ed_only": 0.931, "ae_mean": 1.023}   # v457/v459
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v461", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("fit_AE", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL_CELLS)
    ps = g0.get("per_shape", {}) if ok else {}
    je = lambda c: ps.get(c, {}).get("joint_extraction")
    ur = lambda c: ps.get(c, {}).get("units_recovery")
    ae_v = [je(c) for c in BASE_AE]
    fit_v = [je(c) for c in FIT_CELLS]
    fg_v, fh_v = je(FG), je(FH)
    have = ok and all(x is not None for x in fit_v + [fg_v, fh_v])
    a = have and min(fit_v) >= B["joint_min"]
    b = have and fg_v >= B["joint_min"]
    c = have and abs(fh_v - PRIOR["fh_ed_only"]) <= B["tol_pool"]
    d = have and abs(sum(ae_v) / len(ae_v) - PRIOR["ae_mean"]) <= B["tol_pool"]
    e = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in FIT_CELLS + NEW_TYPE)
    return {"pred_a_in_distribution": bool(a), "pred_b_fg_repaired": bool(b),
            "pred_c_fh_unchanged": bool(c), "pred_d_no_cost_to_fitted": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V461_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V461_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v461",
              "candidate_id": "corpus.unit_broad_circuit_v461", "bars": BARS,
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
