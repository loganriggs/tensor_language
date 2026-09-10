#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v457: does the pooled direction reach a CLAUSE TYPE the fit never saw?

WHAT IS ESTABLISHED. A single rank-1 direction covers seven sentence frames of one mapping at mean 1.015 with no
capacity cost out to seven members (v455), and two frames of a different clause type -- interrogative with a bare
verb under do-support, and a main clause behind a fronted adjunct -- sat among the best-covered cells at 1.026 and
1.018. But BOTH OF THOSE WERE INSIDE THE POOLED FIT. Covering a shape you were fitted on is not transfer, and the
question the corpus actually needs answered is whether these circuits are template-bound: the frames in every pool
so far are similar because they were authored that way.
This rung splits the fit from the evaluation by CLAUSE TYPE. The direction is fitted on frames A-E only -- all
declarative matrix-plus-complement shapes ending in an adverbial -- and then evaluated, with the same units and the
same q, on three kinds of rows:
    KNOWN-GOOD   held-out rows of frames A-E: unseen rows, seen shapes. If this fails nothing else is readable.
    THE QUESTION frames G and H, whose clause type appears NOWHERE in the fit.
    KNOWN-BAD    verb_preposition_at_to, a DIFFERENT mapping in a frame the fit did see. Its readout pair is a
                 different axis, so a direction that recovers it is generic rather than this behaviour's, and this
                 control can fail at any site (lesson 4). It is scored on ABSOLUTE recovery, not on the normalised
                 ratio, because a unit set chosen for about/for may barely move at/to and a ratio with a small
                 denominator is unstable (the absolute-versus-normalised error is a recorded one).
Rank stays at 1; budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution     held-out rows of the FITTED frames A-E reach MINIMUM joint extraction
                             joint_min = 0.80.                                                        prior 90%
  pred_b_new_clause_types    frames G and H EACH reach 0.80 with their clause type absent from the fit.
                             Capable of failing: this instrument has produced 0.506 on a single cell.  prior 60%
  pred_c_no_loss_out_of_type the mean over G and H is no more than gain = 0.05 below the mean over the held-out
                             A-E cells. Worked example: 0.98 against 1.01 is 0.03 below, TRUE; 0.85 against 1.01 is
                             0.16 below, FALSE and the new clause types cost real coverage.            prior 55%
  pred_d_different_mapping   the different-mapping control's ABSOLUTE recovery under q stays at or below
                             cross_max = 0.30, i.e. the direction is this mapping's and not a generic preposition
                             axis. Worked example: 0.10, TRUE; 0.85, FALSE and the transfer above means nothing.
                                                                                                      prior 75%
  pred_e_units_available     every evaluated cell, the control included, reports its own exact-set unit recovery so
                             a transfer failure can be attributed to the DIRECTION rather than to units that do not
                             reach the shape: all non-control cells >= unit_min = 0.80.               prior 80%
SCOPE. One mapping, one construction, two new clause types, this rank and budget. A pass says the direction is not
bound to the shapes it was fitted on, for this mapping. It does not say every mapping behaves this way, and the
next question if it passes is whether the same holds when the fit sees only ONE frame.
Smoke: V457_SMOKE=<out.json> (CPU, V457_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v457_result.json"
# fit on declarative complement frames only; evaluate on clause types the fit never saw
VP = "verb_preposition_"
FIT_CELLS = tuple(VP + c for c in ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"))
NEW_TYPE = tuple(VP + c for c in ("fg_about_for", "fh_about_for"))
CONTROL = VP + "at_to"                       # different mapping, frame the fit did see
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}               # one fit; the group map is kept so the plan/gate shape is unchanged
RELAXED = "ALL"
PRIOR = {"same7_in_pool": 1.015, "fg_in_pool": 1.026, "fh_in_pool": 1.018}   # v455, with G and H inside the fit
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v457", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    fit_v = [je(c) for c in FIT_CELLS]
    new_v = [je(c) for c in NEW_TYPE]
    have = ok and all(x is not None for x in fit_v + new_v)
    a = have and min(fit_v) >= B["joint_min"]
    b = have and all(x >= B["joint_min"] for x in new_v)
    c = have and (sum(new_v) / len(new_v)) >= (sum(fit_v) / len(fit_v)) - B["gain"]
    cross = ps.get(CONTROL, {}).get("cross_abs_recovery")
    d = ok and cross is not None and abs(cross) <= B["cross_max"]
    e = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in FIT_CELLS + NEW_TYPE) \
        and ur(CONTROL) is not None
    return {"pred_a_in_distribution": bool(a), "pred_b_new_clause_types": bool(b),
            "pred_c_no_loss_out_of_type": bool(c), "pred_d_different_mapping": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V457_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V457_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v457",
              "candidate_id": "corpus.unit_broad_circuit_v457", "bars": BARS,
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
