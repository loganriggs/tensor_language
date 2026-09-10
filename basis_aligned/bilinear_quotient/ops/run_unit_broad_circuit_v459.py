#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v459: was it the CLAUSE TYPE, or the cue's surface FORM? Frame G confounded them.

WHAT v457 MEASURED. A direction fitted on frames A-E (declarative matrix-plus-complement, cue in the -ed form)
holds in-distribution at 0.984-1.054 but reaches only 0.575 on frame G and 0.790 on frame H when those are held out
of the fit -- while the SAME head set recovers those rows at 0.997 and 0.960 exact, so the sites carry them and it
is the direction that does not reach them. The different-mapping control was clean (0.017 through q at
units_recovery 1.038).
THE TWO CELLS DIFFER FROM THE FIT IN MORE THAN ONE WAY, which is why v457 named nothing. Frame G is interrogative
AND its cue is the bare verb under do-support (`care`, not `cared`); frame H keeps `cared` and changes only the
material before the cue, and frame H is the milder failure. Two new cells, capability-checked on CPU before this
runner was written (A1 32/32 rows kept, 0 dropped, on each), separate the factors one at a time:
    fi  `The pilot did really care, then,`     BARE form, clause stays declarative -> a failure here is FORM
    fk  `Who noticed the pilot cared, then,`   wh-interrogative matrix, cue keeps -ed  -> a failure here is TYPE
Fit is on frames A-E only, exactly as in v457; fg is re-evaluated in the same run as the anchor. Rank stays at 1;
budget stays relaxed (0.97 / 0.001, 30 units); nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution   held-out rows of the FITTED frames A-E reach MINIMUM joint extraction
                           joint_min = 0.80. If this fails nothing else is readable.                  prior 90%
  pred_b_replicates_fg     frame G re-measured here agrees with v457's 0.575 within tol_pool = 0.05. The anchor
                           check: the comparison is against a number this run reproduces, not one carried across
                           receipts. Worked example: 0.59 differs by 0.015, TRUE.                     prior 85%
  pred_c_bare_form_fails   the bare-form DECLARATIVE cell fi lands BELOW 0.80.                        prior 50%
  pred_d_ed_form_transfers the wh-interrogative cell fk, which keeps the -ed form, reaches 0.80.      prior 50%
  pred_e_units_available   every evaluated cell except the control reports units_recovery >= unit_min = 0.80, so any
                           failure above is attributable to the DIRECTION and not to units that do not reach the
                           shape.                                                                     prior 80%
HOW THE FOUR OUTCOMES READ, fixed before the run so the result cannot be narrated afterwards:
  c TRUE  and d TRUE   the cue's SURFACE FORM is what the direction is bound to; clause type is not the factor.
  c FALSE and d FALSE  the CLAUSE TYPE is the factor; the -ed form does not rescue an interrogative matrix.
  c TRUE  and d FALSE  both matter, and neither may be quoted as the explanation on its own.
  c FALSE and d TRUE   NEITHER factor explains frame G: something specific to that frame is doing the work, and the
                       0.575 stays an unexplained number rather than becoming a mechanism.
SCOPE. One mapping, one construction, this rank and budget. Whatever comes out is a statement about what a rank-1
direction fitted on ONE clause type generalises to, not about the head set, which v457 already showed carries all
of these shapes.
Smoke: V459_SMOKE=<out.json> (CPU, V459_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v459_result.json"
# fit on frames A-E only; evaluate the two factors frame G confounded, one at a time
VP = "verb_preposition_"
FIT_CELLS = tuple(VP + c for c in ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"))
FORM_CELL, TYPE_CELL, ANCHOR = VP + "fi_about_for", VP + "fk_about_for", VP + "fg_about_for"
NEW_TYPE = (ANCHOR, FORM_CELL, TYPE_CELL)
CONTROL = VP + "at_to"                       # different mapping, frame the fit did see
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}
RELAXED = "ALL"
PRIOR = {"fg_held_out": 0.575, "fh_held_out": 0.790}          # v457, both held out of the fit
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v459", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    anchor, form_v, type_v = je(ANCHOR), je(FORM_CELL), je(TYPE_CELL)
    have = ok and all(x is not None for x in fit_v + [anchor, form_v, type_v])
    a = have and min(fit_v) >= B["joint_min"]
    b = have and abs(anchor - PRIOR["fg_held_out"]) <= B["tol_pool"]
    c = have and form_v < B["joint_min"]
    d = have and type_v >= B["joint_min"]
    e = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in FIT_CELLS + NEW_TYPE)
    return {"pred_a_in_distribution": bool(a), "pred_b_replicates_fg": bool(b),
            "pred_c_bare_form_fails": bool(c), "pred_d_ed_form_transfers": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V459_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V459_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v459",
              "candidate_id": "corpus.unit_broad_circuit_v459", "bars": BARS,
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
