#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v469: the READOUT ENVIRONMENT -- the structural constant every rung so far has held fixed.

WHY THIS RUNG EXISTS. A corpus audit run tonight: 193 of 480 candidate cells end with the IDENTICAL suffix
`, of course,` and roughly 236 end with some comma-bracketed adverbial, so in about half the corpus the readout slot
sits immediately after a comma-adverbial. Tonight varied the CLAUSE TYPE (interrogative with do-support, fronted
adjunct, wh-question matrix) and the VERB MORPHOLOGY (bare form under do-support) and measured what transferred --
but every one of those cells still put the readout after a comma-adverbial. The environment the readout is actually
read in has never been varied, in any rung, in either construction.
Two cells authored for this, both capability-checked on CPU first (A1 32/32 rows kept, 0 dropped, each), and both
with NO punctuation anywhere after the cue:
    fn  `Word spread that the pilot cared a lot`      shared tail ` a lot`
    fq  `Near the ledger the pilot cared again`       shared tail ` again`, and a different tail so a result cannot
                                                      turn on the one word ` a lot`
The fit is the five frames A-E, every one of them comma-adverbial, exactly as in v457; fn and fq are held out.
The known-bad control is at_to, a different mapping, scored on ABSOLUTE recovery. Rank 1; relaxed budget; nothing
counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution   held-out rows of the fitted comma-adverbial frames reach MINIMUM joint extraction
                           joint_min = 0.80. If this fails nothing else is readable.                  prior 90%
  pred_b_nocomma_transfer  BOTH comma-free cells reach 0.80. Capable of failing: held-out clause types reached only
                           0.575 and 0.790 on this same mapping and fit.                              prior 55%
  pred_c_no_loss           the mean over fn and fq is no more than gain = 0.05 below the mean over the held-out A-E
                           cells. Worked example: 0.99 against 1.02 is 0.03 below, TRUE.              prior 45%
  pred_d_different_mapping the control's ABSOLUTE recovery stays at or below cross_max = 0.30, as it did at 0.016
                           and -0.004 in the two earlier rungs using this fit.                        prior 85%
  pred_e_units_available   every evaluated cell except the control reports units_recovery >= unit_min = 0.80, so a
                           failure is attributable to the DIRECTION and not to units that miss the shape. prior 80%
HOW IT READS, fixed in advance: b FAILING would mean the readout environment binds the direction, and that the
corpus's concentration on one suffix is a real limitation on every circuit counted rather than a cosmetic one --
the most consequential outcome available here. b PASSING means the environment is not a factor for this mapping and
the narrow suffix distribution is cosmetic, which would be worth knowing but changes nothing.
SCOPE, AND IT IS THE LESSON OF v465. This is ONE mapping. Tonight the surface-form binding held on about/for and
turned out to be one mapping in four, so whichever way this comes out it needs the same per-mapping replication
before it is stated as a fact about the corpus, and that replication is the rung after this one either way.
Smoke: V469_SMOKE=<out.json> (CPU, V469_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v469_result.json"
# fit on declarative complement frames only; evaluate on clause types the fit never saw
VP = "verb_preposition_"
FIT_CELLS = tuple(VP + c for c in ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"))
NEW_TYPE = tuple(VP + c for c in ("fn_about_for", "fq_about_for"))   # comma-free readout environments
CONTROL = VP + "at_to"                       # different mapping, frame the fit did see
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}               # one fit; the group map is kept so the plan/gate shape is unchanged
RELAXED = "ALL"
PRIOR = {"fg_held_out": 0.575, "fh_held_out": 0.790}   # v457: held-out clause types, same fit and mapping
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v469", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    return {"pred_a_in_distribution": bool(a), "pred_b_nocomma_transfer": bool(b),
            "pred_c_no_loss": bool(c), "pred_d_different_mapping": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V469_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V469_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v469",
              "candidate_id": "corpus.unit_broad_circuit_v469", "bars": BARS,
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
