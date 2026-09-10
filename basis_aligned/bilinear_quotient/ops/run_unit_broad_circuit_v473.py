#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v473: my comma-free result was CONFOUNDED. Separating the comma from the distance.

THE RED-TEAM FINDING THIS ANSWERS. An adversarial audit of this family reported that the fit site is the final
input token and that it is a COMMA in 111 of 119 cells -- the eight exceptions being exactly the fn_/fq_ cells I
authored, the ones that failed to transfer. It also reported that cue-to-readout distance is four or five tokens in
109 of 119 cells, and that the two failing frames sit at three (` a lot`) and two (` again`). SO v469 MOVED TWO
THINGS AT ONCE: it removed the comma AND shortened the distance, and its 0.662 / 0.622 cannot say which did the
work. That is the same class of error I logged against v457 earlier tonight, and I did not see it in my own design.
Two cells, capability-checked on CPU before this runner was written (A1 32/32 rows kept, 0 dropped, each), each
holding the comma and moving ONE factor:
    fu  `... cared, last winter,`                               comma kept, distance kept at five; only the
                                                                parenthetical's CLASS changes, from a speaker-stance
                                                                adverbial to a temporal NP. `, last winter,` is four
                                                                tokens, the same as `, of course,`.
    fw  `... cared, as everyone in the village already knew,`   comma kept, register kept; only DISTANCE changes,
                                                                from five tokens to ten.
Fit is the five frames A-E, all `, of course,`; fu and fw are held out. Control is at_to on absolute recovery.
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_in_distribution  held-out rows of the fitted frames reach MINIMUM joint extraction joint_min = 0.80.
                                                                                                      prior 90%
  pred_b_class_transfers  fu reaches 0.80 -- changing the parenthetical's syntactic class at constant comma and
                          constant distance costs nothing.                                            prior 60%
  pred_c_distance_transfers fw reaches 0.80 -- stretching the distance from five to ten at constant comma costs
                          nothing.                                                                    prior 45%
  pred_d_different_mapping the control's ABSOLUTE recovery stays at or below cross_max = 0.30.        prior 85%
  pred_e_units_available  both new cells report units_recovery >= unit_min = 0.80, so a failure is the direction and
                          not units that miss the shape. One of the six cells in v471 came in at 0.786 and was
                          reported, so this bar does bite.                                            prior 75%
HOW IT READS, fixed in advance. b TRUE and c TRUE: neither the adverbial class nor the distance matters, and v469's
failure was specifically the loss of the comma. b TRUE and c FALSE: the direction is a FIXED-OFFSET read of the verb
slot, v469 measured distance rather than punctuation, and the name "readout environment" was wrong. b FALSE: the
direction is bound to the closed stance-adverbial set itself, which is narrower than either story and would mean
about/for's direction keys on the specific words `of course` and their kin. b FALSE and c FALSE together would say
the whole tail is fragile and only the exact fitted suffix works.
SCOPE. about/for only -- the mapping that IS environment-bound; the other three transfer at 0.874-1.001 (v471), so
this rung is diagnosing the exception, not the family.
Smoke: V473_SMOKE=<out.json> (CPU, V473_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v473_result.json"
# fit on declarative complement frames only; evaluate on clause types the fit never saw
VP = "verb_preposition_"
FIT_CELLS = tuple(VP + c for c in ("about_for", "fb_about_for", "fc_about_for", "fd_about_for", "fe_about_for"))
NEW_TYPE = tuple(VP + c for c in ("fu_about_for", "fw_about_for"))   # comma HELD; class, then distance
CONTROL = VP + "at_to"                       # different mapping, frame the fit did see
EVAL_CELLS = FIT_CELLS + NEW_TYPE + (CONTROL,)
GROUPS = {"fit_AE": FIT_CELLS}               # one fit; the group map is kept so the plan/gate shape is unchanged
RELAXED = "ALL"
PRIOR = {"fn_nocomma": 0.662, "fq_nocomma": 0.622, "fitted_mean": 1.023}   # v469, same fit and mapping
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v473", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    class_v, dist_v = je(VP + "fu_about_for"), je(VP + "fw_about_for")
    have = ok and all(x is not None for x in fit_v + [class_v, dist_v])
    a = have and min(fit_v) >= B["joint_min"]
    b = have and class_v >= B["joint_min"]
    c = have and dist_v >= B["joint_min"]
    cross = ps.get(CONTROL, {}).get("cross_abs_recovery")
    d = ok and cross is not None and abs(cross) <= B["cross_max"]
    e = ok and all(ur(x) is not None and ur(x) >= B["unit_min"] for x in NEW_TYPE)
    return {"pred_a_in_distribution": bool(a), "pred_b_class_transfers": bool(b),
            "pred_c_distance_transfers": bool(c), "pred_d_different_mapping": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V473_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V473_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v473",
              "candidate_id": "corpus.unit_broad_circuit_v473", "bars": BARS,
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
