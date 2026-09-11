#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v561: spend the three pairs v559 found in the family I had written off.

WHY THESE THREE EXIST AT ALL. v511 labelled adjective_preposition NOT construction-general from a single pair at
0.682, and on that basis I described row 4 as unestablishable here -- in v511's write-up, in v521's, and in what I
told the user. v559 measured five fresh pairs with five distinct partners and found THREE reachable: at_about ->
for_of at 0.967, of_amid -> in_with at 0.919, over_for -> in_about at 0.858, every one inert between -0.008 and
0.001, with the v511 anchor reproducing 0.682 exactly. So the family was written off on one reading and most of the
pairs I tested can in fact carry an informative C row. This rung spends them.
ALL SIX TASKS ARE ADJECTIVE-VERSUS-ADJECTIVE. About eight tasks in this family pit an adjective selecting a
complement against a PASSIVE PARTICIPLE taking a locative adjunct -- hidden beneath, subsumed under, glimpsed past --
which confounds the interchange with adjective-versus-participle; none of those is used here, as a target or a
partner.
TWO CAVEATS REGISTERED IN ADVANCE, BOTH FROM THIS FAMILY'S OWN AUDIT. First, its A2 varies only the copula, `was`
against `seemed`, where verb_preposition's A2 changes the whole matrix frame -- so a passing A2 here is a WEAKER
test of construction generality than the same clause elsewhere, and I will not report it as equivalent. Second, the
shared suffix contains the token ` of` and several tasks in the family answer ` of`; v483 tested that induction-echo
worry directly on of_at against a matched non-of cell and refuted it (0.892 against 0.881), so it is addressed
rather than ignored, and none of the three targets here answers ` of` anyway.
RANK FIXED AT 1 AND REGISTERED. Nothing counted: these are DAS receipts, not new behaviours.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_measured_A1  every group that produced data reaches MINIMUM held-out joint extraction joint_min = 0.80.
                      v559 already showed these three fitting at 0.982 to 1.008.                      prior 88%
  pred_b_measured_A2  every measured group reaches 0.80 on its A2 construction -- the clause the copula caveat
                      applies to.                                                                     prior 75%
  pred_c_measured_P_and_C every measured group has BOTH its P and its partner C same-answer effects at or below
                      p_max = 0.23.                                                                   prior 80%
  pred_d_partners_reached every partner reports units_recovery >= unit_min = 0.80, reproducing v559 within
                      tol_pool = 0.05. Deterministic fits, so a drift means these are not the pairs that census
                      cleared.                                                                        prior 88%
  pred_e_all_measured ZERO groups error.                                                              prior 85%
HOW IT READS. All true: three adjective_preposition targets pass all four hypotheses, the battery reaches a THIRD
construction, and a family I wrote off on one reading yields verified-selective circuits -- which is the concrete
cost of having treated a single pair as a classification. b FALSE: A2 is where the copula caveat bites and a failure
there is about this family's A2 design rather than about the direction. d FALSE: the census pairs did not reproduce
and nothing else is comparable.
SCOPE. Three tasks of thirty-eight, rank 1, one partner each, with the copula caveat attached to any A2 pass.
Smoke: V561_SMOKE=<out.json> (CPU, V561_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_adjprep_battery_three_v561_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "adjective_preposition_"
PAIRS = {"at_about": "for_of", "of_amid": "in_with", "over_for": "in_about"}   # v559: reached at 0.967/0.919/0.858
V559_REACH = {"at_about": 0.967, "of_amid": 0.919, "over_for": 0.858}
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v559_reach": V559_REACH, "v483_of_echo": "refuted on of_at (0.892 vs matched 0.881)"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_adjprep_battery_three_v561", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    c = bool(have) and all(abs(vals(k)[2]) <= B["p_max"] and abs(vals(k)[3]) <= B["p_max"] for k in have)
    reach = {k: ps(k).get(VP + PAIRS[k], {}).get("units_recovery") for k in ks}
    d = bool(have) and all(reach[k] is not None and reach[k] >= B["unit_min"]
                            and abs(reach[k] - V559_REACH[k]) <= B["tol_pool"] for k in have)
    e = len(errored) == 0
    R["per_target"] = {k: {"measured": k in have, "A1": vals(k)[0], "A2": vals(k)[1],
                           "P": vals(k)[2], "C": vals(k)[3],
                           "reach": reach.get(k),
                           "passes": bool(k in have and (reach.get(k) or 0) >= B["unit_min"]
                                          and vals(k)[0] >= B["joint_min"]
                                          and vals(k)[1] >= B["joint_min"]
                                          and abs(vals(k)[2]) <= B["p_max"]
                                          and abs(vals(k)[3]) <= B["p_max"])}
                       for k in ks}
    R["errored_groups"] = errored
    return {"pred_a_measured_A1": bool(a), "pred_b_measured_A2": bool(b),
            "pred_c_measured_P_and_C": bool(c), "pred_d_partners_reached": bool(d),
            "pred_e_all_measured": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V561_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V561_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
              "errored_groups": R.get("errored_groups"), "schema": "unit_adjprep_battery_three_v561",
              "candidate_id": "corpus.unit_adjprep_battery_three_v561", "bars": BARS,
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
