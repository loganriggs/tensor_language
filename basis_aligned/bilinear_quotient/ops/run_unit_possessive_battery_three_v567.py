#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v567: spend possessive's three reachable pairs, with the MAPPING-SHARING one as the registered variable.

WHY. v563 censused possessive -- the family v511 scored lowest of six and called NOT construction-general -- and
found THREE reachable partners, not the at most two I had registered: number_his_their -> pronoun_mine_yours at
0.887, pronoun_person_ours_theirs -> number_attractor_conflict at 0.867, and pronoun_number_hers_theirs ->
number_its_their at 0.850. The v509 anchor reproduced 0.729 exactly. None has been spent, so the family has no
four-hypothesis receipt.
THE ONE ROW I AM NOT AVERAGING IN. Two of those partners leak +0.004 and +0.017; the third, number_its_their, leaks
+0.151 -- ten to forty times the others, though still inside the 0.3 bar. It is also the ONE target-partner pair
here whose two sides share the NUMBER mapping: hers/theirs against its/their. The corpus record is that margins
degrade against mapping-sharing siblings rather than against control-set size, so this pair is the predicted hard
case, and folding it into an average over three would hide exactly the thing worth measuring. So the two clean pairs
carry a to d, and the mapping-sharing pair gets its own predicate. Either answer is informative: if it passes all
four despite the leak, the 0.3 bar is admitting pairs that still support clean rows; if it fails C, the
mapping-sharing prediction is confirmed on a pair chosen in advance for that property.
RANK FIXED AT 1 AND REGISTERED. Nothing counted: DAS receipts are not new behaviours.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_clean_A1     BOTH clean pairs hold held-out joint extraction at or above joint_min = 0.80. v563 fitted them
                      at 0.995 and 0.975.                                                             prior 90%
  pred_b_clean_A2     BOTH clean pairs reach 0.80 on their A2 construction.                           prior 75%
  pred_c_clean_P_and_C BOTH clean pairs have their P and their partner C same-answer effects at or below
                      p_max = 0.23, which with a and b makes them possessive circuits passing all four. prior 75%
  pred_d_clean_reach  BOTH clean partners report units_recovery >= unit_min = 0.80, reproducing v563 within
                      tol_pool = 0.05.                                                                prior 90%
  pred_e_sharing_also_passes the MAPPING-SHARING pair hers_theirs ALSO passes all four rows on the same bars. This is
                      the registered variable and I expect it to be the one that breaks: the leak is 0.151 against
                      0.004 and 0.017.                                                                prior 40%
HOW IT READS, one outcome per branch. a to d true: possessive gets its first tasks passing all four and the battery
reaches a FOURTH construction, in the family v511 ranked last. e FALSE with a to d true: the mapping-sharing partner
is the one that fails, on a pair named in advance for that property, which is the cleanest confirmation the corpus
has that sharing a mapping -- not control-set size -- is what degrades these rows. e TRUE: a partner can leak 0.151
and still support clean P and C rows, so the leak bar is not the thing separating good pairs from bad, and the next
question is what is. c FALSE: possessive is a family where row 4 is measurable but not passed, which is a different
claim from unreachable and a more interesting one.
SCOPE. One family, three pairs, rank 1, one partner each; not a claim about possessives in general.
Smoke: V567_SMOKE=<out.json> (CPU, V567_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_battery_three_v567_result.json"
# one fit, four hypotheses read on the same rank-1 direction
VP = "possessive_"
PAIRS = {"number_his_their": "pronoun_mine_yours", "pronoun_person_ours_theirs": "number_attractor_conflict",
         "pronoun_number_hers_theirs": "number_its_their"}
CLEAN = ("number_his_their", "pronoun_person_ours_theirs")   # partners leak +0.004 and +0.017
SHARING = "pronoun_number_hers_theirs"                       # partner leaks +0.151 and SHARES the number mapping
V563_REACH = {"number_his_their": 0.887, "pronoun_person_ours_theirs": 0.867, "pronoun_number_hers_theirs": 0.850}
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v563_reach": V563_REACH, "v563_leak": {"clean": (0.004, 0.017), "sharing": 0.151}, "v509_anchor": 0.729}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_battery_three_v567", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    reach = {k: ps(k).get(VP + PAIRS[k], {}).get("units_recovery") for k in ks}

    def four(k):
        return bool(k in have and vals(k)[0] >= B["joint_min"] and vals(k)[1] >= B["joint_min"]
                    and abs(vals(k)[2]) <= B["p_max"] and abs(vals(k)[3]) <= B["p_max"]
                    and (reach.get(k) or 0) >= B["unit_min"])

    cl = [k for k in CLEAN if k in have]
    a = len(cl) == len(CLEAN) and all(vals(k)[0] >= B["joint_min"] for k in cl)
    b = len(cl) == len(CLEAN) and all(vals(k)[1] >= B["joint_min"] for k in cl)
    c = len(cl) == len(CLEAN) and all(abs(vals(k)[2]) <= B["p_max"] and abs(vals(k)[3]) <= B["p_max"] for k in cl)
    d = len(cl) == len(CLEAN) and all(reach[k] is not None and reach[k] >= B["unit_min"]
                                      and abs(reach[k] - V563_REACH[k]) <= B["tol_pool"] for k in cl)
    e = four(SHARING)
    R["per_target"] = {k: {"measured": k in have, "A1": vals(k)[0], "A2": vals(k)[1],
                           "P": vals(k)[2], "C": vals(k)[3], "reach": reach.get(k),
                           "mapping_sharing": k == SHARING, "passes": four(k)}
                       for k in ks}
    R["errored_groups"] = errored
    return {"pred_a_clean_A1": bool(a), "pred_b_clean_A2": bool(b),
            "pred_c_clean_P_and_C": bool(c), "pred_d_clean_reach": bool(d),
            "pred_e_sharing_also_passes": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V567_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V567_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
              "errored_groups": R.get("errored_groups"), "schema": "unit_possessive_battery_three_v567",
              "candidate_id": "corpus.unit_possessive_battery_three_v567", "bars": BARS,
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
