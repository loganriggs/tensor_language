#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v497: v495's control shared a token with its own answer. Measure the leak instead of asserting it.

WHAT v495 LEFT AS A REGISTERED WEAKNESS. On the DAS target correlative_pair.both_vs_neither, the canonical rank-1
direction reached 0.882 on a plain long frame and only 0.686 once a DISCHARGED `either ... or` was interposed --
a real result, but its different-mapping control was correlative_or_and, which came in at 0.283 against a 0.30 bar
AND SHARES THE TOKEN ` and` WITH THIS MAPPING'S BASE ANSWER. Lesson 3 is explicit that a related control masks a
site, and "related" plainly includes sharing an answer token. I flagged it in the receipt rather than leaving it
implicit; this rung repairs it, and does so by MEASURING the leak rather than by swapping the control and hoping.
The same fit is re-run -- same cell, same seed, same budget, deterministic -- with TWO controls side by side:
    correlative_or_and     different mapping, SAME family, shares ` and` with the base answer   (partially related)
    verb_preposition_at_to different construction entirely, answer vocabulary ` at`/` to`       (disjoint)
Both are answer-changing behaviours, so both are capable of failing at a carrying site: lesson 4's question -- what
is this control's ceiling at the strongest site? -- has the answer "about 1.0" for each, which is what makes the
comparison meaningful rather than two saturating numbers.
Rank stays at 1, as registered on the parent rung. Nothing here is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_reproduces_v495     the canonical cell's own held-out rows and the discharged-correlative cell both land
                             within tol_pool = 0.05 of v495's 1.001 and 0.686. The fit is deterministic and this is
                             the instrument check; if it fails, the controls measured here belong to a different
                             direction than the one v495 reported and nothing below is comparable.    prior 85%
  pred_b_disjoint_inert      the DISJOINT control's absolute recovery stays at or below cross_max = 0.30.  prior 90%
  pred_c_related_leaks_more  the partially related control exceeds the disjoint one by at least gain = 0.05.
                             Worked example: 0.28 against 0.05 is 0.23, TRUE and the shared token is doing
                             measurable work; 0.28 against 0.26 is 0.02, FALSE and the 0.283 was NOT about the
                             shared token, which would mean my flag on v495 was wrong and the control is simply
                             a related BEHAVIOUR.                                                     prior 65%
  pred_d_units_available     every evaluated cell reports units_recovery >= unit_min = 0.80, controls included --
                             a control whose sites are unreachable cannot be inert for an informative reason. prior 75%
  pred_e_related_under_bar   the partially related control still lands at or below 0.30, i.e. v495's control did
                             not actually fail its bar on a re-fit. Registered because 0.283 is close enough to 0.30
                             that a re-run could put it over, and if it does then v495's pred_e was a near miss and
                             I will say so.                                                           prior 75%
HOW IT READS, one outcome per branch. c TRUE with e TRUE: the shared token leaks measurably, v495's control was weak
but held, and the correlative family needs a disjoint-vocabulary control from here on. c TRUE with e FALSE: the leak
is large enough to push the control over its bar on a re-fit, and v495's control row should be read as failed.
c FALSE: the 0.283 was not about the shared token at all -- my flag was wrong, the number reflects a related
BEHAVIOUR rather than a related TOKEN, and the fix is a different control for a different reason.
SCOPE. One fit of one mapping. This rung measures the control, not the circuit.
Smoke: V497_SMOKE=<out.json> (CPU, V497_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_correlative_control_v497_result.json"
# one fit, two controls side by side: one sharing an answer token with the target, one fully disjoint
OWN = "correlative_pair"
KEY = "correlative_open_rec"                 # the discharged-correlative cell v495 measured at 0.686
RELATED = "correlative_or_and"               # different mapping, same family, shares ' and' with the base answer
DISJOINT = "verb_preposition_at_to"          # different construction, disjoint answer vocabulary
SIBS = (KEY, RELATED, DISJOINT)
CTRL = DISJOINT
MAPS = {"fit_one": {"fit": (OWN,), "held": SIBS, "control": DISJOINT}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1
PRIOR = {"v495_canonical_own": 1.001, "v495_discharged": 0.686, "v495_related_control": 0.283}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_correlative_control_v497", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(MAPS[k]["control"], {}).get("cross_abs_recovery")
    own_v, key_v = je("fit_one", OWN), je("fit_one", KEY)
    ps0 = G.get("fit_one", {}).get("per_shape", {})
    rel_v = ps0.get(RELATED, {}).get("cross_abs_recovery")
    dis_v = ps0.get(DISJOINT, {}).get("cross_abs_recovery")
    have = ok and all(x is not None for x in (own_v, key_v, rel_v, dis_v))
    a = have and abs(own_v - PRIOR["v495_canonical_own"]) <= B["tol_pool"] \
        and abs(key_v - PRIOR["v495_discharged"]) <= B["tol_pool"]
    b = have and abs(dis_v) <= B["cross_max"]
    c = have and (abs(rel_v) - abs(dis_v)) >= B["gain"]
    d = ok and all(ur("fit_one", cc) is not None and ur("fit_one", cc) >= B["unit_min"]
                   for cc in (OWN,) + SIBS)
    e = have and abs(rel_v) <= B["cross_max"]
    return {"pred_a_reproduces_v495": bool(a), "pred_b_disjoint_inert": bool(b),
            "pred_c_related_leaks_more": bool(c), "pred_d_units_available": bool(d),
            "pred_e_related_under_bar": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V497_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V497_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            for cell in EVAL[gname]:
                m = importlib.import_module(f"circuit_fast_screen_candidate_{cell}")
                a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
                per_shape_rows[cell] = (cut(held_half(a1)), cut(held_half(cc)))
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
                backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
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
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_correlative_control_v497",
              "candidate_id": "corpus.unit_correlative_control_v497", "bars": BARS,
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
