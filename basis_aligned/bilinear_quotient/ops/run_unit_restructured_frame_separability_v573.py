#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v573: does a NEW FRAME CLASS give distinct behaviours, tested both ways?

WHAT THIS IS NOT. It is not a test of whether my existing directions generalize to a restructured sentence. v337
already settled the mechanism: it changed the matrix clause and the parenthetical while reusing BOTH cue -> token
mappings, and the cell behaved like its FRAME-sharing sibling rather than its CUE-sharing one, so identity is
frame-bound. Running a rung to rediscover that would test nothing, and the audit that prompted this -- every
verb_preposition A2 differs from its A1 only in the sentence-initial adjunct, similarity 0.907 to 0.924 -- is a
measurement of my own receipts that needs no GPU. Row 2 in those twelve receipts licenses "survives an adjunct swap"
and I have said so in the ledger.
WHAT THIS IS. v367 puts the inventory at frame classes times mappings, so if the restructured frame is a new class
it should yield NEW behaviours rather than duplicates of the originals. Two cells were authored and passed CPU
capability at 32/32 on both families: rs_at_to (aimed/appealed -> at/to) and rs_on_about (depended/complained ->
on/about). A third, rs_by_from, screened at A1 5/32 and was DROPPED, not repaired -- the batch had already spent its
one repair on the A2 opener -- and is kept in the tree as a recorded null, since the restructured frame supporting
some mappings and not this one is itself a fact worth not rediscovering.
TESTED BOTH WAYS, BECAUSE SEPARABILITY IS DIRECTIONAL. The corpus record is that interchange direction can carry an
asymmetry, so fitting only on the new cell and checking the original is out of reach would leave the other direction
unmeasured. This rung fits FOUR pools -- each new cell and each original -- and asks in both directions whether the
same-mapping partner is reached.
NOTHING IS COUNTED HERE. A separation makes these CANDIDATE behaviours; the protocol requires a within-family
separability pass before the corpus count moves, and it stays at 139 whatever this returns.
RANK FIXED AT 1 AND REGISTERED.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold     all four pools hold joint extraction at or above joint_min = 0.80, so any non-reach is about
                       the frame and not a failed fit.                                                prior 85%
  pred_b_new_cells_separate NEITHER new cell reaches its same-mapping original: units_recovery < unit_min = 0.80 in
                       the new -> original direction for both. This is what v337 predicts.            prior 80%
  pred_c_originals_separate NEITHER original reaches its restructured counterpart either, so the separation holds in
                       BOTH directions and is not an artifact of which side was fitted.               prior 75%
  pred_d_new_cells_pass_rows both new cells clear A2 at 0.80 and keep P and C same-answer effects at or below
                       p_max = 0.23, so they are behaviours worth having and not merely different.    prior 70%
  pred_e_all_measured  zero pools error.                                                              prior 90%
HOW IT READS, one outcome per branch. b, c and d all true: the restructured frame is a distinct frame class carrying
two mappings, these are two candidate behaviours, and the route to more circuits is authoring FRAMES rather than more
readout pairs inside one frame -- the most useful outcome for the throughput goal. b true and c false: separation is
one-directional, which would be a genuine asymmetry and would mean the new cells are the narrower behaviour rather
than a parallel one. b false: the restructured frame FUSES with the original despite v337, which would contradict the
frame-bound result on a pair where the mappings are held fixed, and would need reporting as a conflict rather than a
tidy confirmation. d false with b and c true: the frame gives distinct behaviours that do not survive their own rows,
so it is not a source of usable circuits.
SCOPE. One frame class, two mappings, four pools, rank 1.
Smoke: V573_SMOKE=<out.json> (CPU, V573_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_restructured_frame_separability_v573_result.json"
# two restructured cells and their same-mapping originals, each fit and each holding the other
VP = "verb_preposition_"
NEW_OLD = {"rs_at_to": "at_to", "rs_on_about": "on_about"}   # restructured cell -> its same-mapping original
SEARCH = {}
for _n, _o in NEW_OLD.items():
    SEARCH[_n] = (_o,)        # fit the new cell, hold the original
    SEARCH[_o] = (_n,)        # and the reverse, because separability is directional
PAIRS = {k: v[0] for k, v in SEARCH.items()}
MAPS = {k: {"fit": (VP + k,), "held": tuple(VP + q for q in v)} for k, v in SEARCH.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v337": "identity is frame-bound", "a2_similarity_audit": "verb_preposition A2 is an adjunct swap, 0.907-0.924", "rs_by_from": "dropped, A1 5/32"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"p_max": 0.23, "strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_restructured_frame_separability_v573", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    ks = list(SEARCH)
    fit_v = {k: fld(k, VP + k, "joint_extraction") for k in ks}
    reach = {k: {q: fld(k, VP + q, "units_recovery") for q in SEARCH[k]} for k in ks}
    hyp = {k: G.get(k, {}).get("hypotheses", {}) for k in ks}
    have = ok and all(v is not None for v in fit_v.values()) \
        and all(x is not None for k in ks for x in reach[k].values())
    a = have and min(fit_v.values()) >= B["joint_min"]
    news, olds = list(NEW_OLD), list(NEW_OLD.values())
    b = have and all(reach[n][NEW_OLD[n]] < B["unit_min"] for n in news)
    c = have and all(reach[o][n] < B["unit_min"] for n, o in NEW_OLD.items())

    def rows_ok(k):
        h = hyp.get(k) or {}
        a2, pp, cc = h.get("A2_extraction"), h.get("P_same_answer_effect"), h.get("C_same_answer_effect")
        return bool(a2 is not None and pp is not None and cc is not None and a2 >= B["joint_min"]
                    and abs(pp) <= B["p_max"] and abs(cc) <= B["p_max"])

    d = have and all(rows_ok(n) for n in news)
    e = ok
    R["per_pool"] = {k: {"fit": fit_v[k], "reach": reach[k], "is_new": k in NEW_OLD,
                         "A2": (hyp.get(k) or {}).get("A2_extraction"),
                         "P": (hyp.get(k) or {}).get("P_same_answer_effect"),
                         "C": (hyp.get(k) or {}).get("C_same_answer_effect"),
                         "rows_pass": rows_ok(k)} for k in ks}
    return {"pred_a_fits_hold": bool(a), "pred_b_new_cells_separate": bool(b),
            "pred_c_originals_separate": bool(c), "pred_d_new_cells_pass_rows": bool(d),
            "pred_e_all_measured": bool(e)}


def main() -> None:
    effective_budget = {"target": TARGET, "min_gain": MIN_GAIN}   # recipe must record what was USED, not the constant
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V573_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V573_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            effective_budget["target"], effective_budget["min_gain"] = tgt, mg
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
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "per_pool": R.get("per_pool"),
              "schema": "unit_restructured_frame_separability_v573",
              "candidate_id": "corpus.unit_restructured_frame_separability_v573", "bars": BARS,
              "recipe": {"pool": pool, "target": effective_budget["target"],
                         "min_gain": effective_budget["min_gain"],
                         "target_module_constant": TARGET, "min_gain_module_constant": MIN_GAIN,
                         "max_units": max_units,
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
