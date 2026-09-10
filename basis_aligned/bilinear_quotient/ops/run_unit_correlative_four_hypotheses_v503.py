#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v503: the DAS follow-up as specified -- all FOUR hypotheses on the rank-1 subspace, not just A1.

WHY THIS IS ONLY NOW RUNNABLE. The standing protocol asks that a DAS subspace pass A1, A2, P and C, with the rank
fixed in advance. On correlative_pair.both_vs_neither the A1 side has been measured repeatedly, but C was not
informative: every correlative cell shares a readout token with (` and`, ` nor`), so the family control leaked 0.259
(v497), and the six cross-construction candidates were all below the units floor, which made their near-zero
readings uninformative rather than clean (v499). v501 fixed that by authoring a same-construction cell with a
disjoint readout pair -- reached at units 0.973 and inert at -0.021 -- so a C row can now fail for a reason.
RANK IS FIXED AT 1 AND REGISTERED HERE IN ADVANCE, per the protocol; a null is not permission to raise it. The four
hypotheses are read on the ONE fitted direction, and each is measured with the instrument that suits it:
    A1  held-out rows of the fitted cell        EXTRACTION, because the interchange has a margin to move
    A2  the cell's own second construction      EXTRACTION, same reason
    P   the answer-preserving edit              REMOVAL, because P's two sides carry the SAME answer, so an
                                                extraction ratio has a degenerate denominator. This project has
                                                already recorded that P is an interchange control only and that its
                                                removal tracks the same-side A1 removal, so P is read as damage
                                                under the direction and not as a recovery fraction.
    C   the v501 disjoint cell                  REMOVAL, same instrument and same quantity as P, so their bars are
                                                comparable to each other and to the corpus row-4 bar.
No bar here is carried across instruments: A1 and A2 are extraction against joint_min, P and C are ce_ub975 against
the corpus c_ub_max. Nothing is counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_A1_holdout    held-out A1 rows reach MINIMUM joint extraction joint_min = 0.80. Measured at 1.001 three
                       times on this deterministic fit, so this is the instrument check.              prior 90%
  pred_b_A2_carries    the SAME direction reaches 0.80 on the cell's A2 construction. This is the hypothesis the
                       family has never had tested on the direction, only on the site.                prior 55%
  pred_c_P_spared      the P rows' ce_ub975 under removal stays at or below c_ub_max = 0.01.          prior 55%
  pred_d_C_spared      the v501 disjoint cell's ce_ub975 under removal stays at or below c_ub_max = 0.01. Capable
                       of failing for a reason now, which is the whole point of authoring it.         prior 70%
  pred_e_units_available every evaluated cell reports units_recovery >= unit_min = 0.80, so any failure above is the
                       DIRECTION and not units that miss the shape.                                   prior 80%
HOW IT READS, one outcome per branch. All four true: the rank-1 subspace is SELECTIVE on this DAS target by the
standing definition, and that is the first time this corpus can say so with a C that is inert for an informative
reason. b FALSE alone: the direction carries the variable in the fitted construction but not in the second one, so
it is construction-bound and the DAS pass fails on generality, not on specificity. c FALSE alone: the direction is
not answer-preserving-edit-safe, which would mean it carries something the P rewrite disturbs. d FALSE alone: the
direction damages a same-site behaviour with disjoint tokens, so it is not specific to this variable and the row-4
claim fails outright. Two or more false: reported as such, no single story.
SCOPE. One mapping, rank 1, this budget. A pass is a statement about correlative_pair's direction, not the family.
Smoke: V503_SMOKE=<out.json> (CPU, V503_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_correlative_four_hypotheses_v503_result.json"
# one fit, four hypotheses read on the same rank-1 direction
CANON = "correlative_pair"
DISJOINT_C = "correlative_disjoint_either_not"      # v501: reached at 0.973, inert at -0.021
MAPS = {"fit_canonical": {"fit": (CANON,), "held": (DISJOINT_C,)}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v495_A1": 1.001, "v501_disjoint_units": 0.973, "v501_disjoint_abs": -0.021}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_correlative_four_hypotheses_v503", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    g0 = G.get("fit_canonical", {})
    ok = bool(g0) and "error" not in g0 and set(g0.get("per_shape", {})) == set(EVAL["fit_canonical"])
    ps = g0.get("per_shape", {}) if ok else {}
    hyp = g0.get("hypotheses", {}) if ok else {}
    a1_v = ps.get(CANON, {}).get("joint_extraction")
    a2_v = hyp.get("A2_extraction")
    p_ub = (hyp.get("P_damage") or {}).get("ce_ub975")
    c_ub = ps.get(DISJOINT_C, {}).get("C_ub975")
    c_dmg = (ps.get(DISJOINT_C, {}).get("A1_damage") or {}).get("ce_ub975")
    have = ok and all(x is not None for x in (a1_v, a2_v, p_ub, c_dmg))
    a = have and a1_v >= B["joint_min"]
    b = have and a2_v >= B["joint_min"]
    c = have and abs(p_ub) <= B["c_ub_max"]
    d = have and abs(c_dmg) <= B["c_ub_max"]
    e = ok and ps.get(CANON, {}).get("units_recovery", 0) >= B["unit_min"] \
        and ps.get(DISJOINT_C, {}).get("units_recovery", 0) >= B["unit_min"] \
        and (hyp.get("A2_units_recovery") or 0) >= B["unit_min"]
    return {"pred_a_A1_holdout": bool(a), "pred_b_A2_carries": bool(b),
            "pred_c_P_spared": bool(c), "pred_d_C_spared": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V503_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V503_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
                if cell == CANON:                       # the other two hypotheses, read on the same direction
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
            hyp = {"A2_units_recovery": e_a2,
                   "A2_extraction": round(ext(p_a2, units, q=q_joint) / e_a2, 3) if abs(e_a2) > 1e-6 else None,
                   "P_damage": dmg(p_p, units, q_joint, mu_joint),
                   "P_units_recovery": ext(p_p, units),
                   "n_a2": len(p_a2.base_batch.row_ids), "n_p": len(p_p.base_batch.row_ids)}
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "hypotheses": hyp}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_correlative_four_hypotheses_v503",
              "candidate_id": "corpus.unit_correlative_four_hypotheses_v503", "bars": BARS,
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
