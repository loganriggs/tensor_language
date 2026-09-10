#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v531: is an unreachable control a property of the PAIR, or of the partner task?

WHAT v529 LEFT. Four of five pairs in the largest family had a control the fitted nodes reach; the one failure was
by_from -> at_over at 0.688, and the same fitted task reaches at_to at 0.932 (v527). I concluded that availability
is a property of the PAIR rather than of the fitted task, which is right as far as it goes -- but it does not
distinguish two very different situations. Either at_over is a HARD PARTNER for everyone, in which case the rule is
simply "some tasks make poor controls" and a census should avoid them; or by_from and at_over are specifically
mismatched, in which case each pair really does need its own check and no shortlist of good partners will do.
THREE MORE FITS, EACH EVALUATING at_over AND ITS OWN KNOWN-GOOD PARTNER, so the comparison is WITHIN-FIT and a low
at_over number cannot be blamed on a weak fit:
    about_for  -> at_over   alongside at_to     (known 0.859, v527)
    in_to      -> at_over   alongside by_with   (known 0.907, v529)
    from_for   -> at_over   alongside at_into   (known 0.857, v529)
Every pair is vocabulary-disjoint: at_over is (` at`, ` over`) and none of about_for, in_to or from_for shares a
token with it. Standard budget; rank 1; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold        all three fitted tasks reach MINIMUM held-out joint extraction joint_min = 0.80.
                                                                                                      prior 85%
  pred_b_known_partners_hold each fit reaches unit_min = 0.80 on its OWN known-good partner, reproducing 0.859,
                          0.907 and 0.857 within tol_pool_wide = 0.15. THIS IS THE LOAD-BEARING CONTROL: without it
                          a low at_over number says nothing, because the fit itself might be poor.     prior 85%
  pred_c_at_over_unreachable at_over stays BELOW 0.80 from all three fits.                            prior 55%
  pred_d_at_over_consistent the three at_over readings span at most tol_pool_wide = 0.15 between highest and
                          lowest, i.e. at_over behaves the same way toward every fitted task rather than varying by
                          partner.                                                                    prior 60%
  pred_e_matches_by_from  the mean of the three at_over readings is within tol_pool_wide of by_from's 0.688, so the
                          earlier failure is reproduced rather than being an oddity of that one fit.   prior 60%
HOW IT READS, one outcome per branch. b, c, d and e TRUE: at_over is simply a HARD PARTNER -- some tasks make poor
controls, the rule is to avoid them, and v529's per-pair framing overstates the problem. c FALSE: at_over is
reachable from some fits and not others, so the mismatch really is pair-specific and every four-hypothesis claim
needs its own control check, exactly as v529 said. d FALSE with c TRUE: at_over is unreachable everywhere but by
varying amounts, which is a weaker version of the hard-partner reading and I would report the spread rather than a
verdict. b FALSE: the fits are the problem and nothing else here is readable -- I will not interpret c, d or e.
SCOPE. One partner task, three fits, rank 1, standard budget.
Smoke: V531_SMOKE=<out.json> (CPU, V531_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_partner_property_v531_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "verb_preposition_"
PARTNER = "at_over"                                   # the partner under test (TARGET collides with the greedy budget constant)
KNOWN = {"about_for": "at_to", "in_to": "by_with", "from_for": "at_into"}   # each fit's known-good partner
PAIRS = {k: PARTNER for k in KNOWN}
MAPS = {k: {"fit": (VP + k,), "held": (VP + PARTNER, VP + KNOWN[k])} for k in KNOWN}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"by_from_at_over": 0.688, "known": {"about_for": 0.859, "in_to": 0.907, "from_for": 0.857}}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_partner_property_v531", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    ks = list(KNOWN)
    fit_v = [fld(k, VP + k, "joint_extraction") for k in ks]
    over = {k: fld(k, VP + PARTNER, "units_recovery") for k in ks}
    known = {k: fld(k, VP + KNOWN[k], "units_recovery") for k in ks}
    have = ok and all(x is not None for x in fit_v) \
        and all(v is not None for v in over.values()) and all(v is not None for v in known.values())
    a = have and min(fit_v) >= B["joint_min"]
    b = have and all(known[k] >= B["unit_min"] for k in ks) \
        and all(abs(known[k] - PRIOR["known"][k]) <= B["tol_pool_wide"] for k in ks)
    c = have and all(over[k] < B["unit_min"] for k in ks)
    d = have and (max(over.values()) - min(over.values())) <= B["tol_pool_wide"]
    e = have and abs(sum(over.values()) / len(over) - PRIOR["by_from_at_over"]) <= B["tol_pool_wide"]
    return {"pred_a_fits_hold": bool(a), "pred_b_known_partners_hold": bool(b),
            "pred_c_at_over_unreachable": bool(c), "pred_d_at_over_consistent": bool(d),
            "pred_e_matches_by_from": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V531_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V531_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_partner_property_v531",
              "candidate_id": "corpus.unit_partner_property_v531", "bars": BARS,
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
