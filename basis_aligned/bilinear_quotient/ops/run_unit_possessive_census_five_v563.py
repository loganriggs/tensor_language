#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v563: the FOURTH family censused on the same predicate, and the one v511 scored lowest.

WHY. v511 read one pair per family and called three of six construction-general. Three families have since been
censused with five fresh pairs and five distinct partners each, and v511's label was MISLEADING FOR TWO of them:
noun_preposition, called general at 0.913, measures one of five (v555); adjective_preposition, called NOT general at
0.682, measures three of five (v559); verb_particle, called not general at 0.677, measures one of six and is the one
label that held. So the reading per family is a RATE, and a single pair has been a poor guide to it. possessive is
v511's LOWEST score -- 0.729, from v509 fitting possessive_adjacent and reaching possessive_disjoint_my_your -- and
is still one of two families resting on one pair. This censuses it.
DESIGN, IDENTICAL TO v559 SO THE RATES ARE COMPARABLE. Five new targets, five DISTINCT partners, every partner
vocabulary-disjoint from its target: gender -> person_our_your, number_his_their -> pronoun_mine_yours,
number_its_their -> gender, pronoun_number_hers_theirs -> number_its_their, pronoun_person_ours_theirs ->
number_attractor_conflict. Reach is measured WITHIN the family, as in every prior census, because a cross-family
partner would not be comparable to the one-of-five and three-of-five numbers. The v509 anchor pair runs alongside and
is EXCLUDED from the rate.
WHY CORRELATIVE IS NOT HERE. It is the other one-pair family, and it cannot take this design: its readouts are a
small closed set of conjunctions -- and, but, or, nor -- so within-family vocabulary-disjoint pairs exist for only
two targets and both would need the SAME partner. That is a fact about the family's readout inventory, not a gap I
can close by authoring one more cell, and it is why possessive goes first.
RANK FIXED AT 1 AND REGISTERED. Nothing counted: reach and inertness are not new behaviours.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold      every group that produced data holds joint extraction at or above joint_min = 0.80, so a low
                        reach is about the partner and not a failed fit.                              prior 88%
  pred_b_some_reachable at least one NEW pair reports units_recovery >= unit_min = 0.80.              prior 75%
  pred_c_reached_are_inert every NEW pair that reaches also has |same-answer effect| <= p_max = 0.23, so reach is
                        not bought by disturbing the partner.                                         prior 85%
  pred_d_rate_low       AT MOST TWO of the five new pairs reach unit_min -- the same bar v559 failed and v555/v557
                        met. I put this near even: v511 scored this family lowest, which on the noun_preposition
                        precedent argues d holds, and on the adjective_preposition precedent argues it fails.
                                                                                                      prior 45%
  pred_e_anchor_reproduces the anchor pair reproduces v509's 0.729 within tol_pool = 0.05. Deterministic fits, so a
                        drift means the census is not measuring what v509 measured.                   prior 88%
HOW IT READS, one outcome per branch. d TRUE: possessive joins noun_preposition and verb_particle as a family whose
partners are mostly out of reach, v511's label holds for a second family, and the spread runs from roughly one in
five to sixteen in seventeen. d FALSE: v511's label was misleading for THREE of four censused families, its
three-of-six classification is superseded in both directions, and the newly reachable pairs are spendable on a
battery exactly as v559's were. b FALSE with a TRUE: no partner in this family is reachable at all and row 4 is
genuinely unestablishable here, which is the strongest form of d TRUE. e FALSE: the anchor did not reproduce and no
rate here is comparable to the earlier three.
SCOPE. One family, five pairs, rank 1, one partner each; a rate over five, not a property of the construction.
Smoke: V563_SMOKE=<out.json> (CPU, V563_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_census_five_v563_result.json"
# five fits in ONE family, each evaluating a vocabulary-disjoint sibling from the same family
VP = "possessive_"
PAIRS = {"gender": "person_our_your", "number_his_their": "pronoun_mine_yours",
         "number_its_their": "gender", "pronoun_number_hers_theirs": "number_its_their",
         "pronoun_person_ours_theirs": "number_attractor_conflict",
         "adjacent": "disjoint_my_your"}                      # last entry is the v509 ANCHOR
ANCHOR_KEY = "adjacent"
MAPS = {k: {"fit": (VP + k,), "held": (VP + v,)} for k, v in PAIRS.items()}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v509_anchor": 0.729, "v511_label": "possessive scored LOWEST of six and was called NOT general"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"strong_inert": 0.10, "tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_census_five_v563", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    ks = list(PAIRS)
    fit_v = [fld(k, VP + k, "joint_extraction") for k in ks]
    reach = {k: fld(k, VP + PAIRS[k], "units_recovery") for k in ks}
    absr = {k: fld(k, VP + PAIRS[k], "cross_abs_recovery") for k in ks}
    have = ok and all(x is not None for x in fit_v) \
        and all(v is not None for v in reach.values()) and all(v is not None for v in absr.values())
    new = [k for k in ks if k != ANCHOR_KEY]
    a = have and min(fit_v) >= B["joint_min"]
    b = have and any(reach[k] >= B["unit_min"] for k in new)
    reached = [k for k in new if have and reach[k] >= B["unit_min"]]
    c = bool(reached) and all(abs(absr[k]) <= B["cross_max"] for k in reached)
    d = have and sum(1 for k in new if reach[k] >= B["unit_min"]) <= 2
    e = have and abs(reach[ANCHOR_KEY] - PRIOR["v509_anchor"]) <= B["tol_pool"]
    return {"pred_a_fits_hold": bool(a), "pred_b_some_reachable": bool(b),
            "pred_c_reached_are_inert": bool(c), "pred_d_rate_low": bool(d),
            "pred_e_anchor_reproduces": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V563_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V563_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_possessive_census_five_v563",
              "candidate_id": "corpus.unit_possessive_census_five_v563", "bars": BARS,
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
