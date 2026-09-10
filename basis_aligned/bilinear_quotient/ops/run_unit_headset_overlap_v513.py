#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v513: why are the six reachability numbers bimodal? Head-set OVERLAP is the next candidate.

WHAT v511 LEFT. Across six constructions, whether a fitted head set reaches a vocabulary-disjoint sibling in its own
construction came out bimodal with nothing between: 0.677 (verb_particle), 0.682 (adjective_preposition),
0.729 (possessive) against 0.913 (noun_preposition), 0.924 (verb_preposition), 0.973 (correlative). Row 4 is
establishable with an existing corpus cell in the upper group and not in the lower one, so the factor matters.
THE OBVIOUS HYPOTHESIS IS ALREADY DEAD AND IT COST NO GPU. Frame identity does not explain it -- I read the cells:
verb_particle's pair have IDENTICAL frames apart from the cue verb (`the leader woke` / `the leader leaked`) and it
MISSES at 0.677, while noun_preposition's pair differ in word order, matrix verb and an extra object NP
(`The leader near the window showed interest` / `Near the window the leader explained the difference`) and it
REACHES at 0.913. So the split is not about how similar the sentences are.
THIS RUNG TESTS THE NEXT CANDIDATE: do the two cells of a pair independently CHOOSE THE SAME HEADS? If reachability
is just "these two behaviours are read by the same units", then the pairs that reach should have overlapping greedy
sets and the pairs that miss should not -- and reachability stops being a mystery and becomes a property that can be
checked before a control is nominated. Each of eight cells is fitted ALONE, its greedy set recorded, and the
Jaccard overlap computed per pair. v481 established that a one-cell fit is a sound instrument here (at_to alone
reached its four frame siblings at 0.914-1.090), so fitting singly is not a compromise.
    REACHED pairs  verb_preposition at_to / about_for      noun_preposition interest / between_with
    MISSED pairs   verb_particle up_down / out_away        adjective_preposition of_at / in_beneath
Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold        all EIGHT cells reach MINIMUM held-out joint extraction joint_min = 0.80 on their own
                          rows. The instrument check; a cell that does not fit says nothing about overlap. prior 80%
  pred_b_reached_overlap  BOTH reached pairs have Jaccard overlap at or above jac_min = 0.50.          prior 55%
  pred_c_missed_overlap   BOTH missed pairs have Jaccard overlap BELOW 0.50.                           prior 55%
  pred_d_separation       the smaller of the two reached overlaps exceeds the larger of the two missed overlaps by
                          at least gain = 0.05, i.e. the two groups separate on this measure rather than merely
                          straddling a bar I chose. Worked example: 0.62 against 0.30 is 0.32, TRUE; 0.52 against
                          0.49 is 0.03, FALSE and the bar was doing the work, not the data.            prior 50%
  pred_e_sets_nonempty    every fitted cell chose at least three units, so a Jaccard is meaningful rather than an
                          artefact of a one-unit set.                                                  prior 90%
HOW IT READS, one outcome per branch. b, c and d TRUE: reachability is head-set overlap, and it can be checked
CHEAPLY -- fit both cells and compare sets -- before nominating a control, which would turn this session's repeated
row-4 failures into something predictable in advance. d FALSE with b and c TRUE: the groups sit either side of 0.50
but do not separate, so the bar is carrying the result and overlap is at best a weak correlate. b or c FALSE:
overlap does not track reachability and the factor is something else again -- in which case I have eliminated the
second candidate as cheaply as the first and will say so rather than reaching for a third in the same receipt.
SCOPE. Four pairs, one mapping each, rank 1, this budget.
Smoke: V513_SMOKE=<out.json> (CPU, V513_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_headset_overlap_v513_result.json"
# eight single-cell fits; the pairs are the four sibling pairs v511 measured
PAIRS = {
    "vp_reached": ("verb_preposition_at_to", "verb_preposition_about_for"),
    "np_reached": ("noun_preposition_interest", "noun_preposition_between_with"),
    "particle_missed": ("verb_particle_up_down", "verb_particle_out_away"),
    "adj_missed": ("adjective_preposition_of_at", "adjective_preposition_in_beneath"),
}
REACHED, MISSED = ("vp_reached", "np_reached"), ("particle_missed", "adj_missed")
MAPS = {f"{k}__{i}": {"fit": (c,), "held": ()} for k, v in PAIRS.items() for i, c in enumerate(v)}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] for k in MAPS}
RELAXED = "ALL"
PRIOR = {"v511_reach": (0.913, 0.924, 0.973), "v511_miss": (0.677, 0.682, 0.729)}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"jac_min": 0.5, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_headset_overlap_v513", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def uset(k, i):
        return set(G.get(f"{k}__{i}", {}).get("units") or [])

    def ext_of(k, i):
        cell = PAIRS[k][i]
        return G.get(f"{k}__{i}", {}).get("per_shape", {}).get(cell, {}).get("joint_extraction")

    def jac(k):
        a, b = uset(k, 0), uset(k, 1)
        return (len(a & b) / len(a | b)) if (a or b) else None

    exts = [ext_of(k, i) for k in PAIRS for i in (0, 1)]
    js = {k: jac(k) for k in PAIRS}
    have = ok and all(x is not None for x in exts) and all(v is not None for v in js.values())
    a = have and min(exts) >= B["joint_min"]
    b = have and all(js[k] >= B["jac_min"] for k in REACHED)
    c = have and all(js[k] < B["jac_min"] for k in MISSED)
    d = have and (min(js[k] for k in REACHED) - max(js[k] for k in MISSED)) >= B["gain"]
    e = ok and all(len(uset(k, i)) >= 3 for k in PAIRS for i in (0, 1))
    return {"pred_a_fits_hold": bool(a), "pred_b_reached_overlap": bool(b),
            "pred_c_missed_overlap": bool(c), "pred_d_separation": bool(d),
            "pred_e_sets_nonempty": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V513_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V513_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            groups[gname] = {"cells": list(cells), "n_units": len(units), "per_shape": per_shape,
                             "units": [str(u) for u in units]}
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            groups[gname] = {"cells": list(cells), "error": f"{type(err).__name__}: {err}"}
        print(f"[{gname}] {'error' if 'error' in groups[gname] else groups[gname]['n_units']} units, "
              f"{round(time.perf_counter() - t0, 1)}s", flush=True)

    R = {"groups": groups}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_headset_overlap_v513",
              "candidate_id": "corpus.unit_headset_overlap_v513", "bars": BARS,
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
