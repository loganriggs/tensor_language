#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v511: how often is a fitted head set CONSTRUCTION-GENERAL? Row 4 depends on the answer.

WHY THIS IS THE QUESTION. Row 4 -- the C hypothesis, the one that makes a claim SELECTIVE rather than merely
present -- needs a control that is both VOCABULARY-DISJOINT from the target and REACHED by the fitted head set. A
near-zero reading from an unreachable control is uninformative, which this session established twice: v499 found
all six cross-construction candidates below the units floor for correlative, and v507 found all five below it for
possessive. So the controls that are disjoint tend to be the ones the head set cannot reach, and whether a family
can support row 4 at all comes down to a single property: does its head set reach OTHER MAPPINGS IN ITS OWN
CONSTRUCTION?
THREE ANSWERS ARE ALREADY IN HAND AND THEY DISAGREE, which is why a count is worth a rung rather than a guess:
    correlative       0.973  construction-general, but only after v501 authored a disjoint cell for it
    verb_preposition  0.924  construction-general already -- about_for under the at_to fit (v487), inert at 0.014
    possessive        0.729  NOT -- and the same move that gave correlative 0.973 gave this 0.729 (v509), so the
                             head set there is specific to the NUMBER VARIABLE rather than to the construction
This rung adds the three remaining constructions that have a disjoint sibling pair in the corpus. Each group fits ONE
cell and evaluates a sibling whose answer vocabulary shares no token with the fitted cell's:
    verb_particle           up_down (` up`/` down`)  ->  out_away (` out`/` away`)
    adjective_preposition   of_at   (` of`/` at`)    ->  in_beneath (` in`/` beneath`)
    noun_preposition        interest(` in`/` for`)   ->  between_with (` between`/` with`)
Rank 1; relaxed budget; nothing counted. Inertness is read as ABSOLUTE recovery against cross_max, the same
instrument and bar used for panel members in v499 and v501, so no bar is carried across instruments.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_fits_hold         all three FITTED cells reach MINIMUM held-out joint extraction joint_min = 0.80. The
                           instrument check; if a fit does not hold, its group says nothing.          prior 85%
  pred_b_particle_reaches  the verb_particle sibling reports units_recovery >= unit_min = 0.80.       prior 55%
  pred_c_adjective_reaches the adjective_preposition sibling reports units_recovery >= 0.80.          prior 55%
  pred_d_noun_reaches      the noun_preposition sibling reports units_recovery >= 0.80.               prior 55%
  pred_e_reached_are_inert EVERY sibling that clears the units floor has ABSOLUTE recovery at or below
                           cross_max = 0.30. Quantified over the reached ones, not over one nominated afterwards --
                           and vacuously false if none is reached, which is the correct behaviour and the same one
                           v499 and v507 showed.                                                      prior 80%
HOW IT READS. The count is the result. b, c and d all TRUE makes it five of six constructions construction-general,
possessive the lone exception, and row 4 establishable with an existing cell nearly everywhere -- which would mean
the two failures this session were about those two families and not about the method. All three FALSE makes it two
of six, row 4 the exception rather than the rule, and every C row in this corpus that was never checked for
reachability becomes suspect. A mixed count is the honest middle and is reported as a proportion, with the
per-family numbers, rather than as a verdict either way.
SCOPE. One mapping per construction at this rank and budget; a family with several mappings could behave differently
at a different one, and the count is over constructions tested, not over the corpus.
Smoke: V511_SMOKE=<out.json> (CPU, V511_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_construction_generality_v511_result.json"
# one fit per construction; each evaluates a vocabulary-disjoint sibling from the SAME construction
MAPS = {
    "verb_particle": {"fit": ("verb_particle_up_down",), "held": ("verb_particle_out_away",)},
    "adjective_preposition": {"fit": ("adjective_preposition_of_at",), "held": ("adjective_preposition_in_beneath",)},
    "noun_preposition": {"fit": ("noun_preposition_interest",), "held": ("noun_preposition_between_with",)},
}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"correlative": 0.973, "verb_preposition": 0.924, "possessive": 0.729}   # three answers already in hand
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_construction_generality_v511", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)

    def cell(k, c, field):
        return G.get(k, {}).get("per_shape", {}).get(c, {}).get(field)

    fit_v = [cell(k, MAPS[k]["fit"][0], "joint_extraction") for k in GROUPS]
    sib = {k: (cell(k, MAPS[k]["held"][0], "units_recovery"),
               cell(k, MAPS[k]["held"][0], "cross_abs_recovery")) for k in GROUPS}
    have = ok and all(x is not None for x in fit_v) and all(u is not None and x is not None
                                                            for u, x in sib.values())
    a = have and min(fit_v) >= B["joint_min"]
    b = have and sib["verb_particle"][0] >= B["unit_min"]
    c = have and sib["adjective_preposition"][0] >= B["unit_min"]
    d = have and sib["noun_preposition"][0] >= B["unit_min"]
    reached = [(k, x) for k, (u, x) in sib.items() if have and u >= B["unit_min"]]
    e = bool(reached) and all(abs(x) <= B["cross_max"] for _, x in reached)
    return {"pred_a_fits_hold": bool(a), "pred_b_particle_reaches": bool(b),
            "pred_c_adjective_reaches": bool(c), "pred_d_noun_reaches": bool(d),
            "pred_e_reached_are_inert": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V511_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V511_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_construction_generality_v511",
              "candidate_id": "corpus.unit_construction_generality_v511", "bars": BARS,
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
