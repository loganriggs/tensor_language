#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v491: does a noun_preposition direction survive the cue moving five tokens further away?

WHAT v489 PRODUCED. Testing the audit's top confound for this family gave, as a by-product, the first performable
LONG-INTERPOSITION cell it has: `The pilot showed interest, despite the ongoing delay, of course,` keeps 32/32 rows
at margins +/-0.8, while the same frame with a COMPETING cue noun collapses entirely. So the family now has two cells
with the SAME mapping -- same cue pair, same readout pair, same suffix -- differing only in how far the cue sits from
the readout: about -5 in the ordinary cell and about -10 with the interposed PP.
WHY THIS IS ASKED HERE AND NOT ASSUMED FROM v473. In verb_preposition, about/for's direction degraded to 0.688 when
the cue moved from five tokens back to ten (v473), but the same environment manipulation was NOT binding on three
other mappings there (v471: 0.874-1.001). So the corpus's baseline for this question is MIXED and covers neither
outcome -- which is the condition I now require before registering a branch, after v487 registered one whose TRUE
agreed with the baseline and therefore tested nothing.
Both cells carry the SAME mapping, so a failure to transfer cannot be explained by directions being cue-pair keyed.
That is the specific defect that voided v487 and it is designed out here rather than argued away.
    fit_tight  noun_preposition_interest (cue at about -5); evaluate the long cell held out
    fit_long   noun_preposition_neu_interest (cue at about -10); evaluate the ordinary cell held out
    control    noun_preposition_between_with, a different mapping in the same family, on ABSOLUTE recovery
Two one-cell fits, which v481 validated as an instrument (at_to alone reached its four siblings at 0.914-1.090), so
an asymmetry here is content rather than fit size. Rank 1; relaxed budget; nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_both_in_distribution  held-out rows of BOTH fitted cells reach MINIMUM joint extraction joint_min = 0.80.
                               If this fails nothing else is readable.                                prior 80%
  pred_b_tight_to_long         the direction fitted at the ordinary distance reaches 0.80 on the long cell.
                                                                                                      prior 50%
  pred_c_long_to_tight         the direction fitted on the long cell reaches 0.80 on the ordinary one. prior 50%
  pred_d_units_available       every evaluated cell except the controls reports units_recovery >= unit_min = 0.80,
                               so a transfer failure is the DIRECTION and not units that miss the shape. prior 70%
  pred_e_different_mapping     the control's ABSOLUTE recovery stays at or below cross_max = 0.30. This one IS
                               predicted by the baseline and is registered as a CONTROL, not as a finding.  prior 85%
HOW IT READS, one outcome per branch. b TRUE and c TRUE: the direction survives a five-token interposition, and
distance is not binding for this mapping -- putting it with v471's three rather than v473's one. b FALSE and c FALSE:
both directions are bound to their own frame and distance binds here as it did for about/for, which would make it two
constructions where it bites and would matter for every noun_preposition cell, since all nine sit at exactly -5.
b TRUE with c FALSE: the tight-fitted direction is the broader object. c TRUE with b FALSE: the reverse.
SCOPE. One mapping of one family at rank 1 and this budget. The audit's point was that all nine cells sit at a
constant offset; this rung tests one of them.
Smoke: V491_SMOKE=<out.json> (CPU, V491_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_noun_distance_v491_result.json"
# two fits, one cell each: the family's constant offset -5, and the same mapping with the cue five tokens further
NP = "noun_preposition_"
NORMAL, BROKEN = "interest", "neu_interest"
CTRL = NP + "between_with"
MAPS = {"fit_tight": {"fit": (NORMAL,), "held": (BROKEN,)},
        "fit_long": {"fit": (BROKEN,), "held": (NORMAL,)}}
GROUPS = {k: tuple(NP + c for c in v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(NP + c for c in v["held"]) + (CTRL,) for k, v in MAPS.items()}
RELAXED = "ALL"
PRIOR = {"v473_about_for_at_ten": 0.688, "v471_three_mappings": (0.874, 1.001)}   # mixed baseline, covers neither
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_noun_distance_v491", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values()) \
        and all(set(G[k].get("per_shape", {})) == set(EVAL[k]) for k in GROUPS)
    def je(k, c):
        return G.get(k, {}).get("per_shape", {}).get(NP + c, {}).get("joint_extraction")
    def ur(k, c):
        return G.get(k, {}).get("per_shape", {}).get(NP + c, {}).get("units_recovery")
    def cross(k):
        return G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery")
    n_own, b_own = je("fit_tight", NORMAL), je("fit_long", BROKEN)
    n_to_b, b_to_n = je("fit_tight", BROKEN), je("fit_long", NORMAL)
    have = ok and all(x is not None for x in (n_own, b_own, n_to_b, b_to_n))
    a = have and min(n_own, b_own) >= B["joint_min"]
    b = have and n_to_b >= B["joint_min"]
    c = have and b_to_n >= B["joint_min"]
    d = ok and all(ur(k, cc) is not None and ur(k, cc) >= B["unit_min"]
                   for k in GROUPS for cc in tuple(MAPS[k]["fit"]) + tuple(MAPS[k]["held"]))
    cross = [G.get(k, {}).get("per_shape", {}).get(CTRL, {}).get("cross_abs_recovery") for k in GROUPS]
    e = ok and all(x is not None and abs(x) <= B["cross_max"] for x in cross)
    return {"pred_a_both_in_distribution": bool(a), "pred_b_tight_to_long": bool(b),
            "pred_c_long_to_tight": bool(c), "pred_d_units_available": bool(d),
            "pred_e_different_mapping": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V491_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V491_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_noun_distance_v491",
              "candidate_id": "corpus.unit_noun_distance_v491", "bars": BARS,
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
