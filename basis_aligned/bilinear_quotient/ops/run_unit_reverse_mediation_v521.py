#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v521: run the pair BACKWARDS. Is the non-overlap symmetric, or is one node set inside the other?

THE QUESTION v519 QUEUED. Two constructions have no reachable disjoint control: a node set fitted on up_down
recovers only 0.677 of the patching effect on out_away, and one fitted on of_at only 0.682 on in_beneath. v517 and
v519 established that this is not a settings artefact -- the unit cap never binds, and the greedy search SATURATES
(tightening from target 0.995/min_gain 0.0002 to 0.999/0.00005 added zero units) with the siblings still at 0.730
and 0.706. So the target's search cannot be pushed to pick up the sibling's nodes.
THAT LEAVES TWO READINGS AND ONLY THE REVERSE DIRECTION SEPARATES THEM. Either the two behaviours are carried by
genuinely different machinery -- in which case the sibling's own node set should fail on the TARGET too, and the
non-overlap is symmetric -- or the sibling's nodes are a superset that the target's greedy ranking simply never
reaches, in which case fitting the sibling and patching the target should work fine and the earlier failure is about
search order rather than mechanism. v513 makes this worth asking rather than assuming: it measured the two sets
sharing 10 of 22 units in verb_particle and 16 of 32 in adjective_preposition, so neither is contained in the other
by unit count, but unit identity is not the same as mediation.
    verb_particle          fit out_away    -> patch up_down      (forward was 0.677)
    adjective_preposition  fit in_beneath  -> patch of_at        (forward was 0.682)
Same standard budget as every earlier rung, so the reverse number is comparable to the forward one. Rank 1;
nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_sibling_fits_hold  both SIBLING fits reach MINIMUM held-out joint extraction joint_min = 0.80 on their own
                            rows. If a sibling does not fit, its reverse number says nothing.          prior 85%
  pred_b_reverse_particle_low  up_down's units_recovery under out_away's node set is BELOW unit_min = 0.80.
                                                                                                      prior 60%
  pred_c_reverse_adjective_low of_at's units_recovery under in_beneath's node set is below 0.80.       prior 60%
  pred_d_symmetric          for BOTH constructions the forward and reverse recoveries differ by at most
                            tol_pool_wide = 0.15. Worked example: 0.677 forward against 0.70 reverse is 0.023,
                            TRUE and the non-overlap is symmetric; 0.677 against 0.95 is 0.273, FALSE and the
                            sibling's set contains what the target's does not.                        prior 55%
  pred_e_sets_reproduce     the sibling fits choose unit counts within 3 of v513's 17 and 30. The fits are
                            deterministic and this is the instrument check across receipts.           prior 80%
HOW IT READS, one outcome per branch. b, c and d TRUE: the non-overlap is SYMMETRIC -- neither behaviour's nodes
mediate the other -- so these are different machinery and no same-construction control will ever serve, which makes
"unexamined" a fact about the model rather than about my search. b and c FALSE: the sibling's nodes DO mediate the
target, the relation is asymmetric, and the earlier failure is about greedy ranking rather than mechanism -- in
which case the reachability framing I have been building on since v499 needs revising, and I would say so. d FALSE
with b and c TRUE: both directions fail but unequally, so one set is closer to containing the other and the honest
report is the asymmetry rather than a verdict.
SCOPE. Two constructions, one pair each, rank 1, standard budget.
Smoke: V521_SMOKE=<out.json> (CPU, V521_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_reverse_mediation_v521_result.json"
# the same two pairs, fitted BACKWARDS: the sibling is the fitted task and the target is patched
MAPS = {
    "verb_particle": {"fit": ("verb_particle_out_away",), "held": ("verb_particle_up_down",)},
    "adjective_preposition": {"fit": ("adjective_preposition_in_beneath",),
                              "held": ("adjective_preposition_of_at",)},
}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
FORWARD = {"verb_particle": 0.677, "adjective_preposition": 0.682}        # v511
PRIOR = {"forward": FORWARD, "v513_sibling_units": {"verb_particle": 17, "adjective_preposition": 30}}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"tol_pool_wide": 0.15, "joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05, "cross_max": 0.3}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_reverse_mediation_v521", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    G = R.get("groups", {})
    ok = bool(G) and set(G) == set(GROUPS) and all("error" not in g for g in G.values())

    def fld(k, cell, field):
        return G.get(k, {}).get("per_shape", {}).get(cell, {}).get(field)

    fams = list(GROUPS)
    fit_v = [fld(k, MAPS[k]["fit"][0], "joint_extraction") for k in fams]
    rev = {k: fld(k, MAPS[k]["held"][0], "units_recovery") for k in fams}
    counts = {k: G.get(k, {}).get("n_units") for k in fams}
    have = ok and all(x is not None for x in fit_v) and all(v is not None for v in rev.values()) \
        and all(v is not None for v in counts.values())
    a = have and min(fit_v) >= B["joint_min"]
    b = have and rev["verb_particle"] < B["unit_min"]
    c = have and rev["adjective_preposition"] < B["unit_min"]
    d = have and all(abs(rev[k] - FORWARD[k]) <= B["tol_pool_wide"] for k in fams)
    e = have and all(abs(counts[k] - PRIOR["v513_sibling_units"][k]) <= 3 for k in fams)
    return {"pred_a_sibling_fits_hold": bool(a), "pred_b_reverse_particle_low": bool(b),
            "pred_c_reverse_adjective_low": bool(c), "pred_d_symmetric": bool(d),
            "pred_e_sets_reproduce": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V521_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V521_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_reverse_mediation_v521",
              "candidate_id": "corpus.unit_reverse_mediation_v521", "bars": BARS,
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
