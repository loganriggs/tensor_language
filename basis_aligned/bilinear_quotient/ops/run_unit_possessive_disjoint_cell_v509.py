#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v509: a disjoint POSSESSIVE cell was authored. Is it reached, and does row 4 finally close?

THE CHAIN. v507 got three of four hypotheses on possessive_number.adjacent_antecedent -- A1 0.988, A2 1.016,
P 0.0814 -- and row 4 failed for a reason that was not about the direction: none of five cross-construction control
candidates was REACHABLE (units 0.474, 0.465, 0.395, 0.329, 0.322). With v499, that made a pattern: a fitted head
set reaches only its own construction, so the vocabulary-disjoint controls are exactly the unreachable ones and
their near-zero readings say nothing. For correlative the fix was to author a same-construction cell with a disjoint
readout pair (v501, reached at units 0.973, inert at -0.021). Every possessive cell shares ` their` or ` his`, so
that fix had to be built rather than borrowed.
possessive_disjoint_my_your changes the CUE from number to PERSON while keeping the possessive-determiner slot, the
frame shape and a shared matched suffix: `Whenever I checked` -> ` my` against `Whenever you checked` -> ` your`.
Both cues and both answers are single GPT-2 tokens, so base and donor are length-matched. It screens at 32/32 rows
with margins +/-2.96 and +/-2.06. WHETHER THE POSSESSIVE HEAD SET REACHES IT IS NOT ASSUMED -- that is pred_b, and
v507 is the reason to doubt it.
Same deterministic fit as v507. RANK FIXED AT 1 and registered. adjective_preposition_of_at, v507's best
cross-construction candidate at 0.465, is carried along as a reference point. Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_reproduces_fit    the A1 held-out extraction lands within tol_pool = 0.05 of v507's 0.988. Deterministic
                           fit; instrument check.                                                     prior 90%
  pred_b_new_cell_reachable the new disjoint cell reports units_recovery >= unit_min = 0.80. Capable of failing:
                           five candidates missed this floor on this exact fit, the best at 0.474.    prior 60%
  pred_c_new_cell_inert    its same-answer effect is at or below p_max = 0.23, the top of the recorded range.
                                                                                                      prior 75%
  pred_d_P_and_C_comparable the P family's effect and the new cell's differ by at most tol_pool_wide = 0.15, i.e.
                           the direction treats an answer-preserving edit and a disjoint same-construction
                           behaviour alike. P was 0.0814 in v507.                                     prior 70%
  pred_e_units_available   the fitted cell and its A2 family both report units_recovery >= 0.80.      prior 85%
HOW IT READS, one outcome per branch. b, c and d TRUE: row 4 closes for this target, the second DAS target passes
all four hypotheses, and the recipe -- when a family has no disjoint control, author one INSIDE the construction --
is confirmed on a second family rather than resting on the correlative case alone. b FALSE: even a
same-construction cell with a different cue TYPE is not reached, which would mean the head set is specific to the
NUMBER variable rather than to the construction, a sharper limit than v507 established and one that would make row 4
unreachable for this target by any route I currently have. c FALSE: the cell is reached but the direction moves it,
so the direction is not specific and row 4 fails on the evidence rather than for want of a control.
SCOPE. One mapping, rank 1, this budget.
Smoke: V509_SMOKE=<out.json> (CPU, V509_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_disjoint_cell_v509_result.json"
# one fit, four hypotheses, and the C control screened rather than nominated
CANON = "possessive_adjacent"
NEW_DISJOINT = "possessive_disjoint_my_your"      # same construction, cue changed from number to PERSON
PANEL = (NEW_DISJOINT, "adjective_preposition_of_at")   # the new cell, and v507's best candidate at 0.465
DISJOINT_C = PANEL[0]                        # kept for the shared loop; every member is scored the same way
MAPS = {"fit_canonical": {"fit": (CANON,), "held": PANEL}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v507_A1": 0.988, "v507_P": 0.0814, "v507_best_units": 0.465}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_disjoint_cell_v509", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    p_eff = hyp.get("P_same_answer_effect")
    panel = hyp.get("panel_same_answer_effect") or {}
    new_eff = panel.get(NEW_DISJOINT)
    new_u = ps.get(NEW_DISJOINT, {}).get("units_recovery")
    have = ok and all(x is not None for x in (a1_v, p_eff, new_eff, new_u))
    a = have and abs(a1_v - PRIOR["v507_A1"]) <= B["tol_pool"]
    b = have and new_u >= B["unit_min"]
    c = have and abs(new_eff) <= B["p_max"]
    d = have and abs(abs(p_eff) - abs(new_eff)) <= B["tol_pool_wide"]
    e = ok and ps.get(CANON, {}).get("units_recovery", 0) >= B["unit_min"] \
        and (hyp.get("A2_units_recovery") or 0) >= B["unit_min"]
    return {"pred_a_reproduces_fit": bool(a), "pred_b_new_cell_reachable": bool(b),
            "pred_c_new_cell_inert": bool(c), "pred_d_P_and_C_comparable": bool(d),
            "pred_e_units_available": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V509_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V509_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            scale = g.target_scale(g.prepare(backend, per_shape_rows[CANON][0], **V))
            eff = lambda p: round(g.same_answer_effect(p, g.patched_axis(backend, p, units, q=q_joint), scale), 4)
            panel_eff = {c: eff(g.prepare(backend, per_shape_rows[c][0], **V)) for c in PANEL}
            p_disj = g.prepare(backend, per_shape_rows[DISJOINT_C][0], **V)
            hyp = {"A2_units_recovery": e_a2,
                   "scale": round(scale, 4),
                   "P_same_answer_effect": eff(p_p),
                   "C_same_answer_effect": eff(p_disj),
                   "panel_same_answer_effect": panel_eff,
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
    result = {"predictions": predictions, "schema": "unit_possessive_disjoint_cell_v509",
              "candidate_id": "corpus.unit_possessive_disjoint_cell_v509", "bars": BARS,
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
