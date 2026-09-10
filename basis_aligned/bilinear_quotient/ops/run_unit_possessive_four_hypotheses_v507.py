#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v507: the four-hypothesis protocol on the SECOND named DAS target, with the control screened in the same rung.

WHERE THIS FOLLOWS FROM. v505 completed the four hypotheses on correlative_pair.both_vs_neither at rank 1 -- A1
1.001, A2 0.935, P 0.0191, C 0.0204 -- but only after a four-rung detour to obtain a C that was inert for an
INFORMATIVE reason: v497 bounded the leak, v499 found that all six cross-construction candidates were below the
units floor and that every existing correlative cell shared a readout token with the target, and v501 authored one.
THE LESSON I AM APPLYING RATHER THAN RE-LEARNING: a reachable, disjoint control cannot be assumed to exist for a
new target, so this rung SCREENS a panel in the same run instead of nominating one in advance and discovering
afterwards that its sites are unreachable.
possessive_number.adjacent_antecedent is the second named target, and its answer vocabulary is (` their`, ` his`).
Every possessive cell in the corpus shares one of those two tokens, so a same-construction disjoint control does not
exist here either; the panel is therefore cross-construction, and whether any member is REACHED is exactly what
pred_d asks. Five candidates, all with vocabularies disjoint from (` their`, ` his`), spanning four constructions:
    verb_preposition_at_to, correlative_disjoint_either_not, verb_particle_up_down,
    adjective_preposition_of_at, noun_preposition_interest
RANK IS FIXED AT 1 AND REGISTERED HERE IN ADVANCE, per the protocol. Instruments are the ones the corpus ships:
extraction for A1 and A2, `same_answer_effect` scaled by `target_scale` for P and for every panel member, so P and
C are directly comparable and no bar is carried across instruments. Nothing counted.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_A1_holdout      held-out A1 rows reach MINIMUM joint extraction joint_min = 0.80. v475 established that
                         this direction transfers across attractor siblings, so a failure here would be a surprise
                         and would void the rest.                                                     prior 85%
  pred_b_A2_carries      the SAME direction reaches 0.80 on the cell's A2 construction.               prior 60%
  pred_c_P_effect_small  the P family's same-answer effect is at or below p_max = 0.23, the top of this project's
                         recorded saturation range for same-answer controls. 0.35 would be a clause that cannot
                         fail and is deliberately not used.                                           prior 70%
  pred_d_some_C_reachable at least ONE panel member reports units_recovery >= unit_min = 0.80. Capable of failing,
                         and the correlative case is the reason to doubt it: there, ALL SIX cross-construction
                         candidates missed the floor. If none is reached, row 4 cannot be established for this
                         target either, and the honest output is that limit plus the requirement to author a cell.
                                                                                                      prior 55%
  pred_e_reachable_C_inert EVERY panel member that clears the units floor has same-answer effect at or below 0.23.
                         Quantified over the reachable ones rather than over a member nominated afterwards, which
                         is what keeps this from being post-hoc selection.                            prior 80%
HOW IT READS, one outcome per branch. a, b, c, d, e all true: the second DAS target passes all four hypotheses with
a control that is inert for an informative reason, and the protocol is satisfied on two targets rather than one.
d FALSE: no cross-construction control is reached, so row 4 is not establishable here without authoring a
disjoint cell -- the same conclusion v499 reached for correlative, and it would make that a pattern rather than a
one-off. b FALSE: the direction is construction-bound. c FALSE: it is disturbed by the answer-preserving edit.
SCOPE. One mapping, rank 1, this budget. The protocol's matched-sibling clause is answered for this target by v475,
which measured transfer across the attractor siblings; it is not re-run here.
Smoke: V507_SMOKE=<out.json> (CPU, V507_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_four_hypotheses_v507_result.json"
# one fit, four hypotheses, and the C control screened rather than nominated
CANON = "possessive_adjacent"
PANEL = ("verb_preposition_at_to", "correlative_disjoint_either_not", "verb_particle_up_down",
         "adjective_preposition_of_at", "noun_preposition_interest")
DISJOINT_C = PANEL[0]                        # kept for the shared loop; every member is scored the same way
MAPS = {"fit_canonical": {"fit": (CANON,), "held": PANEL}}
GROUPS = {k: tuple(v["fit"]) for k, v in MAPS.items()}
EVAL = {k: GROUPS[k] + tuple(v["held"]) for k, v in MAPS.items()}
RELAXED = "ALL"
RANK = 1                                     # fixed and registered in advance, per the DAS protocol
PRIOR = {"v475_sibling_transfer": (0.838, 0.804), "v499_all_six_below_floor": True}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8, "gain": 0.05, "tol_pool": 0.05,
        "cross_max": 0.3, "p_max": 0.23, "tol_pool_wide": 0.15}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_four_hypotheses_v507", "groups": len(GROUPS), "cells": sum(len(v) for v in GROUPS.values()),
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
    p_eff = hyp.get("P_same_answer_effect")
    panel = hyp.get("panel_same_answer_effect") or {}
    reach = [(c, panel.get(c)) for c in PANEL
             if ps.get(c, {}).get("units_recovery") is not None
             and ps.get(c, {}).get("units_recovery") >= B["unit_min"]]
    have = ok and all(x is not None for x in (a1_v, a2_v, p_eff))
    a = have and a1_v >= B["joint_min"]
    b = have and a2_v >= B["joint_min"]
    c = have and abs(p_eff) <= B["p_max"]
    d = bool(reach)
    e = d and all(x is not None and abs(x) <= B["p_max"] for _, x in reach)
    return {"pred_a_A1_holdout": bool(a), "pred_b_A2_carries": bool(b),
            "pred_c_P_effect_small": bool(c), "pred_d_some_C_reachable": bool(d),
            "pred_e_reachable_C_inert": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V507_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V507_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_possessive_four_hypotheses_v507",
              "candidate_id": "corpus.unit_possessive_four_hypotheses_v507", "bars": BARS,
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
