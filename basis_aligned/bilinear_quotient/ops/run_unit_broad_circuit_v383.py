#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v383: LOOKING FOR THE EDGE. Ten shapes of one mapping pair, three of them varying axes this corpus has never varied.

WHY THIS IS THE RIGHT RUNG TO SPEND ON. Six pooled rungs -- v371, v373, v375, v377, v379, v381 -- have now covered
every shape put into a pool: twelve entries, more than forty shape-instances, joint extraction 0.97 to 1.04
throughout, and pred_e_some_shape_resists false every single time. That is not yet evidence that circuits are
unbounded; it is evidence that the shapes I have been authoring are too similar to challenge them. Every cell in this
corpus, including the seven that gave woke -> ` up` / calmed -> ` down` a breadth of seven, shares three properties I
have never varied: the subject is a simple definite singular noun phrase, nothing intervenes between the subject and
the cue verb, and what separates the cue from the readout is a parenthetical PHRASE rather than a clause.
THE THREE STRESS SHAPES VARY EXACTLY THOSE, one each:
    R_relclause_subject   `The {agent} who trained the cadets {verb}, of course,`     relative clause inside the subject
    S_conjoined_subject   `The {agent} and the {object} {verb}, of course,`           conjoined, plural subject
    T_clause_between      `Near the {object} the {agent} {verb}, once the lanterns had been put out,`
                                                                                      a full subordinate CLAUSE, with
                                                                                      its own verb, between cue and readout
They join the seven shapes v377 already covered. All three cleared the CPU capability floor with zero dropped rows
(A1 -4.47/+4.55, -3.18/+3.24, -1.57/+1.57); the clause-between shape is the weakest of the ten by a factor of three
against the strongest, which is the price of putting a verb in the span.
A SHAPE THAT RESISTS IS THE RESULT I AM AFTER. If one direction covers all ten, this mapping's circuit is robust to
subject complexity, subject number and an intervening clause, and the corpus's breadth figure for it becomes ten. If
the stress shapes resist while the seven original ones hold, I will have located the edge and named the axis that
crosses it -- which is worth more than another entry at 0.99, and is the thing the corpus has never had.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers_all_ten  the joint direction reaches extraction >= joint_min = 0.80 on ALL TEN shapes, scored
                           per shape.                                                                    prior 40%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape.
                           v379 found a shape at 0.763 whose direction was still covered, so this can fail while
                           pred_a holds and the receipt shows which.                                     prior 45%
  pred_c_control_clean     the joint direction's own-C damage UB975 stays <= 0.01 on every shape.        prior 50%
  pred_d_stress_shapes_covered  the THREE stress shapes specifically are all covered at >= 0.80. This is the edge
                           question isolated from the seven shapes already known to pool. Worked example: the three
                           at 0.93, 0.88, 0.81 gives TRUE and the breadth figure becomes ten; 0.93, 0.88, 0.54 gives
                           FALSE and the intervening-clause axis is the edge.                            prior 40%
  pred_e_some_shape_resists  at least one of the ten falls below 0.80. I am registering this at a HIGHER prior than
                           pred_a for the first time, because six rungs of universal coverage most likely mean my
                           shapes were too easy rather than that no edge exists, and because a rung that can only
                           confirm is not worth the GPU.                                                 prior 60%
COUNTING. Nothing here is counted; this mapping already counts once. The output is either a breadth figure of ten or
the first located limit on a circuit's generality.
Smoke: V383_SMOKE=<out.json> (CPU, V383_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v383_result.json"
SHAPES = (("A_locative", "verb_particle_up_down", 0.991),
          ("B_agreement", "verb_particle_fb_up_down", 0.998),
          ("C_report", "verb_particle_fc_up_down", 0.998),
          ("D_denial", "verb_particle_fd_up_down", 0.998),
          ("J_temporal", "verb_particle_fj_up_down", None),
          ("M_question", "verb_particle_fm_up_down", None),
          ("P_longparen", "verb_particle_fp_up_down", None),
          ("R_relclause_subject", "verb_particle_fr_up_down", None),
          ("S_conjoined_subject", "verb_particle_fs_up_down", None),
          ("T_clause_between", "verb_particle_ft_up_down", None))
STRESS = ("R_relclause_subject", "S_conjoined_subject", "T_clause_between")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v383", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    per = R.get("per_shape", {})
    ok = bool(per) and all("error" not in s for s in per.values())
    jx = {k: s.get("joint_extraction") for k, s in per.items()}
    a = ok and all(v is not None and v >= B["joint_min"] for v in jx.values())
    b = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"] for s in per.values())
    c = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in per.values())
    d = ok and all(jx.get(k) is not None and jx[k] >= B["joint_min"] for k in STRESS)
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_joint_covers_all_ten": bool(a), "pred_b_units_transfer": bool(b),
            "pred_c_control_clean": bool(c), "pred_d_stress_shapes_covered": bool(d),
            "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V383_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V383_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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

    pooled_fit, pooled_c_fit, per_shape_rows = [], [], {}
    for tag, name, _spec in SHAPES:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{name}")
        a1, cc = g.rows_of(m, "A1"), g.rows_of(m, "C")
        per_shape_rows[tag] = (cut(held_half(a1)), cut(held_half(cc)))
        pooled_fit += cut(fit_half(a1))
        pooled_c_fit += cut(fit_half(cc))

    P_fit = g.prepare(backend, pooled_fit, **V)
    P_cfit = g.prepare(backend, pooled_c_fit)
    singles, ranked, greedy = g.greedy_heads(backend, P_fit, pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
    units = list(greedy["chosen"])
    mu_joint = mu_of(P_fit, units)
    q_joint, hist = g.fit_block_subspace_constrained(
        backend, P_fit, units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW,
        controls=(P_cfit,), control_weight=LAM, mu=mu_joint)

    per_shape = {}
    for tag, name, spec in SHAPES:
        held_rows, c_rows = per_shape_rows[tag]
        try:
            p_held = g.prepare(backend, held_rows, **V)
            p_c = g.prepare(backend, c_rows)
            e_exact = ext(p_held, units)
            e_joint = ext(p_held, units, q=q_joint)
            per_shape[tag] = {
                "cell": name,
                "units_recovery": e_exact,
                "joint_extraction": round(e_joint / e_exact, 3) if abs(e_exact) > 1e-6 else None,
                "specialist_extraction": spec,
                "A1_damage": dmg(p_held, units, q_joint, mu_joint),
                "C_damage": dmg(p_c, units, q_joint, mu_joint),
                "n_held": len(p_held.base_batch.row_ids),
            }
            per_shape[tag]["C_ub975"] = per_shape[tag]["C_damage"]["ce_ub975"]
        except Exception as err:                                   # noqa: BLE001 - recorded, never silently dropped
            per_shape[tag] = {"cell": name, "error": f"{type(err).__name__}: {err}"}

    R = {"per_shape": per_shape}
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v383",
              "candidate_id": "corpus.unit_broad_circuit_v383", "bars": BARS,
              "recipe": {"pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                         "lam": LAM, "steps": steps, "lr": LR, "complement_weight": CW},
              "joint_units": units, "n_joint_units": len(units),
              "pooled_fit_rows": len(P_fit.base_batch.row_ids), "pooled_control_rows": len(P_cfit.base_batch.row_ids),
              "final_loss": (round(float(hist[-1][-1]), 5) if isinstance(hist[-1], (list, tuple)) else round(float(hist[-1]), 5)) if hist else None,
              "per_shape": per_shape, "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "n_joint_units": len(units),
                      "per_shape": {k: {kk: v[kk] for kk in ("joint_extraction", "specialist_extraction", "units_recovery")
                                        if kk in v} for k, v in per_shape.items()},
                      "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
