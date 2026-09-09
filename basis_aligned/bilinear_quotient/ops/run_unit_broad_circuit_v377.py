#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v377: THE BREADTH CURVE. How much structural variety can ONE direction absorb?

WHY. The user's instruction is that our circuits must be broader and more general, and today's own receipts showed the
criticism was right: every cell shares one skeleton, and I had been counting frame copies of one mapping as separate
circuits until v371, v373 and v375 showed a single pooled direction covers them at specialist level. The count came
down 154 -> 139 under rule R7. This rung is the constructive half: instead of asking whether two shapes are the same
circuit, it asks how WIDE a pool one direction can carry.
SEVEN SHAPES OF woke -> ` up` / calmed -> ` down`, deliberately spanning more than wording:
    A_locative    `Near the {object} the {agent} {verb}, of course,`
    B_agreement   `Everyone agreed the {agent} {verb}, frankly,`
    C_report      `It turned out the {agent} {verb}, apparently,`
    D_denial      `Nobody doubted the {agent} {verb}, honestly,`
    J_temporal    `Last winter the {agent} {verb}, admittedly,`
    M_question    `Did anyone notice the {agent} {verb}, really,`
    P_longparen   `Near the {object} the {agent} {verb}, as everyone standing in the yard already knew,`
The last one is the LENGTH axis the user named: every other cell in this corpus separates cue from readout by a
three-token parenthetical, and this one uses nine. It is also the only shape here without a four-row battery -- its
CPU screen cleared at A1 -2.12/+2.13 with zero dropped rows, and its per-shape number below is therefore exploratory
and labelled as such rather than counted. A second long-parenthetical shape was screened and DROPPED (agreement frame
with a nine-token parenthetical, A1 -0.81/+0.97, under the floor), which is itself the length finding: the same
mapping in the same corpus survives a long parenthetical in the locative frame and fails it in the agreement frame.
J and M come from v369, where both passed all four rows (the causal-subordinate frame missed row 4 and is absent).

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers      the joint direction reaches extraction >= joint_min = 0.80 of the exact-set effect on the
                           held-out rows of ALL SEVEN shapes, scored per shape so none is carried by the others. Four
                           shapes gave 0.993 to 1.008 at v371; seven is the real test of breadth.        prior 50%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape.
                           If this fails the shapes use different components and no single direction could work.
                                                                                                        prior 60%
  pred_c_control_clean     the joint direction's own-C damage UB975 stays <= 0.01 on every shape.        prior 55%
  pred_d_long_parenthetical  the LONG-parenthetical shape specifically reaches >= 0.80. This is the user's length
                           axis isolated: a direction that covers six three-token shapes and fails the nine-token one
                           is a direction tied to a distance, not to a mechanism.                        prior 45%
  pred_e_some_shape_resists  at least ONE of the seven falls below 0.80. Registered so that a uniform pass is
                           informative rather than assumed, and so the rung cannot be read as a success no matter what
                           it returns: pred_a and pred_e cannot both hold, and if both fail something is wrong with
                           the instrument rather than with the shapes.                                    prior 50%
COUNTING. Nothing here is counted; the corpus already counts this mapping once under R7. What this produces is a
BREADTH figure for that single entry -- the number of structurally distinct shapes one direction demonstrably carries
-- which is the property the user asked for and which the corpus has never reported.
Smoke: V377_SMOKE=<out.json> (CPU, V377_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v377_result.json"
SHAPES = (("A_locative", "verb_particle_up_down", 0.991),
          ("B_agreement", "verb_particle_fb_up_down", 0.998),
          ("C_report", "verb_particle_fc_up_down", 0.998),
          ("D_denial", "verb_particle_fd_up_down", 0.998),
          ("J_temporal", "verb_particle_fj_up_down", None),
          ("M_question", "verb_particle_fm_up_down", None),
          ("P_longparen", "verb_particle_fp_up_down", None))
LONG = "P_longparen"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v377", "shapes": len(SHAPES),
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
    d = ok and jx.get(LONG) is not None and jx[LONG] >= B["joint_min"]
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_joint_covers": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_long_parenthetical": bool(d), "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V377_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V377_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v377",
              "candidate_id": "corpus.unit_broad_circuit_v377", "bars": BARS,
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
