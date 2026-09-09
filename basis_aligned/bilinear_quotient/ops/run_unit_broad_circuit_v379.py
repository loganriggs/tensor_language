#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v379: DOES POOLING RESCUE A SHAPE THAT FAILED AS A STANDALONE CELL? Breadth for the preposition entry, with two
battery failures deliberately included.

WHERE THIS SITS. v377 gave the particle entry a measured BREADTH of seven: one 13-unit direction fitted on pooled rows
covers a locative adjunct, an agreement matrix, a report matrix, a denial matrix, a temporal adjunct, an interrogative
matrix and a nine-token parenthetical at 0.994 to 1.006 of the exact-set effect, with own-C damage 0.0045 everywhere.
The preposition entry (cared -> ` about` / voted -> ` for`) has a breadth of three from v373 and this rung extends it --
but the interesting part is what it extends it WITH.
THE TWO FAILED SHAPES ARE THE EXPERIMENT. At v361 I authored frames D and E for this mapping pair and BOTH missed
rows 2 and 4 of the amended battery, so neither is a circuit by the corpus's standard and neither is counted. Both
cleared the CPU capability floor comfortably (A1 -2.10/+2.15 and -2.63/+2.74, zero dropped rows), so the model can do
the task in those frames; what failed was fitting a SPECIALIST direction on that shape's own sixteen fit rows.
Pooling changes the fit, not the task: the joint direction is fitted on rows from five shapes at once and then scored
on each shape's held-out rows separately. If it covers a shape that failed standalone, the battery miss was a
small-sample fitting failure rather than an absent mechanism, and a class of receipts I have been treating as nulls
needs re-reading. If it does not, the battery misses are real and the standard holds.
    A_locative        `Near the {object} the {agent} {verb}, of course,`        counted circuit
    B_agreement       `Everyone agreed the {agent} {verb}, frankly,`            counted, merged into A under R7
    C_report          `It turned out the {agent} {verb}, apparently,`           fused at v345
    D_denial_FAILED   `Nobody doubted the {agent} {verb}, honestly,`            battery: rows 2 and 4 MISSED
    E_spread_FAILED   `Word spread that the {agent} {verb}, evidently,`         battery: rows 2 and 4 MISSED

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers_passing  the joint direction reaches extraction >= joint_min = 0.80 on all THREE shapes that
                           passed their batteries. v373 got 1.001, 0.988 and 1.025 on these three with a direction
                           fitted on those three alone; adding two harder shapes to the pool could degrade it, and
                           this predicate is what would show that.                                        prior 75%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape
                           INCLUDING the two failures -- if the units do not even reach the failed shapes, the
                           rescue question is answered negatively before the direction is considered.     prior 55%
  pred_c_control_clean     the joint direction's own-C damage UB975 stays <= 0.01 on all five shapes.     prior 55%
  pred_d_rescues_a_failed_shape  at least ONE of the two battery failures reaches >= 0.80 under the joint direction.
                           This is the rescue claim. Worked example: D at 0.91 and E at 0.62 gives TRUE and says the
                           denial frame's battery miss was a fitting failure on sixteen rows; both at 0.55 gives
                           FALSE and the misses stand as honest nulls.                                    prior 40%
  pred_e_some_shape_resists  at least one of the five falls below 0.80, registered so that a uniform pass is
                           informative rather than assumed. Note pred_d and pred_e can BOTH be true -- one failure
                           rescued and the other not is the outcome I would find most likely if the rescue is real,
                           and I am saying so here because at v367 I wrote two predicates were mutually exclusive and
                           both came out true.                                                            prior 65%
COUNTING. Nothing here is counted. A rescue would change how battery misses are READ, not how many circuits exist.
Smoke: V379_SMOKE=<out.json> (CPU, V379_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v379_result.json"
SHAPES = (("A_locative", "verb_preposition_about_for", 0.998),
          ("B_agreement", "verb_preposition_fb_about_for", 0.998),
          ("C_report", "verb_preposition_fc_about_for", 0.998),
          ("D_denial_FAILED", "verb_preposition_fd_about_for", None),
          ("E_spread_FAILED", "verb_preposition_fe_about_for", None))
FAILED = ("D_denial_FAILED", "E_spread_FAILED")
PASSED = ("A_locative", "B_agreement", "C_report")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v379", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    per = R.get("per_shape", {})
    ok = bool(per) and all("error" not in s for s in per.values())
    jx = {k: s.get("joint_extraction") for k, s in per.items()}
    a = ok and all(jx.get(k) is not None and jx[k] >= B["joint_min"] for k in PASSED)
    b = ok and all(s.get("units_recovery") is not None and s["units_recovery"] >= B["unit_min"] for s in per.values())
    c = ok and all(s.get("C_ub975") is not None and s["C_ub975"] <= B["c_ub_max"] for s in per.values())
    d = ok and any(jx.get(k) is not None and jx[k] >= B["joint_min"] for k in FAILED)
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_joint_covers_passing": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_rescues_a_failed_shape": bool(d), "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V379_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V379_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v379",
              "candidate_id": "corpus.unit_broad_circuit_v379", "bars": BARS,
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
