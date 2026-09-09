#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v407: I SAID DISTINCT-MAPPING NULLS STAY NULLS. This rung tests that claim on the mapping that just failed.

THE CLAIM UNDER TEST IS MINE. At 15:41 I wrote that of 134 cells which missed only rows 2 and/or 4, exactly eleven --
the SHAPE VARIANTS -- are recoverable by pooling, and that the other 123, which are distinct mappings, "stay honest
nulls unless someone gives them shape variants of their own". v405 then produced a fresh instance of exactly that
class: determiner_number_barrels (numerous / another -> ` barrels` / ` barrel`) MISSED row 2 with a held-out recovery
of 0.429, despite an A1 capability margin of -4.55/+4.61 with zero dropped rows. The model does the task; a six-unit
greedy set fitted on that shape's own sixteen rows does not carry it.
So this rung gives that mapping the shape variants my own sentence said it would need, and pools them:
    A_inspect_FAILED  `Near the {object} the {agent} inspected numerous cracked`   held-out recovery 0.429 at v405
    B_agreement       `Everyone agreed the {agent} inspected numerous cracked`     A1 -4.35/+4.35
    C_report          `It turned out the {agent} inspected numerous cracked`       A1 -4.58/+4.54
    D_denial          `Nobody doubted the {agent} inspected numerous cracked`      A1 -4.16/+4.13
If the pooled direction covers the failing shape, then a row-2 miss on a DISTINCT mapping is also a fitting failure
and my 15:41 scope sentence was too conservative -- which I would rather find out by testing it than by leaving 123
receipts filed under a claim I never checked. If it does not, the scope holds and the failing mapping is genuinely
different from the shape variants that were rescued.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers      the joint direction reaches extraction >= joint_min = 0.80 on ALL FOUR shapes.  prior 45%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape.
                           The failing shape's own set reached 0.429, so this predicate is where a components-level
                           explanation would show.                                                        prior 40%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on all four.                                        prior 55%
  pred_d_failing_shape_rescued  the shape that FAILED row 2 at v405 specifically reaches >= 0.80 under the joint
                           direction. Worked example: 0.88 means a row-2 miss on a distinct mapping is a fitting
                           failure like the shape-variant ones and my scope sentence needs widening; 0.45 means it is
                           not, and the 123 distinct-mapping nulls stay nulls on evidence rather than by assumption.
                                                                                                          prior 45%
  pred_e_some_shape_resists  at least one shape falls below 0.80. Registered at a prior above pred_a because the
                           anchor shape of this pool is a measured failure rather than a pass, which is the first
                           time that has been true of a pooled rung today.                                prior 55%
COUNTING. Nothing here is counted. determiner_number_barrels has no four-row pass, so even a rescue does not make it
an entry -- it would make it a pool member and would change how I read row-2 misses on distinct mappings.
Smoke: V407_SMOKE=<out.json> (CPU, V407_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v407_result.json"
SHAPES = (("A_inspect_FAILED", "determiner_number_barrels", None),
          ("B_agreement", "determiner_number_fb_barrels", None),
          ("C_report", "determiner_number_fc_barrels", None),
          ("D_denial", "determiner_number_fd_barrels", None))
FAILED_SHAPE = "A_inspect_FAILED"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v407", "shapes": len(SHAPES),
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
    d = ok and jx.get(FAILED_SHAPE) is not None and jx[FAILED_SHAPE] >= B["joint_min"]
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_joint_covers": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_failing_shape_rescued": bool(d), "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V407_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V407_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v407",
              "candidate_id": "corpus.unit_broad_circuit_v407", "bars": BARS,
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
