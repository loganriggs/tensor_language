#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v373: does the BROAD CIRCUIT result hold in the other readout class, and on three shapes rather than four?

WHAT v371 FOUND, AND WHY IT NEEDS REPEATING BEFORE IT IS ACTED ON. One greedy unit set (13 units) and ONE constrained
rank-1 direction, fitted on rows pooled from four structurally distinct sentence shapes of woke -> ` up` /
calmed -> ` down`, reached 0.993, 0.994, 1.000 and 1.008 of the exact-set effect per shape -- matching or beating the
four SPECIALIST directions (0.991, 0.998, 0.998, 0.998) -- with pooled-unit recovery 0.876 to 0.944 in every shape and
own-C damage UB975 at 0.0096 everywhere. pred_e_generality_costs came out FALSE: generality cost nothing.
That result contradicts the reading I had been giving the frame receipts. Those same four shapes are pairwise
SEPARABLE at rank 1 (v337, v341, v355, v359, v363), and I had been counting each as its own circuit. Both facts are
true, and together they say something sharper than either: rank-1 separability detects that a SPECIALISED direction
exists which spares the other shapes -- not that the mechanism differs. A joint direction covering all of them at
specialist level is the stronger evidence, and it says the four are one object.
Before I revise the corpus count downward on the strength of one mapping pair in one readout class, this rung repeats
the test in the PREPOSITION class on cared -> ` about` / voted -> ` for`, which has counted circuits in three shapes:
    A  `Near the {object} the {agent} {verb}, of course,`   locative adjunct
    B  `Everyone agreed the {agent} {verb}, frankly,`       agreement matrix
    C  `It turned out the {agent} {verb}, apparently,`      report/evidential matrix
Three shapes rather than four, and a different readout class, so a pass is a replication and a failure localises the
v371 result to particles.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers    the joint direction reaches extraction >= joint_min = 0.80 of the exact-set effect on the
                         held-out rows of ALL THREE shapes, scored per shape so none can be carried by the others.
                         v371 got 0.993 to 1.008 on four shapes in the particle class.                prior 70%
  pred_b_cost_bounded    on every shape, the joint direction's extraction is at least keep_frac = 0.85 of that
                         shape's specialist extraction (0.998 for all three, from the receipts on disk). Worked
                         example: a joint of 0.87 against a specialist of 0.998 gives 0.872, TRUE; 0.80 gives 0.80,
                         FALSE.                                                                       prior 65%
  pred_c_control_clean   the joint direction's own-C damage UB975 stays <= c_ub_max = 0.01 on all three shapes.
                         v371 landed at 0.0096, just inside, so this is not a formality.              prior 55%
  pred_d_units_shared    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape's
                         held rows, separating "the components differ" from "only the direction differs".
                                                                                                      prior 75%
  pred_e_generality_costs  on at least one shape the joint direction gives up at least cost_min = 0.05 against that
                         shape's specialist. Registered alongside pred_b, not against it: both can be true. In v371
                         this was FALSE -- generality was free -- and if it is false again the case for merging
                         frame copies into single broad circuits is strong in both readout classes.   prior 40%
COUNTING. Nothing here is counted. If this replicates, the frame copies I counted today should be MERGED into their
originals and the corpus count revised downward by roughly the number of frame copies added since 11:00, and I will
compute that number from the receipts rather than estimate it.
Smoke: V373_SMOKE=<out.json> (CPU, V373_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v373_result.json"
SHAPES = (("A", "verb_preposition_about_for", 0.998),
          ("B", "verb_preposition_fb_about_for", 0.998),
          ("C", "verb_preposition_fc_about_for", 0.998))
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "keep_frac": 0.85, "c_ub_max": 0.01, "unit_min": 0.8, "cost_min": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v373", "shapes": len(SHAPES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    per = R.get("per_shape", {})
    ok = bool(per) and all("error" not in s for s in per.values())
    joint = {k: s.get("joint_extraction") for k, s in per.items()}
    spec = {k: s.get("specialist_extraction") for k, s in per.items()}
    exact = {k: s.get("units_recovery") for k, s in per.items()}
    cub = {k: s.get("C_ub975") for k, s in per.items()}
    a = ok and all(v is not None and v >= B["joint_min"] for v in joint.values())
    b = ok and all(joint[k] is not None and spec[k] and joint[k] >= B["keep_frac"] * spec[k] for k in per)
    c = ok and all(v is not None and v <= B["c_ub_max"] for v in cub.values())
    d = ok and all(v is not None and v >= B["unit_min"] for v in exact.values())
    e = ok and any(joint[k] is not None and spec[k] is not None and spec[k] - joint[k] >= B["cost_min"] for k in per)
    return {"pred_a_joint_covers": bool(a), "pred_b_cost_bounded": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_units_shared": bool(d), "pred_e_generality_costs": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V373_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V373_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v373",
              "candidate_id": "corpus.unit_broad_circuit_v373", "bars": BARS,
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
