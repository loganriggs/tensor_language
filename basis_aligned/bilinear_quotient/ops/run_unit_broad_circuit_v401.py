#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v401: DOES BREADTH GENERALISE PAST VERB-CUE CIRCUITS? Four shapes of the corpus's first determiner-number circuit.

WHY THIS IS THE TEST THAT MATTERS TODAY. Everything measured since 13:00 -- frame copies merging, breadth of seven
then ten and eleven shapes, out-of-sample transfer at 0.951 to 0.996, nine discarded cells recovered, the circuit
turning out to be a recency reader -- came from ONE kind of behaviour: a lexical verb selecting a particle or a
preposition. v397 gave the corpus its first circuit of a different kind, and it passed all four rows on the first
attempt (n_units 8, held 0.898, own-C UB975 0.0026): the cue is a FUNCTION WORD (several / each) and the readout is a
CONTENT noun's NUMBER, which is the opposite assignment of roles from every other entry.
    A_inspect     `Near the {object} the {agent} inspected several damaged`   -> ` crates`   counted at v397
    B_agreement   `Everyone agreed the {agent} inspected several damaged`
    C_report      `It turned out the {agent} inspected several damaged`
    D_denial      `Nobody doubted the {agent} inspected several damaged`
The three variants have no battery of their own -- under today's evidence a shape variant's battery is a formality
(v393 recovered nine cells that missed a row and every one was covered by its mapping's pooled direction), so what
they carry is a CPU capability screen: A1 -3.60/+3.74, -3.83/+3.92 and -3.29/+3.34, zero dropped rows each.
IF THIS FAILS IT IS THE MOST USEFUL FAILURE OF THE DAY. Twelve pooled rungs have found nothing below 0.80, and every
one of them was a verb-cue circuit. A determiner-number circuit whose shapes do NOT pool would say the breadth result
is a property of that construction class rather than of this model, which is exactly the qualification the corpus
would otherwise be missing.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers      the joint direction reaches extraction >= joint_min = 0.80 on ALL FOUR shapes, scored per
                           shape so no shape is carried by the others.                                   prior 55%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape.
                           A new construction may not share components across frames even if the verb-cue ones do.
                                                                                                        prior 55%
  pred_c_control_clean     own-C damage UB975 <= 0.01 on all four. v397's specialist managed 0.0026, and three of the
                           eight pooled directions at v393 breached this bound, so it is not a formality. prior 55%
  pred_d_counted_shape_kept  the frame-A shape -- the one that actually counts -- still reaches >= 0.80 under the
                           joint direction. A pool must not buy three uncounted variants by giving up the entry.
                                                                                                        prior 80%
  pred_e_some_shape_resists  at least one shape falls below 0.80. Registered at a prior above pred_a because twelve
                           consecutive rungs of universal coverage all came from one construction class, and the
                           first test outside it is where a class-specific result should break.          prior 50%
COUNTING. Nothing here is counted. determiner_number_crates counts once already; a pass gives it a breadth of four
and gives the corpus its first breadth figure outside verb-cue circuits.
Smoke: V401_SMOKE=<out.json> (CPU, V401_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v401_result.json"
SHAPES = (("A_inspect", "determiner_number_crates", 0.898),
          ("B_agreement", "determiner_number_fb_crates", None),
          ("C_report", "determiner_number_fc_crates", None),
          ("D_denial", "determiner_number_fd_crates", None))
COUNTED_SHAPE = "A_inspect"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v401", "shapes": len(SHAPES),
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
    d = ok and jx.get(COUNTED_SHAPE) is not None and jx[COUNTED_SHAPE] >= B["joint_min"]
    e = ok and any(v is not None and v < B["joint_min"] for v in jx.values())
    return {"pred_a_joint_covers": bool(a), "pred_b_units_transfer": bool(b), "pred_c_control_clean": bool(c),
            "pred_d_counted_shape_kept": bool(d), "pred_e_some_shape_resists": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V401_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V401_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v401",
              "candidate_id": "corpus.unit_broad_circuit_v401", "bars": BARS,
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
