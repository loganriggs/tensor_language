#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v431: five mappings onto DIFFERENT variables -- the test that can refute the story v429 invites.

WHAT v429 FOUND AND WHAT IT DID NOT. One 7-unit set and one rank-1 direction covered five distinct cue -> token
mappings at 0.965 to 1.011 (own-C 0.0193, nearly double the bar, which is the caveat). But all five of those
mappings encode the SAME abstract variable -- grammatical number -- realised on different nouns: several/many/numerous
set plural, each/one/another/every set singular. The tidy account is that the unit of identity is the VARIABLE rather
than the (cue -> token) mapping, and the ledger deliberately does not contain that sentence, because three
after-the-fact accounts written today were each refuted by the next screen within the hour.
This rung is that account's first real chance to fail. Five counted verb_particle circuits, all in the SAME frame,
each mapping onto a DIFFERENT variable -- which particle the verb selects, and the particles differ:
    up_down     woke -> ` up`     / calmed -> ` down`
    out_down    found -> ` out`   / settled -> ` down`
    away_up     shied -> ` away`  / sidled -> ` up`
    down_in     hunkered -> ` down` / caved -> ` in`
    out_up      blurted -> ` out` / brightened -> ` up`
These are five entries the corpus counts SEPARATELY, and v317 measured them mutually separable at fam-arm leaks under
0.02 with their siblings as controls. The determiner five were not counted separately and shared a direction. If these
five ALSO pool, then the variable story is wrong, pooled coverage is close to universal in this model, and the
corpus's distinctness criterion needs restating from the ground up rather than patching -- which would be the largest
consequence any rung today could produce, and is why it is worth the minute.
    v317 evidence, for contrast: each of these five spares the other four at fam-arm leak <= 0.02 when fitted with
    them as controls. That is the SEPARABILITY criterion. This rung applies the POOLED criterion, which has disagreed
    with separability in both directions today.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers      the joint direction reaches extraction >= joint_min = 0.80 on ALL FIVE mappings. If this
                           holds for mappings onto different variables, the variable story is refuted and pooled
                           coverage is essentially unbounded within a construction.                      prior 35%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every cell.
                                                                                                        prior 45%
  pred_c_control_clean     own-C damage UB975 <= c_ub_max = 0.01 on all five. v429's shared direction came in at
                           0.0193 where its per-cell directions were 0.0014-0.0083, so a pooled direction paying for
                           coverage with control damage is now an observed pattern rather than a worry. prior 40%
  pred_d_counted_shape_kept  up_down, the oldest counted member, reaches >= 0.80 under the joint direction.
                                                                                                        prior 55%
  pred_e_some_shape_resists  at least one of the five falls below 0.80. Registered ABOVE pred_a because these five
                           map onto genuinely different variables and are the corpus's clearest case of distinct
                           circuits; if nothing resists HERE, fourteen rungs of universal coverage stop looking like
                           a series of facts about particular families and start looking like one fact about the
                           instrument.                                                                   prior 65%
COUNTING. Nothing here is counted; all five already count. But if pred_a holds and pred_e fails, the corpus's count of
139 rests on a distinctness criterion this rung would have refuted, and the honest next step would be a board proposal
to re-derive the count under the pooled criterion -- not an edit I would make alone.
Smoke: V431_SMOKE=<out.json> (CPU, V431_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v431_result.json"
SHAPES = (("up_down", "verb_particle_up_down", None),
          ("out_down", "verb_particle_out_down", None),
          ("away_up", "verb_particle_away_up", None),
          ("down_in", "verb_particle_down_in", None),
          ("out_up", "verb_particle_out_up", None))
COUNTED_SHAPE = "up_down"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v431", "shapes": len(SHAPES),
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
    smoke = os.environ.get("V431_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V431_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v431",
              "candidate_id": "corpus.unit_broad_circuit_v431", "bars": BARS,
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
