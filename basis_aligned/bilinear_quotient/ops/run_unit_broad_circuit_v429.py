#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v429: how many DISTINCT cue mappings can one construction hold before a single direction stops covering them?

THE QUESTION NOTHING TODAY HAS ANSWERED. Every pooled test so far has held the MAPPING fixed and varied something
else -- the sentence frame (v371-v383, up to eleven shapes of one mapping), or the behaviour within a construction
where the mapping barely changes (v425's correlative trio, v427's possessive five, both covered by one direction).
This asks the orthogonal question: five DIFFERENT cue -> token mappings, in ONE frame, in one construction.
    several / each     -> ` crates` / ` crate`
    many / one         -> ` ropes` / ` rope`
    numerous / another -> ` barrels` / ` barrel`
    several / every    -> ` ropes` / ` rope`
    many / each        -> ` barrels` / ` barrel`
These five are the determiner-number class I built this afternoon. They share the construction, the frame, the
position and the P control, and differ in exactly the thing the corpus treats as a behaviour's identity: which cue
word selects which readout token. v321 and v325 established that the mapping IS the identity -- cells sharing a
readout pair but not a mapping are separable, cells sharing mappings fuse -- so if one direction covers all five,
"identity" means something weaker than the corpus has been assuming, and if it does not, this is the first pooled
test today to find a limit.
WHY THIS IS THE RIGHT TEST TO SPEND ON. Thirteen pooled rungs have run and not one has found a shape or a behaviour
a single direction could not cover. Every one of them varied a dimension the model might reasonably be indifferent to.
This varies the one dimension the corpus's entire counting scheme rests on.
    v413 already looked at these five cells, but PER DIRECTION and on the greedy site set only -- it found the
    each-to-several direction shares a four-unit set across all five while several-to-each does not. This rung pools
    BOTH directions and fits an actual rank-1 direction, which v413 did not.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers      the joint direction reaches extraction >= joint_min = 0.80 on ALL FIVE mappings, scored
                           per cell so none is carried by the others. If this holds, five distinct cue -> token
                           mappings share one rank-1 direction and the mapping is not the unit of identity it has
                           been taken for.                                                               prior 40%
  pred_b_units_transfer    the pooled greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every cell.
                           v413 measured the both-direction sets at 0.768-0.819 per cell on dirA alone, so this bar
                           is genuinely in doubt.                                                        prior 45%
  pred_c_control_clean     own-C damage UB975 <= c_ub_max = 0.01 on all five.                            prior 55%
  pred_d_counted_shape_kept  several_each_crates -- the only member with a four-row pass -- reaches >= 0.80 under the
                           joint direction.                                                              prior 60%
  pred_e_some_shape_resists  at least one cell falls below 0.80. Registered ABOVE pred_a for the second time today,
                           because thirteen consecutive rungs of universal coverage came from varying dimensions the
                           model is indifferent to, and this is the first that varies the one it should not be.
                                                                                                          prior 60%
COUNTING. Nothing here is counted -- the determiner class has one four-row pass and no family, so it counts zero
either way. What the result changes is the READING of every merged group in the corpus: if distinct mappings pool as
readily as frames do, then rule R7's merges and the R5 group clauses are all instances of one fact about this model
rather than three separate observations.
Smoke: V429_SMOKE=<out.json> (CPU, V429_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v429_result.json"
SHAPES = (("several_each_crates", "determiner_number_crates", None),
          ("many_one_ropes", "determiner_number_ropes", None),
          ("numerous_another_barrels", "determiner_number_barrels", None),
          ("several_every_ropes", "determiner_number_ropes_d", None),
          ("many_each_barrels", "determiner_number_barrels_d", None))
COUNTED_SHAPE = "several_each_crates"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "c_ub_max": 0.01, "unit_min": 0.8}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v429", "shapes": len(SHAPES),
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
    smoke = os.environ.get("V429_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V429_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v429",
              "candidate_id": "corpus.unit_broad_circuit_v429", "bars": BARS,
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
