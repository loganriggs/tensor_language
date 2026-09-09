#!/usr/bin/env python3
# BQGATE: five frozen predictions; shapes, mapping pair, unit budget and bars fixed before the run.
"""v371: CAN ONE DIRECTION COVER FOUR SENTENCE SHAPES? The first BROAD circuit in this corpus.

WHY THIS EXISTS. Every circuit I have counted is fitted and evaluated inside ONE sentence shape:
`[prefix] the {agent} {verb}, [parenthetical],` with the prefix and parenthetical held fixed per cell. Today's frame
work showed that is not a cosmetic restriction. The same cue -> token mappings in a different frame come out
SEPARABLE from the original at rank 1 -- fam-arm leak 0.003 to 0.024 while the two collide by 0.29 to 1.34
uncontrolled (v337, v341, v355, v359, thirteen mapping pairs) -- and frames fall into equivalence classes, with four
report/evidential shapes behaving as one and the locative, agreement and denial shapes each behaving as their own
(v363, v367). So "circuit" in this corpus has meant "this mapping, in this register", and a reader could fairly say
the corpus counts templates rather than mechanisms.
This runner asks the question that follows: is there a SINGLE unit set and a SINGLE rank-1 direction that carries
woke -> ` up` / calmed -> ` down` across four structurally distinct shapes at once?
    A  `Near the {object} the {agent} {verb}, of course,`      locative adjunct
    B  `Everyone agreed the {agent} {verb}, frankly,`          agreement matrix
    C  `It turned out the {agent} {verb}, apparently,`         report/evidential matrix
    D  `Nobody doubted the {agent} {verb}, honestly,`          denial matrix
One greedy unit set is chosen on the POOLED fit rows of all four shapes, one constrained rank-1 direction is fitted on
those pooled rows against the pooled C controls, and then extraction and collateral are measured PER SHAPE on that
shape's held-out rows. The specialists are on disk for comparison: the per-shape cdas directions reach 0.991 (A),
0.998 (B), 0.998 (C) and 0.998 (D) of their own exact-set effect.
WHAT THE OUTCOMES MEAN. If one direction covers all four near the specialists' level, the narrow circuits are four
views of one broader object and the corpus should be reporting the broad one. If it covers some and not others, the
shapes that resist name where the mechanism genuinely differs. If it covers none, then the per-shape directions are
the real objects and the honest description of a corpus entry is "mapping x register", which I would rather state
plainly than leave implied by a count.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_joint_covers    the joint direction reaches extraction >= joint_min = 0.80 of the exact-set effect on the
                         held-out rows of ALL FOUR shapes. This is the breadth claim in one number, and it is scored
                         per shape rather than pooled so that a shape cannot be carried by the others. prior 45%
  pred_b_cost_bounded    on every shape, the joint direction's extraction is at least keep_frac = 0.85 of that
                         shape's SPECIALIST extraction, taken from the receipts already on disk (0.991, 0.998, 0.998,
                         0.998). Worked example: a specialist at 0.998 and a joint at 0.87 gives a ratio of 0.872,
                         above 0.85, TRUE for that shape; a joint at 0.80 gives 0.80, FALSE.        prior 40%
  pred_c_control_clean   the joint direction's own-C damage UB975 stays <= c_ub_max = 0.01 on all four shapes, i.e.
                         breadth is not bought by a direction that damages the lexical control everywhere.  prior 65%
  pred_d_units_shared    the POOLED greedy unit set reaches exact-set recovery >= unit_min = 0.80 on every shape's
                         held rows. This separates two ways of failing: if the units do not transfer, no direction
                         over them could, and the right conclusion is that the shapes use different components; if
                         the units transfer but the direction does not, the components are shared and only the
                         direction is register-specific, which is what v337-v367 imply.             prior 70%
  pred_e_generality_costs  on at least one shape the joint direction gives up at least cost_min = 0.05 of extraction
                         against that shape's specialist. Registered ALONGSIDE pred_b rather than against it: both
                         can be true, and a cost between 5% and 15% is the outcome where a broad circuit exists AND
                         the registers still differ measurably. I am saying that here because at v367 I wrote that two
                         predicates were mutually exclusive, both came out true, and the exclusivity was the error.
                                                                                                    prior 75%
COUNTING. Nothing here is counted as a new circuit. The four per-shape circuits are already counted; this rung asks
whether they should have been counted as ONE, and a strong pass is an argument for revising the count DOWNWARD, not
for adding to it. That is the honest direction for a result like this and it is registered before the run.
Smoke: V371_SMOKE=<out.json> (CPU, V371_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_broad_circuit_v371_result.json"
SHAPES = (("A", "verb_particle_up_down", 0.991),
          ("B", "verb_particle_fb_up_down", 0.998),
          ("C", "verb_particle_fc_up_down", 0.998),
          ("D", "verb_particle_fd_up_down", 0.998))
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"joint_min": 0.8, "keep_frac": 0.85, "c_ub_max": 0.01, "unit_min": 0.8, "cost_min": 0.05}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_broad_circuit_v371", "shapes": len(SHAPES),
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
    smoke = os.environ.get("V371_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V371_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
    result = {"predictions": predictions, "schema": "unit_broad_circuit_v371",
              "candidate_id": "corpus.unit_broad_circuit_v371", "bars": BARS,
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
