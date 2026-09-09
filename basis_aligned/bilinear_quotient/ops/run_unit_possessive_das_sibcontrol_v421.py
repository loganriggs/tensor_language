#!/usr/bin/env python3
# BQGATE: five frozen predictions; RANK, unit budget, siblings and bars fixed before the run.
"""v421: were the sibling damages an artefact of fitting on ONE behaviour, or do these behaviours share a direction?

WHAT THE TWO DAS RUNGS FOUND. v417 and v419 each fitted a rank-1 direction on a single behaviour's rows with only its
own C rows as the control, and both localised cleanly and then damaged every matched sibling:
    correlative_pair.both_vs_neither   A1 0.880, A2 0.845, own-C 0.0064; siblings 0.229 and 0.232
    possessive_number.adjacent_antecedent  A1 0.865, A2 0.922, own-C 0.0047; siblings 0.295, 0.555, 0.591, 0.679
Two independent targets, the same shape of result: the direction is specific to the CONSTRUCTION and indifferent to
which member of it produced it. But neither fit was ever ASKED to spare the siblings. Every separability rung in this
corpus fits with the siblings in the control set at control_weight 30 x n_controls, and under that recipe the
verb_particle and verb_preposition families come out separable at fam-arm leaks under 0.02. So the two DAS results are
consistent with two quite different worlds and this rung separates them.
DESIGN: the SAME cell, the SAME units, the SAME rank, fitted TWICE.
    own arm  -- controls = own C only. This reproduces v419 and is the comparison baseline.
    fam arm  -- controls = own C PLUS all four matched siblings' A1 fit rows, at the corpus's standard
                control_weight of 30 x n_controls.
Both arms are then scored the same way on rows neither fit saw: absolute A1, absolute A2, own-C damage, and damage to
each of the four siblings.
RANK IS FIXED AT 1 and registered; a null is not permission to raise it.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_own_arm_reproduces   the own arm reproduces v419 within tol = 0.05 on absolute A1 (v419: 0.865) and on the
                              largest sibling damage (v419: 0.679). This is the instrument check -- if the own arm
                              does not reproduce the receipt it is derived from, nothing else in this rung is
                              interpretable. Worked example: own-arm A1 0.84 and max sibling 0.65 both land inside
                              0.05, TRUE.                                                              prior 85%
  pred_b_fam_spares_siblings  in the fam arm, ALL FOUR siblings take absolute A1 damage <= sib_max = 0.05. This is
                              the claim that the sibling damage was a fitting artefact: given controls that ask it to
                              spare them, a rank-1 direction can.                                       prior 55%
  pred_c_fam_keeps_A1         the fam arm still carries at least floor = 0.80 of the full interchange effect on held
                              A1 rows. Sparing the siblings by giving up the behaviour is not separability, and this
                              is the clause that catches it. Worked example: fam-arm A1 0.83 with all siblings under
                              0.05 is the interesting outcome; fam-arm A1 0.41 with siblings spared says the
                              behaviours cannot be separated at rank 1.                                 prior 50%
  pred_d_cost_of_control      the fam arm's absolute A1 is within cost = 0.10 of the own arm's. Registered separately
                              from pred_c because a direction that clears 0.80 while losing 0.20 to the controls is a
                              different object from one that loses 0.02, and today's pooled rungs have repeatedly
                              shown that generality can be free -- I want the price on the record either way.
                                                                                                        prior 55%
  pred_e_C_still_spared       own-C damage UB975 <= c_ub = 0.01 in BOTH arms. Adding four controls should not break
                              the one that already held.                                                prior 80%
COUNTING. Nothing here is counted. possessive_argument already counts once as a family-less singleton; this rung
decides whether the five possessive behaviours are one circuit or five, which is a question the count currently
answers by assumption rather than by measurement -- and if pred_b and pred_c both hold, the honest follow-up is a
separability rung that would let them count properly.
Smoke: V421_SMOKE=<out.json> (CPU, V421_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_das_sibcontrol_v421_result.json"
TARGET_CELL = "possessive_adjacent"
SIBLINGS = ("possessive_medial", "possessive_attractor", "possessive_long_simple", "possessive_number")
V419_A1_ABSOLUTE = 0.865          # from circuits/followups/unit_correlative_das_v419_result.json (cell: possessive_adjacent)
V419_MAX_SIBLING = 0.679
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
RANK = 1                     # fixed in advance; a null is not permission to raise it
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"floor": 0.8, "c_ub": 0.01, "sib_max": 0.05, "tol": 0.05, "cost": 0.10, "rank": RANK}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_das_sibcontrol_v421", "cell": TARGET_CELL, "rank": RANK,
            "siblings": list(SIBLINGS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 2 * STEPS, "model_updates": 0, "fit_parameters": 2 * MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    arms = R.get("arms", {})
    ok = bool(arms) and all("error" not in a for a in arms.values())
    own, fam = arms.get("own", {}), arms.get("fam", {})

    def maxsib(a):
        s = a.get("siblings") or {}
        return max(abs(v) for v in s.values()) if s else None

    a = ok and own.get("A1_absolute") is not None and maxsib(own) is not None \
        and abs(own["A1_absolute"] - V419_A1_ABSOLUTE) <= B["tol"] \
        and abs(maxsib(own) - V419_MAX_SIBLING) <= B["tol"]
    b = ok and maxsib(fam) is not None and maxsib(fam) <= B["sib_max"]
    c = ok and (fam.get("A1_absolute") or 0) >= B["floor"]
    d = ok and fam.get("A1_absolute") is not None and own.get("A1_absolute") is not None \
        and (own["A1_absolute"] - fam["A1_absolute"]) <= B["cost"]
    e = ok and all(arm.get("C_ub975") is not None and arm["C_ub975"] <= B["c_ub"] for arm in (own, fam))
    return {"pred_a_own_arm_reproduces": bool(a), "pred_b_fam_spares_siblings": bool(b),
            "pred_c_fam_keeps_A1": bool(c), "pred_d_cost_of_control": bool(d), "pred_e_C_still_spared": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V421_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V421_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    V = dict(valid_only=True)
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]
    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float()
                                for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0)
                for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975")}

    arms = {}
    try:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{TARGET_CELL}")
        rows = {fam_: g.rows_of(m, fam_) for fam_ in ("A1", "A2", "P", "C")}
        P = {"fit": g.prepare(backend, cut(fit_half(rows["A1"])), **V),
             "held": g.prepare(backend, cut(held_half(rows["A1"])), **V),
             "A2_held": g.prepare(backend, cut(held_half(rows["A2"])), **V),
             "P_held": g.prepare(backend, cut(held_half(rows["P"]))),
             "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))),
             "C_held": g.prepare(backend, cut(held_half(rows["C"])))}
        sib_fit, sib_held = {}, {}
        for s in SIBLINGS:
            sm = importlib.import_module(f"circuit_fast_screen_candidate_{s}")
            srows = g.rows_of(sm, "A1")
            sib_fit[s] = g.prepare(backend, cut(fit_half(srows)), **V)
            sib_held[s] = g.prepare(backend, cut(held_half(srows)), **V)
        _s, _r, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET,
                                        min_gain=MIN_GAIN, max_units=max_units)
        units = list(greedy["chosen"])
        mu = mu_of(P["fit"], units)
        e_a1, e_a2 = ext(P["held"], units), ext(P["A2_held"], units)
        for arm, controls in (("own", (P["C_fit"],)),
                              ("fam", (P["C_fit"],) + tuple(sib_fit[s] for s in SIBLINGS))):
            try:
                q, _hist = g.fit_block_subspace_constrained(
                    backend, P["fit"], units, rank=RANK, steps=steps, lr=LR, seed=0, complement_weight=CW,
                    controls=controls, control_weight=LAM * len(controls), mu=mu)
                cd, pd = dmg(P["C_held"], units, q, mu), dmg(P["P_held"], units, q, mu)
                arms[arm] = {"n_controls": len(controls),
                             "A1_absolute": ext(P["held"], units, q=q),
                             "A2_absolute": ext(P["A2_held"], units, q=q),
                             "C_damage": cd, "C_ub975": cd["ce_ub975"],
                             "P_damage": pd, "P_ub975": pd["ce_ub975"],
                             "siblings": {s: dmg(sib_held[s], units, q, mu)["ce_damage"] for s in SIBLINGS}}
            except Exception as err:                               # noqa: BLE001 - recorded, never silently dropped
                arms[arm] = {"error": f"{type(err).__name__}: {err}"}
            print(f"[{arm}] {round(time.perf_counter() - t0, 1)}s", flush=True)
        meta = {"units": units, "n_units": len(units), "exact_A1": e_a1, "exact_A2": e_a2}
    except Exception as err:                                       # noqa: BLE001 - recorded, never silently dropped
        arms = {"own": {"error": f"{type(err).__name__}: {err}"}, "fam": {"error": "not reached"}}
        meta = {"error": f"{type(err).__name__}: {err}"}

    predictions = PREDS({"arms": arms})
    result = {"predictions": predictions, "schema": "unit_possessive_das_sibcontrol_v421",
              "candidate_id": "corpus.unit_possessive_das_sibcontrol_v421", "bars": BARS, "rank": RANK,
              "cell": TARGET_CELL, "siblings": list(SIBLINGS), "meta": meta, "arms": arms,
              "v419_reference": {"A1_absolute": V419_A1_ABSOLUTE, "max_sibling": V419_MAX_SIBLING},
              "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "meta": meta, "arms": arms, "seconds": result["seconds"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
