#!/usr/bin/env python3
# BQGATE: five frozen predictions; RANK, unit budget, siblings and bars fixed before the run.
"""v423: the same own/fam refit on the CORRELATIVE trio, where the count already assumes one circuit.

WHY THIS IS THE PAIR TO RUN. v421 refitted the possessive subspace twice over the same units at the same rank -- once
with own C only, once with the four matched siblings added to the control set -- and the result overturned the reading
of v419: the own arm reproduced it exactly (A1 0.865, siblings 0.295 to 0.679) while the fam arm spared those same
siblings at -0.045, -0.050, +0.046 and +0.053 and still carried 0.814 of the effect. The sibling damage was the
OBJECTIVE, not the model. So the five possessive behaviours look separable at rank 1.
The correlative trio is the case where the corpus's count rule already says the opposite. ops/circuit_count.py counts
correlative_both_either / both_neither / either_neither as ONE group under R5 -- a rule adopted from an early
interchange result and never re-tested -- and v417 found the same non-selectivity there (siblings 0.229 and 0.232)
that turned out to be an artefact for possessive. Running the identical two-arm refit here decides between:
    the rule is RIGHT and measured  -- the fam arm cannot both spare the siblings and keep A1, so the three
                                       behaviours genuinely share one direction and counting them once is correct;
    the rule is WRONG and inherited -- the fam arm does both, as it did for possessive, and the corpus has been
                                       counting three circuits as one.
Either way the count stops resting on an assumption. RANK IS FIXED AT 1 and registered; a null is not permission to
raise it, and the bars are identical to v421's so the two rungs are directly comparable.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_own_arm_reproduces   the own arm reproduces v417 within tol = 0.05 on absolute A1 (v417: 0.880) and on the
                              largest sibling damage (v417: 0.2315). The instrument check: without it nothing else in
                              this rung is interpretable, and it is the clause that passed at v421 and made that
                              comparison usable.                                                        prior 85%
  pred_b_fam_spares_siblings  in the fam arm, BOTH siblings take absolute A1 damage <= sib_max = 0.05.  prior 55%
  pred_c_fam_keeps_A1         the fam arm still carries at least floor = 0.80 of the full interchange effect on held
                              A1 rows. Sparing the siblings by giving up the behaviour is not separability.
                                                                                                        prior 50%
  pred_d_cost_of_control      the fam arm's absolute A1 is within cost = 0.10 of the own arm's. At v421 this cost was
                              0.051 for four controls; two controls should cost no more.                prior 65%
  pred_e_C_still_spared       own-C damage UB975 <= c_ub = 0.01 in BOTH arms.                            prior 80%
COUNTING. If pred_b and pred_c both hold, the R5 group clause is counting three circuits as one and the corpus is
UNDERSTATED by two -- which I would take to the board as a measured proposal rather than change myself, because
SINGLETON_GROUPS is a registered rule. If either fails, the clause is confirmed by measurement for the first time.
Smoke: V423_SMOKE=<out.json> (CPU, V423_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_correlative_das_sibcontrol_v423_result.json"
TARGET_CELL = "correlative_pair"
SIBLINGS = ("correlative_but_and", "correlative_or_and")
V417_A1_ABSOLUTE = 0.880          # from circuits/followups/unit_correlative_das_v417_result.json
V417_MAX_SIBLING = 0.2315
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
RANK = 1                     # fixed in advance; a null is not permission to raise it
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"floor": 0.8, "c_ub": 0.01, "sib_max": 0.05, "tol": 0.05, "cost": 0.10, "rank": RANK}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_correlative_das_sibcontrol_v423", "cell": TARGET_CELL, "rank": RANK,
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
        and abs(own["A1_absolute"] - V417_A1_ABSOLUTE) <= B["tol"] \
        and abs(maxsib(own) - V417_MAX_SIBLING) <= B["tol"]
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
    smoke = os.environ.get("V423_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V423_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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
            except Exception as arm_err:                           # noqa: BLE001 - recorded, never silently dropped
                arms[arm] = {"error": f"{type(arm_err).__name__}: {arm_err}"}
            print(f"[{arm}] {round(time.perf_counter() - t0, 1)}s", flush=True)
        meta = {"units": units, "n_units": len(units), "exact_A1": e_a1, "exact_A2": e_a2}
    except Exception as err:                                       # noqa: BLE001 - recorded, never silently dropped
        arms = {"own": {"error": f"{type(err).__name__}: {err}"}, "fam": {"error": "not reached"}}
        meta = {"error": f"{type(err).__name__}: {err}"}

    predictions = PREDS({"arms": arms})
    result = {"predictions": predictions, "schema": "unit_correlative_das_sibcontrol_v423",
              "candidate_id": "corpus.unit_correlative_das_sibcontrol_v423", "bars": BARS, "rank": RANK,
              "cell": TARGET_CELL, "siblings": list(SIBLINGS), "meta": meta, "arms": arms,
              "v417_reference": {"A1_absolute": V417_A1_ABSOLUTE, "max_sibling": V417_MAX_SIBLING},
              "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "meta": meta, "arms": arms, "seconds": result["seconds"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
