#!/usr/bin/env python3
# BQGATE: five frozen predictions; RANK, unit budget, siblings and bars fixed before the run.
"""v419: DAS on possessive_number.adjacent_antecedent -- the second named standing target, and the one with the
sibling structure the protocol actually asks for.

WHY THIS TARGET. The standing direction names three first targets; v417 did the first and found that
correlative_pair.both_vs_neither localises to a rank-1 subspace (0.880 of the full effect on held A1, 0.845 on the A2
construction it never saw) but damages BOTH its matched siblings by 0.229 and 0.232 -- a correlative-slot direction
rather than a both-versus-neither one, and a localisation without selectivity. The second target is the one the
direction itself flags as having FIVE matched siblings, which is the structure that makes the selectivity question
answerable rather than assumed. Four of them are usable here:
    possessive_medial        possessive_number.medial_antecedent
    possessive_attractor     possessive_number.animate_attractor
    possessive_long_simple   possessive_number.long_simple_intervener
    possessive_number        possessive_number.their_vs_his
All four are the SAME possessive-number behaviour with a different antecedent configuration, so a direction that
encodes "the possessive slot" damages them and one that encodes "this antecedent's number" does not. That is a control
set capable of failing, which is the standing lesson the correlative rung just demonstrated the value of.
RANK IS FIXED AT 1 and registered; a null is not permission to raise it.
THREE PREDICATE LESSONS FROM TODAY ARE APPLIED, and one of them is a bar I am NOT setting:
  * pred_a and pred_b are on the ABSOLUTE fraction of the interchange effect the subspace carries, not on extraction
    normalised by the unit set's own reach (v415: the asymmetry I could not see was entirely in the denominator).
  * P is measured and written to the receipt but carries NO BAR. At v417 I set a removal bar on P citing the
    interchange screen's P = 0.024 -- a number from a DIFFERENT INSTRUMENT -- and P came in at 1.78. No same-instrument
    prior number exists for this cell either, so the honest form is to report P and register nothing on it.
  * the sibling clause is split in two: pred_d asks for MOST siblings spared (k_sibs = 3 of 4) and pred_e for ALL of
    them, so a direction that is specific against three configurations and not the fourth is visible as a partial
    result instead of collapsing into one verdict.

REGISTERED BEFORE THE RUN (bars in BARS; each coded predicate is the sentence here):
  pred_a_A1_absolute       the rank-1 subspace carries at least floor = 0.80 of the FULL interchange effect on held A1
                           rows. Worked example: a unit set reaching 0.90 with a direction extracting 0.95 of it gives
                           0.855, TRUE; a set at 0.60 with extraction 0.99 gives 0.594, FALSE.           prior 55%
  pred_b_A2_absolute       the same subspace carries at least 0.80 on the A2 construction, which the fit never saw.
                                                                                                        prior 45%
  pred_c_C_spared          own-C removal damage UB975 <= c_ub = 0.01 on held C rows.                     prior 60%
  pred_d_most_siblings_spared  at least k_sibs = 3 of the 4 matched siblings take absolute A1 damage <= sib_max = 0.05.
                           v417's correlative subspace damaged 2 of 2 at 0.23, so this bar is not one this corpus
                           passes automatically.                                                         prior 40%
  pred_e_all_siblings_spared  ALL FOUR siblings take damage <= 0.05. This is the full selectivity claim the protocol
                           asks for.                                                                     prior 30%
COUNTING. possessive_argument already counts once as a family-less singleton. Nothing here adds to the count; a pass
would promote it from interchange-localised to subspace-localised with matched-sibling controls, and a failure of
pred_d and pred_e would say the possessive-number behaviours share one direction the way the correlative trio does --
which the count already assumes for correlatives and does NOT assume for these.
Smoke: V419_SMOKE=<out.json> (CPU, V419_SMOKE_ROWS=4 per split).
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
OUT = ROOT / "circuits/followups/unit_possessive_das_v419_result.json"
TARGET_CELL = "possessive_adjacent"
SIBLINGS = ("possessive_medial", "possessive_attractor", "possessive_long_simple", "possessive_number")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
RANK = 1                     # fixed in advance; a null is not permission to raise it
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
BARS = {"floor": 0.8, "c_ub": 0.01, "sib_max": 0.05, "k_sibs": 3, "rank": RANK}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_possessive_das_v419", "cell": TARGET_CELL, "rank": RANK,
            "siblings": list(SIBLINGS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": STEPS, "model_updates": 0, "fit_parameters": MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    r = R.get("result", {})
    ok = bool(r) and "error" not in r
    a = ok and (r.get("A1_absolute") or 0) >= B["floor"]
    b = ok and (r.get("A2_absolute") or 0) >= B["floor"]
    c = ok and r.get("C_ub975") is not None and r["C_ub975"] <= B["c_ub"]
    sibs = (r.get("siblings") or {})
    d = ok and bool(sibs) and sum(1 for v in sibs.values() if abs(v) <= B["sib_max"]) >= B["k_sibs"]
    e = ok and bool(sibs) and max(abs(v) for v in sibs.values()) <= B["sib_max"]
    return {"pred_a_A1_absolute": bool(a), "pred_b_A2_absolute": bool(b), "pred_c_C_spared": bool(c),
            "pred_d_most_siblings_spared": bool(d), "pred_e_all_siblings_spared": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V419_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V419_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
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

    try:
        m = importlib.import_module(f"circuit_fast_screen_candidate_{TARGET_CELL}")
        rows = {fam: g.rows_of(m, fam) for fam in ("A1", "A2", "P", "C")}
        P = {"fit": g.prepare(backend, cut(fit_half(rows["A1"])), **V),
             "held": g.prepare(backend, cut(held_half(rows["A1"])), **V),
             "A2_held": g.prepare(backend, cut(held_half(rows["A2"])), **V),
             "P_held": g.prepare(backend, cut(held_half(rows["P"]))),
             "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))),
             "C_held": g.prepare(backend, cut(held_half(rows["C"])))}
        _s, _r, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET,
                                        min_gain=MIN_GAIN, max_units=max_units)
        units = list(greedy["chosen"])
        mu = mu_of(P["fit"], units)
        q, _hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=RANK, steps=steps, lr=LR, seed=0,
                                                    complement_weight=CW, controls=(P["C_fit"],),
                                                    control_weight=LAM, mu=mu)
        e_a1, e_a2 = ext(P["held"], units), ext(P["A2_held"], units)
        a1_abs, a2_abs = ext(P["held"], units, q=q), ext(P["A2_held"], units, q=q)
        cd, pd = dmg(P["C_held"], units, q, mu), dmg(P["P_held"], units, q, mu)
        sibs = {}
        for s in SIBLINGS:
            sm = importlib.import_module(f"circuit_fast_screen_candidate_{s}")
            ps = g.prepare(backend, cut(held_half(g.rows_of(sm, "A1"))), **V)
            sibs[s] = dmg(ps, units, q, mu)["ce_damage"]
        result_body = {"units": units, "n_units": len(units),
                       "exact_A1": e_a1, "exact_A2": e_a2,
                       "A1_absolute": a1_abs, "A2_absolute": a2_abs,
                       "A1_normalised": round(a1_abs / e_a1, 3) if abs(e_a1) > 1e-6 else None,
                       "A2_normalised": round(a2_abs / e_a2, 3) if abs(e_a2) > 1e-6 else None,
                       "C_damage": cd, "C_ub975": cd["ce_ub975"], "P_damage": pd, "P_ub975": pd["ce_ub975"],
                       "siblings": sibs}
    except Exception as err:                                       # noqa: BLE001 - recorded, never silently dropped
        result_body = {"error": f"{type(err).__name__}: {err}"}

    predictions = PREDS({"result": result_body})
    result = {"predictions": predictions, "schema": "unit_possessive_das_v419",
              "candidate_id": "corpus.unit_possessive_das_v419", "bars": BARS, "rank": RANK,
              "cell": TARGET_CELL, "result": result_body,
              "seconds": round(time.perf_counter() - t0, 1),
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    target = Path(smoke) if smoke else OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "result": result_body, "seconds": result["seconds"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
