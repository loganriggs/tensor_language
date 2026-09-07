#!/usr/bin/env python3
# BQGATE: five frozen predictions; the amended tier-3 battery (v192-v195) on the four row-2 misses + one control; bars fixed before the run.
"""v196: the AMENDED tier-3 battery -- within-direction split, relaxed greedy, row 5 by own direction / own ceiling.

What changed and why (each change has a receipt):
  * split WITHIN interchange direction (v195: parity == direction, so 'EVEN fit / ODD eval' was a direction test):
    FIT = rows[0::4] + rows[1::4], HELD = rows[2::4] + rows[3::4]; row 2 is the MIN over the two held-out directions.
  * greedy gain floor 0.005 with pool 40 / max 30 units (v195: the hard direction is a tail of small heads that min_gain 0.02
    discards; relaxed hard-direction sets reached 0.81-0.90 there and 0.90-1.00 on the easy direction).
  * row 5 = the A2 construction's OWN diff-in-means direction, as a share of A2's own full-rank mean-ablation ceiling, band
    [0.5, 1.4] (v192-v194: A1-fitted directions are construction-keyed on shared heads; a construction's ceiling can be <0.5x
    the other's). The A1-direction share is REPORTED (keyed vs shared), not scored.
  Rows 3-4 unchanged (dim A1 CE damage LB975>0 and >=0.10 on held A1; constrained rank-1 own-C UB975 <= 0.01 on held C, control
  = C FIT rows). Bars are the v113 bars where the row is unchanged: EXT_MIN 0.80 (not relaxed), REM_MIN 0.10, C_UB_MAX 0.01.
Behaviours: degree_result, finiteness_selection, correlative_both_either, polarity_state (v113 row-2 misses) + quantifier_number
(v113 clean pass; control). Magnitudes this rung rests on are the v195 receipt's (32-row fits); the half-size FIT here is the
run's own outcome -- no probe, and the greedy is re-run, so no arm is pre-measured except the instrument.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_row2_amended  -> held-out per-direction exact-set recovery has MIN >= 0.80 on quantifier_number and on at least 3 of
                          the 4 misses (a half-direction fit of 8+8 rows may fall short of v195's 16+16 on one).
  pred_b_row3          -> dim A1 CE damage on held A1 has LB975 > 0 and damage >= 0.10 on all five.
  pred_c_row4          -> constrained rank-1 own-C CE UB975 <= 0.01 on held C on all five (v113 passed row 4 on all five).
  pred_d_row5_amended  -> A2-own-direction CE / A2 full-rank ceiling within [0.5, 1.4] with LB975 > 0 on all five.
  pred_e_instrument    -> the v113 EVEN set's exact recovery on the held EVEN-direction rows (rows[2::4], a subset of v113's
                          EVEN rows) is within +-0.08 of v113's extraction_even on all five.
Circuit accounting if a-d hold: four behaviours move from partial (rows 3-5 or 3-4) to full passes under the AMENDED battery,
recorded as amended-protocol passes (the v113 rows stand as v113 rows). Per-behaviour try/except records an error receipt.
Smoke: V196_SMOKE=<out.json> (CPU, V196_SMOKE_ROWS=4 per split, V196_SMOKE_NAMES=quantifier_number; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_v196_result.json"
V113 = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
NAMES = {"degree_result": "degree_result", "finiteness_selection": "finiteness", "correlative_both_either": "both_either",
         "polarity_state": "polarity_state", "quantifier_number": "quantifier_number"}
MISSES = ("degree_result", "finiteness_selection", "correlative_both_either", "polarity_state")
CONTROL = "quantifier_number"
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "row2_min_misses": 3, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "instr_tol": 0.08}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40000, 1280000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_v196", "behaviours": 5, "constructions": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 5 * STEPS, "model_updates": 0, "fit_parameters": 5 * MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "rows" in r and "error" not in r}
    ok = bool(good)  # an empty or all-error receipt fails everything
    a = ok and CONTROL in good and good[CONTROL]["rows"]["row2"] and sum(good[n]["rows"]["row2"] for n in MISSES if n in good) >= B["row2_min_misses"]
    b = ok and all(good[n]["rows"]["row3"] for n in NAMES if n in good) and len(good) == len(NAMES)
    c = ok and all(good[n]["rows"]["row4"] for n in NAMES if n in good) and len(good) == len(NAMES)
    d = ok and all(good[n]["rows"]["row5"] for n in NAMES if n in good) and len(good) == len(NAMES)
    e = ok and all(abs(good[n]["instrument_v113_set_held_even_dir"] - good[n]["v113_extraction_even"]) <= B["instr_tol"] for n in NAMES if n in good) and len(good) == len(NAMES)
    return {"pred_a_row2_amended": bool(a), "pred_b_row3": bool(b), "pred_c_row4": bool(c), "pred_d_row5_amended": bool(d), "pred_e_instrument": bool(e)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V196_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V196_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V196_SMOKE_NAMES", "quantifier_number").split(",")]
    parent = json.loads(V113.read_text())["behaviours"]
    fit_half = lambda rows: rows[0::4] + rows[1::4]
    held_half = lambda rows: rows[2::4] + rows[3::4]

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    ext = lambda p, units, q=None: round(g.recovery(p, g.patched_axis(backend, p, list(units), q=q)), 3)
    R = {}
    for n in which:
        t1 = time.perf_counter()
        try:
            m = importlib.import_module(f"circuit_fast_screen_candidate_{NAMES[n]}")
            rows = {fam: g.rows_of(m, fam) for fam in ("A1", "A2", "P", "C")}
            P = {"fit": g.prepare(backend, cut(fit_half(rows["A1"]))), "held": g.prepare(backend, cut(held_half(rows["A1"]))),
                 "held_dirA": g.prepare(backend, cut(rows["A1"][2::4])), "held_dirB": g.prepare(backend, cut(rows["A1"][3::4])),
                 "A2_fit": g.prepare(backend, cut(fit_half(rows["A2"]))), "A2_held": g.prepare(backend, cut(held_half(rows["A2"]))),
                 "P_held": g.prepare(backend, cut(held_half(rows["P"]))), "C_fit": g.prepare(backend, cut(fit_half(rows["C"]))), "C_held": g.prepare(backend, cut(held_half(rows["C"])))}
            singles, ranked, greedy = g.greedy_heads(backend, P["fit"], pool=pool, target=TARGET, min_gain=MIN_GAIN, max_units=max_units)
            units = list(greedy["chosen"])
            e_fit, e_held, e_a, e_b = ext(P["fit"], units), ext(P["held"], units), ext(P["held_dirA"], units), ext(P["held_dirB"], units)
            mu1 = mu_of(P["fit"], units)
            q1 = g.block_diff_in_means(backend, P["fit"], units)
            qc, hist = g.fit_block_subspace_constrained(backend, P["fit"], units, rank=1, steps=steps, lr=LR, seed=0, complement_weight=CW, controls=(P["C_fit"],), control_weight=LAM, mu=mu1)
            arms = {}
            for arm, qa in (("dim", q1), ("cdas", qc)):
                arms[arm] = {fam: dmg(P[k], units, qa, mu1) for fam, k in (("A1", "held"), ("A2", "A2_held"), ("P", "P_held"), ("C", "C_held"))}
                arms[arm]["extraction_held"] = round(ext(P["held"], units, q=qa) / e_held, 3) if abs(e_held) > 1e-6 else None
            mu2 = mu_of(P["A2_fit"], units)
            q2 = g.block_diff_in_means(backend, P["A2_fit"], units)
            a2_own = dmg(P["A2_held"], units, q2, mu2)
            a2_full = dmg(P["A2_held"], units, None, mu2)   # q=None: full-rank mean-ablation of the set = the construction's own ceiling
            a1_full = dmg(P["held"], units, None, mu1)
            share = lambda x, y: round(x / y, 4) if y else None
            d = arms["dim"]
            row5_share = share(a2_own["ce_damage"], a2_full["ce_damage"])
            rows_ok = {"row2": min(e_a, e_b) >= EXT_MIN,
                       "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                       "row4": arms["cdas"]["C"]["ce_ub975"] <= C_UB_MAX,
                       "row5": a2_own["ce_lb975"] > 0 and row5_share is not None and BARS["row5_share_band"][0] <= row5_share <= BARS["row5_share_band"][1]}
            R[n] = {"units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
                    "direction_A": str(P["held_dirA"].rows[0].get("direction_id")), "direction_B": str(P["held_dirB"].rows[0].get("direction_id")),
                    "extraction_fit": e_fit, "extraction_held": e_held, "extraction_held_dirA": e_a, "extraction_held_dirB": e_b,
                    "arms": arms, "a2_own": a2_own, "a2_full_ceiling": a2_full, "a1_full_ceiling": a1_full,
                    "row5_share_own": row5_share, "row5_share_a1_direction": share(d["A2"]["ce_damage"], a2_full["ce_damage"]),
                    "a1_share_dim_over_full": share(d["A1"]["ce_damage"], a1_full["ce_damage"]), "old_row5_ratio": share(d["A2"]["ce_damage"], d["A1"]["ce_damage"]),
                    "instrument_v113_set_held_even_dir": ext(P["held_dirA"], parent[n]["units"]), "v113_extraction_even": parent[n]["extraction_even"],
                    "rows": rows_ok, "cdas_final_loss": (hist[-1] if hist else None), "n_rows": {k: len(P[k].rows) for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one behaviour must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "rows": {k: False for k in ("row2", "row3", "row4", "row5")}, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "singles_top16", "arms", "n_rows")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_v196", "candidate_id": "corpus.unit_tier3_batch_amended_v196", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
