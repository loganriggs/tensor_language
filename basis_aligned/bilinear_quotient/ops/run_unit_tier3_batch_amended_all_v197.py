#!/usr/bin/env python3
# BQGATE: six frozen predictions; the amended tier-3 battery (v196 protocol, unchanged) over all 24 lifted behaviours in one receipt.
"""v197: the AMENDED tier-3 battery over ALL 24 lifted behaviours -- one table instead of a patchwork.

Protocol is v196's, unchanged (within-direction split FIT = rows[0::4]+rows[1::4] / HELD = rows[2::4]+rows[3::4]; greedy pool 40,
target 0.88, min_gain 0.005, max 30 units; row 2 = MIN over the two held-out directions >= 0.80; row 3 dim A1 CE LB975 > 0 and
>= 0.10 on held A1; row 4 constrained rank-1 own-C UB975 <= 0.01 on held C, control = C FIT rows; row 5 = A2-own diff-in-means
CE / A2 full-rank ceiling in [0.5, 1.4] with LB975 > 0). Behaviours: the 21 v112 BATCH + quantifier_number (v112 instrument) +
possessive_number + animacy (v190) = 24. The five v196 behaviours are re-run inside the same loop as the instrument.
Why one receipt: the tier standing reported today (21/24 four-row) is assembled across v112/v113/v190/v192-v196 under two
protocols; this rung makes the AMENDED standing one number from one code path and one split.
Magnitudes this rests on: v112/v113 rows (14 four-row passes at the old split), v196 (3 of 4 misses lift; degree_result dirB
0.788; quantifier C UB 0.035 under the half-size C fit; coordination_agreement failed row 4 in v112 AND v113 at 0.02-0.05 UB).
No probe was run for the 19 behaviours not in v196; their within-direction numbers are this run's own outcome.

REGISTERED (bars in BARS; each coded predicate is the sentence here):
  pred_a_four_rows      -> at least 19 of 24 behaviours pass all four rows (belief 19-21: degree_result row 2, coordination
                           row 4 and quantifier row 4 are the expected misses; correlative_both_neither unknown).
  pred_b_row2           -> MIN held-out per-direction exact-set recovery >= 0.80 on at least 21 of 24.
  pred_c_row3           -> dim A1 CE damage LB975 > 0 and >= 0.10 on all 24 (row 3 has never failed on a lifted set).
  pred_d_row4           -> constrained rank-1 own-C CE UB975 <= 0.01 on at least 21 of 24.
  pred_e_row5_amended   -> A2-own / A2 ceiling within [0.5, 1.4] with LB975 > 0 on at least 22 of 24.
  pred_f_instrument     -> on the five v196 behaviours n_units is identical and extraction_held_dirB within +-0.02 of v196
                           (same seed, same split, same code path: this is a determinism check, NOT an independent run
                           outcome, disclosed as such).
Accounting: the receipt's four-row set IS the amended tier-3 standing; v112/v113 rows stand as their own rows. An error receipt
per behaviour (try/except) counts as a miss on every row. Cost ~27 s GPU per behaviour (v196), ~11 min total.
Smoke: V197_SMOKE=<out.json> (CPU, V197_SMOKE_ROWS=4 per split, V197_SMOKE_NAMES=quantifier_number; pool 3/max 3, steps 5).
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
OUT = ROOT / "circuits/followups/unit_tier3_batch_amended_all_v197_result.json"
V196 = ROOT / "circuits/followups/unit_tier3_batch_amended_v196_result.json"
NAMES = {"additive_scope": "additive_scope", "coordination_agreement": "coordination_agreement",
         "correlative_both_either": "both_either", "correlative_both_neither": "correlative_pair",
         "correlative_either_neither": "correlative_state", "degree_frame": "degree_frame", "degree_result": "degree_result",
         "finiteness_selection": "finiteness", "interrogative_licensing": "interrogative_licensing",
         "lexical_number_pp": "lexical_number_pp", "narrative_tense": "narrative_tense",
         "numbered_list_choice": "control_choice", "numeric_sequence_choice": "sequence_control_choice",
         "perfect_number": "perfect_number", "polarity_state": "polarity_state",
         "possessive_adjacent": "possessive_adjacent", "possessive_argument": "possessive_argument",
         "possessive_long_simple": "possessive_long_simple", "possessive_medial": "possessive_medial",
         "possessive_verbfinal": "possessive_verbfinal", "preposition_selection": "preposition_selection",
         "quantifier_number": "quantifier_number", "possessive_number": "possessive_number", "animacy": "animacy"}
INSTR = ("correlative_both_either", "degree_result", "finiteness_selection", "polarity_state", "quantifier_number")
POOL, TARGET, MIN_GAIN, MAX_UNITS = 40, 0.88, 0.005, 30
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX = 0.80, 0.10, 0.01
BARS = {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "row5_share_band": [0.5, 1.4], "k_four": 19, "k_row2": 21, "k_row3": 24, "k_row4": 21, "k_row5": 22, "instr_tol": 0.02}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200000, 6400000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_amended_all_v197", "behaviours": 24, "constructions": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 24 * STEPS, "model_updates": 0, "fit_parameters": 24 * MAX_UNITS * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    good = {n: r for n, r in R.items() if "rows" in r and "error" not in r}
    ok = bool(good)  # an empty or all-error receipt fails everything
    cnt = lambda row: sum(1 for n in good if good[n]["rows"][row])
    four = sum(1 for n in good if all(good[n]["rows"].values()))
    a = ok and four >= B["k_four"]
    b = ok and cnt("row2") >= B["k_row2"]
    c = ok and cnt("row3") >= B["k_row3"]
    d = ok and cnt("row4") >= B["k_row4"]
    e = ok and cnt("row5") >= B["k_row5"]
    f = ok and all(n in good and good[n].get("instr_n_units_match") and good[n].get("instr_dirB_abs_diff") is not None
                   and good[n]["instr_dirB_abs_diff"] <= B["instr_tol"] for n in INSTR)
    return {"pred_a_four_rows": bool(a), "pred_b_row2": bool(b), "pred_c_row3": bool(c), "pred_d_row4": bool(d),
            "pred_e_row5_amended": bool(e), "pred_f_instrument": bool(f)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V197_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    cut = (lambda rows: rows[:int(os.environ.get("V197_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    pool, max_units, steps = (3, 3, 5) if smoke else (POOL, MAX_UNITS, STEPS)
    which = [n for n in NAMES if not smoke or n in os.environ.get("V197_SMOKE_NAMES", "quantifier_number").split(",")]
    parent = json.loads(V196.read_text())["behaviours"]
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
                    "instr_n_units_match": (n in parent and parent[n]["units"] == units), "instr_dirB_abs_diff": (round(abs(parent[n]["extraction_held_dirB"] - e_b), 3) if n in parent else None),
                    "rows": rows_ok, "cdas_final_loss": (hist[-1] if hist else None), "n_rows": {k: len(P[k].rows) for k in P}, "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as exc:  # noqa: BLE001 - one behaviour must not lose the batch
            R[n] = {"error": f"{type(exc).__name__}: {exc}", "rows": {k: False for k in ("row2", "row3", "row4", "row5")}, "seconds": round(time.perf_counter() - t1, 1)}
        print(n, json.dumps({k: v for k, v in R[n].items() if k not in ("units", "singles_top16", "arms", "n_rows")}), round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier3_batch_amended_all_v197", "candidate_id": "corpus.unit_tier3_batch_amended_all_v197", "bars": BARS,
              "protocol": {"split": "within-direction: fit rows[0::4]+rows[1::4], held rows[2::4]+rows[3::4]", "pool": pool, "target": TARGET, "min_gain": MIN_GAIN, "max_units": max_units,
                           "cdas": {"steps": steps, "lr": LR, "complement_weight": CW, "control_weight": LAM, "controls": "own C FIT rows"},
                           "row5": "A2-own diff-in-means CE / A2 full-rank mean-ablation ceiling, LB975 > 0"},
              "four_row_passes": sorted(n for n, r in R.items() if "error" not in r and all(r["rows"].values())), "behaviours": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "rows": {n: r["rows"] for n, r in R.items()}}, indent=2))


if __name__ == "__main__":
    main()
