#!/usr/bin/env python3
# BQGATE: five frozen predictions; behaviour list (v112 row-2 misses), greedy parameters and bars fixed before the run.
"""v113: is v112's row-2 miss a greedy-parameter miss or a heads-only ceiling?

v112 lifted 21 screens in one rung; rows 3-5 held on 18-21/21 but row 2 (head-set ODD A1 extraction >= 0.80) on only 8/21,
with the misses at 0.60-0.78 and the instrument (quantifier) at 0.79 because the EVEN greedy stopped the moment it reached
its 0.80 target (EVEN 0.83 -> ODD 0.79). Two readings: (i) parameter miss -- the greedy target/size was too small for a
split-generalisation gap of 0.01-0.05; (ii) ceiling -- for these behaviours the heads carry only 0.6-0.8 and the rest is
MLP-written (v105 band survival). Same protocol as v112 on the 13 misses + quantifier, with the greedy allowed to go
further: EVEN target 0.88, pool 20, <= 14 heads, min_gain 0.02. Everything else (arms, rows, cross) identical.

REGISTERED BEFORE THE RUN (13 new-behaviour misses; quantifier is the instrument)
    pred_a_row2_recovered   row 2 (ODD >= 0.80) now holds on >= 7 of 13.            Worked: 8 True; 5 False.
    pred_b_even_reaches     EVEN extraction >= 0.88 on >= 7 of 13 (the greedy CAN get there on the fit split). Worked: 9 True; 4 False.
    pred_c_gap_small        EVEN - ODD extraction <= 0.08 on >= 10 of 13 (a ceiling, not overfitting). Worked: 0.05 True; 0.12 False.
    pred_d_rows_kept        rows 3, 4 and 5 all still hold on >= 10 of 13 with the larger sets.  Worked: 11 True; 8 False.
    pred_e_instrument       quantifier: ODD extraction >= 0.80 and cdas own-C UB <= 0.03.       Worked: 0.86 / 0.008 True; 0.79 False.
    Prior: a 50%; b 70%; c 60%; d 75%; e 80%.
"""
from __future__ import annotations

import importlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_selective_removal_four_sets_v51 as v51

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_batch_row2_v113_result.json"
# name -> candidate module suffix (circuit_fast_screen_candidate_<suffix>)
BATCH = {
    "additive_scope": "additive_scope", "coordination_agreement": "coordination_agreement",
    "correlative_both_either": "both_either", "correlative_both_neither": "correlative_pair",
    "correlative_either_neither": "correlative_state", "degree_result": "degree_result",
    "finiteness_selection": "finiteness", "numbered_list_choice": "control_choice", "perfect_number": "perfect_number",
    "polarity_state": "polarity_state", "possessive_adjacent": "possessive_adjacent",
    "possessive_long_simple": "possessive_long_simple", "possessive_medial": "possessive_medial",
}
INSTRUMENT = {"quantifier_number": "quantifier_number"}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 20, 0.88, 0.02, 14
LAM, STEPS, LR, CW = 30.0, 120, 0.05, 1.0
EXT_MIN, REM_MIN, C_UB_MAX, A2_FRAC, INSTR_C_UB, GAP, K_A, K_B, K_C, K_D = 0.80, 0.10, 0.01, 0.50, 0.03, 0.08, 7, 7, 10, 10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 20000, 640000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_row2_v113", "behaviours": len(BATCH), "instrument": list(INSTRUMENT),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": (len(BATCH) + 1) * STEPS, "model_updates": 0, "fit_parameters": (len(BATCH) + 1) * 10 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def run(names, out, candidate_id):
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    modules = {n: importlib.import_module(f"circuit_fast_screen_candidate_{s}") for n, s in names.items()}
    prep = {}
    for n, m in modules.items():
        a1 = g.rows_of(m, "A1")
        prep[n] = {"even": g.prepare(backend, a1[0::2]), "odd": g.prepare(backend, a1[1::2]),
                   "A2": g.prepare(backend, g.rows_of(m, "A2")[1::2]), "P": g.prepare(backend, g.rows_of(m, "P")[1::2]),
                   "C": g.prepare(backend, g.rows_of(m, "C")[1::2]), "C_even": g.prepare(backend, g.rows_of(m, "C")[0::2])}

    def mu_of(p, units):
        return {u: torch.stack([torch.as_tensor(c[(rid, u)]).float() for c in (p.base_cache, p.donor_cache) for rid in p.base_batch.row_ids]).mean(0) for u in units}

    def dmg(p, units, q, mu):
        s = v51.summary(torch, v51.removal(backend, p, units, q, mu))
        return {k: round(s[k], 4) for k in ("ce_damage", "ce_lb975", "ce_ub975", "margin_damage", "top1_change_rate")}

    R = {}
    for n in names:
        t1 = time.perf_counter()
        try:
            P = prep[n]
            singles, ranked, greedy = g.greedy_heads(backend, P["even"], pool=POOL, target=TARGET, min_gain=MIN_GAIN, max_units=MAX_UNITS)
            units = list(greedy["chosen"])
            ext_even = g.recovery(P["even"], g.patched_axis(backend, P["even"], units))
            ext_odd = g.recovery(P["odd"], g.patched_axis(backend, P["odd"], units))
            mu = mu_of(P["even"], units)
            q = {"dim": g.block_diff_in_means(backend, P["even"], units)}
            q["cdas"], hist = g.fit_block_subspace_constrained(backend, P["even"], units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW,
                                                               controls=(P["C_even"],), control_weight=LAM, mu=mu)
            arms = {}
            for arm, qa in q.items():
                arms[arm] = {fam: dmg(P[k], units, qa, mu) for fam, k in (("A1", "odd"), ("A2", "A2"), ("P", "P"), ("C", "C"))}
                arms[arm]["extraction_odd"] = round(g.recovery(P["odd"], g.patched_axis(backend, P["odd"], units, q=qa)) / ext_odd, 3) if abs(ext_odd) > 1e-6 else None
                arms[arm]["cross"] = {m: dmg(prep[m]["odd"], units, qa, mu)["ce_damage"] for m in names if m != n}
                arms[arm]["cross_abs_max"] = round(max(abs(v) for v in arms[arm]["cross"].values()), 4)
            d, c = arms["dim"], arms["cdas"]
            rows = {"row2": ext_odd >= EXT_MIN,
                    "row3": d["A1"]["ce_lb975"] > 0 and d["A1"]["ce_damage"] >= REM_MIN,
                    "row4": c["C"]["ce_ub975"] <= C_UB_MAX,
                    "row5": d["A2"]["ce_lb975"] > 0 and d["A2"]["ce_damage"] >= A2_FRAC * d["A1"]["ce_damage"]}
            R[n] = {"units": units, "n_units": len(units), "singles_top16": {u: round(singles[u], 3) for u in ranked[:16]},
                    "extraction_even": round(ext_even, 3), "extraction_odd": round(ext_odd, 3), "arms": arms, "rows": rows,
                    "cdas_final_loss": (hist[-1] if hist else None), "seconds": round(time.perf_counter() - t1, 1)}
        except Exception as e:  # noqa: BLE001 - a batch must not lose 20 behaviours to one
            R[n] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:],
                    "rows": {"row2": False, "row3": False, "row4": False, "row5": False}, "seconds": round(time.perf_counter() - t1, 1)}
        r = R[n]
        print(n, r.get("units"), "ext", r.get("extraction_odd"), r["rows"],
              {a: (v["A1"]["ce_damage"], v["C"]["ce_ub975"], v["cross_abs_max"]) for a, v in r.get("arms", {}).items()},
              r.get("error", ""), round(time.perf_counter() - t0), "s", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"partial": True, "behaviours": R}, indent=1, sort_keys=True, default=str) + "\n")
    return R, time.perf_counter() - t0


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    R, secs = run({**BATCH, **INSTRUMENT}, OUT, "corpus.unit_tier3_batch_row2_v113")
    new = [n for n in BATCH]
    count = lambda row: sum(R[n]["rows"][row] for n in new)
    qi = R["quantifier_number"]
    predictions = {
        'pred_a_row2_recovered': count("row2") >= K_A,
        'pred_b_even_reaches': sum(R[n].get("extraction_even", 0) >= TARGET for n in new) >= K_B,
        'pred_c_gap_small': sum(("extraction_even" in R[n]) and (R[n]["extraction_even"] - R[n]["extraction_odd"] <= GAP) for n in new) >= K_C,
        'pred_d_rows_kept': sum(R[n]["rows"]["row3"] and R[n]["rows"]["row4"] and R[n]["rows"]["row5"] for n in new) >= K_D,
        'pred_e_instrument': bool(qi["rows"]["row2"] and "arms" in qi and qi["arms"]["cdas"]["C"]["ce_ub975"] <= INSTR_C_UB),
    }
    tiers = {n: ("tier3_all_rows" if all(R[n]["rows"].values()) else "tier3_rows_" + "".join(k[-1] for k, v in R[n]["rows"].items() if v) if R[n]["rows"]["row2"]
                 else "not_localizable" if "error" not in R[n] else "error") for n in R}
    summary = {n: {"n_units": R[n].get("n_units"), "ext_odd": R[n].get("extraction_odd"), "rows": R[n]["rows"], "tier": tiers[n],
                   "cross_abs_max": {a: v["cross_abs_max"] for a, v in R[n].get("arms", {}).items()}} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_tier3_batch_row2_result_v1", "candidate_id": "corpus.unit_tier3_batch_row2_v113",
              "counts": {row: count(row) for row in ("row2", "row3", "row4", "row5")}, "all_four": sum(all(R[n]["rows"].values()) for n in new),
              "errors": {n: R[n]["error"] for n in R if "error" in R[n]}, "summary": summary, "behaviours": R,
              "protocol": {"pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS, "lambda": LAM, "steps": STEPS, "lr": LR,
                           "complement_weight": CW, "controls": "own C EVEN only"},
              "bars": {"ext_min": EXT_MIN, "rem_min": REM_MIN, "c_ub_max": C_UB_MAX, "a2_frac": A2_FRAC, "instr_c_ub": INSTR_C_UB,
                       "gap": GAP, "k": {"a": K_A, "b": K_B, "c": K_C, "d": K_D}},
              "seconds": round(secs, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "all_four": result["all_four"], "tiers": tiers,
                      "errors": result["errors"], "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
