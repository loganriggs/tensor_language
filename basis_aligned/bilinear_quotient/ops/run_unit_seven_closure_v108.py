#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, band, clamp groups and bars fixed before the run; no fitting.
"""v108: is 'head set + mlp7-10 band' a CLOSED circuit -- sufficient by its direct writes alone?

v106: patching the set and the band from the donor recovers 0.92-1.03 of the donor margin with every other unit LIVE
(free to re-read the patched values). v105: with the set alone patched and every non-set unit clamped to base, only
0.09-1.07 survives (the direct share). Closure test: set + band from the donor AND every other unit (all other MLPs,
all non-set heads) clamped to its base value -- only the direct writes of the set heads and the band reach the final
norm. recovery = g.recovery (signed fraction of the donor margin, mean over ODD A1 rows). Conditions per set:
    live      set + band from donor, rest live                     (v106 replicate)
    closed    set + band from donor, rest clamped to base
    set_only  set from donor, rest clamped to base                 (v105 'all' replicate, as recovery)
    late_open set + band from donor, mlp11-17 live, rest clamped   (does re-reading by late MLPs restore the rest?)
    early_open set + band from donor, mlp0-6 live, rest clamped

REGISTERED BEFORE THE RUN
    pred_a_closed_sufficient   closed >= 0.70 on >= 5 of 7. Worked: 0.8 True; 0.5 False.
    pred_b_band_adds_direct    closed - set_only >= 0.10 on >= 5 of 7 (the band's own direct writes matter).
    pred_c_small_remainder     live - closed <= 0.30 on >= 5 of 7.
    pred_d_late_reads_band     late_open - closed >= 0.5 * (live - closed) on >= 5 of 7 (mlp11-17 carry most of the remainder).
    pred_e_instrument          |live - v106 set_band| <= 0.02 on 7/7 and |set_only - v105 all * v106 set| <= 0.05 on 7/7.
    Prior: a 55%; b 70%; c 60%; d 55%; e 80%.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_common_axis_v15 as v15
import run_unit_tier2_characterization_v23 as v23

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_seven_closure_v108_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
V105 = ROOT / "circuits/followups/unit_seven_band_readers_v105_result.json"
V106 = ROOT / "circuits/followups/unit_seven_band_addin_v106_result.json"
BAND = [f"mlp:{L:02d}" for L in range(7, 11)]
CLOSED_MIN, ADD_MIN, REM_MAX, LATE_FRAC, TOL_LIVE, TOL_SET, K = 0.70, 0.10, 0.30, 0.5, 0.02, 0.05, 5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_seven_closure_v108",
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    modules = {**{k: v[0] for k, v in v23.SETS.items()}, **{k: v15.SETS[k][0] for k in ("verb_complementizer", "verb_preposition")},
               "modal_remoteness": m_modal}
    sets = {n: s["units"] for n, s in json.loads(V80.read_text())["sets"].items()}
    sets["modal_remoteness"] = json.loads(V97.read_text())["final"]
    v105 = json.loads(V105.read_text())["summary"]
    v106 = json.loads(V106.read_text())["summary"]
    heads_all = [f"attn:{L:02d}:head:{h:02d}" for L in range(g.N_LAYERS) for h in range(g.N_HEADS)]
    mlps = lambda lo, hi: [f"mlp:{L:02d}" for L in range(lo, hi + 1)]

    def rec(O, donor_units, clamp):
        merged = dict(O.base_cache)
        for rid in O.base_batch.row_ids:
            for u in donor_units:
                merged[(rid, u)] = O.donor_cache[(rid, u)]
        out = g.forward_units(backend, O.base_batch, units=list(donor_units) + list(clamp), donor_cache=merged, base_cache=O.base_cache)
        return round(g.recovery(O, [-(float(a) - float(f)) for a, f in out.tolist()]), 3)

    report = {}
    for n, units in sets.items():
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        others = [u for u in heads_all if u not in units]
        rest = mlps(0, 6) + mlps(11, 17) + others
        r = {"live": rec(O, units + BAND, []),
             "closed": rec(O, units + BAND, rest),
             "set_only": rec(O, units, mlps(0, 17) + others),
             "late_open": rec(O, units + BAND, mlps(0, 6) + others),
             "early_open": rec(O, units + BAND, mlps(11, 17) + others)}
        report[n] = {"units": units, **r, "v106_set_band": v106[n]["set_band"], "v105_all_times_set": round(v105[n]["all"] * v106[n]["set"], 3)}
        print(n, r, "v106", v106[n]["set_band"], "v105·set", report[n]["v105_all_times_set"], flush=True)

    R = report
    predictions = {
        'pred_a_closed_sufficient': sum(R[n]["closed"] >= CLOSED_MIN for n in R) >= K,
        'pred_b_band_adds_direct': sum(R[n]["closed"] - R[n]["set_only"] >= ADD_MIN for n in R) >= K,
        'pred_c_small_remainder': sum(R[n]["live"] - R[n]["closed"] <= REM_MAX for n in R) >= K,
        'pred_d_late_reads_band': sum(R[n]["late_open"] - R[n]["closed"] >= LATE_FRAC * (R[n]["live"] - R[n]["closed"]) for n in R) >= K,
        'pred_e_instrument': all(abs(R[n]["live"] - R[n]["v106_set_band"]) <= TOL_LIVE and abs(R[n]["set_only"] - R[n]["v105_all_times_set"]) <= TOL_SET for n in R),
    }
    summary = {n: {k: R[n][k] for k in ("live", "closed", "set_only", "late_open", "early_open")} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_closure_result_v1", "candidate_id": "corpus.unit_seven_closure_v108",
              "band": BAND, "summary": summary, "sets": report,
              "bars": {"closed_min": CLOSED_MIN, "add_min": ADD_MIN, "rem_max": REM_MAX, "late_frac": LATE_FRAC, "tol_live": TOL_LIVE, "tol_set": TOL_SET, "k": K},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
