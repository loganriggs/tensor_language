#!/usr/bin/env python3
# BQGATE: five frozen predictions; sets, band and bars fixed before the run; no fitting.
"""v106: heads + the mlp7-10 band -- does adding the band from the donor close each set to the donor margin?

v105: every greedy head set's indirect effect runs through the mlp7-10 band (non-set heads carry <= 5% on 6/7). The sets
are partial (greedy-minimal heads); the missing part of each circuit should then be the band itself, not more heads.
Instrument: exact interchange (donor values) of  set | band (mlp:07..10) | set + band | set + each single band MLP,
on ODD A1 rows; recovery = g.recovery (signed fraction of the donor margin, mean over rows). 4 + 4 = 8 forwards per set.

REGISTERED BEFORE THE RUN
    pred_a_band_closes        recovery(set + band) >= 0.90 on >= 5 of 7 sets. Worked: 0.95 True; 0.7 False.
    pred_b_band_alone_partial recovery(band alone) >= 0.30 on >= 5 of 7. Worked: 0.45 True; 0.1 False.
    pred_c_set_beats_band     recovery(set) > recovery(band alone) on >= 5 of 7.
    pred_d_subadditive        recovery(set + band) < recovery(set) + recovery(band) on >= 5 of 7 (the band re-reads what the set already wrote).
    pred_e_mlp08_largest      among the four single-MLP add-ins, mlp:08 gives the largest gain over the set on >= 4 of 7.
    Prior: a 50%; b 60%; c 70%; d 75%; e 45%.
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
OUT = ROOT / "circuits/followups/unit_seven_band_addin_v106_result.json"
V80 = ROOT / "circuits/followups/unit_six_sets_cross_inert_v80_result.json"
V97 = ROOT / "circuits/followups/unit_modal_greedy_v97_result.json"
BAND = [f"mlp:{L:02d}" for L in range(7, 11)]
CLOSE, PARTIAL, K, K_E = 0.90, 0.30, 5, 4
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 200, 6000


def _plan():
    return {"candidate_id": "corpus.unit_seven_band_addin_v106",
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

    report = {}
    for n, units in sets.items():
        O = g.prepare(backend, g.rows_of(modules[n], "A1")[1::2])
        rec = lambda us: round(g.recovery(O, g.patched_axis(backend, O, list(us))), 3)
        r = {"set": rec(units), "band": rec(BAND), "set_band": rec(units + BAND)}
        r["set_plus"] = {m: rec(units + [m]) for m in BAND}
        r["gain"] = {m: round(r["set_plus"][m] - r["set"], 3) for m in BAND}
        report[n] = {"units": units, **r}
        print(n, {k: r[k] for k in ("set", "band", "set_band")}, r["gain"], flush=True)

    R = report
    predictions = {
        'pred_a_band_closes': sum(R[n]["set_band"] >= CLOSE for n in R) >= K,
        'pred_b_band_alone_partial': sum(R[n]["band"] >= PARTIAL for n in R) >= K,
        'pred_c_set_beats_band': sum(R[n]["set"] > R[n]["band"] for n in R) >= K,
        'pred_d_subadditive': sum(R[n]["set_band"] < R[n]["set"] + R[n]["band"] for n in R) >= K,
        'pred_e_mlp08_largest': sum(max(R[n]["gain"], key=lambda m: R[n]["gain"][m]) == "mlp:08" for n in R) >= K_E,
    }
    summary = {n: {k: R[n][k] for k in ("set", "band", "set_band")} | {"best_addin": max(R[n]["gain"], key=lambda m: R[n]["gain"][m])} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_band_addin_result_v1", "candidate_id": "corpus.unit_seven_band_addin_v106",
              "band": BAND, "summary": summary, "sets": report, "bars": {"close": CLOSE, "partial": PARTIAL, "k": K, "k_e": K_E},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": summary, "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
