#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_at_least_one_head_is_one_directional pred_c_head_11_3_is_not_one_directional
"""Number family DoD, step 5 (v59): per-head keep-only — which of {11.3, 5.7, 7.8, 9.7} is not a one-direction readout?

For each head alone: zero its slice (its full was/were contribution) vs keep only its were−was readout projection
(in-forward keep_only); retention per head. A head is "one-directional" if retention >= 0.70.

PREDICTIONS: pred_a instrument <= 1e-4; pred_b at least one head is one-directional; pred_c 11.3 is NOT (prior: it serves
number and tense with entangled directions, v56).
PRICE (registered maximum): native 2 + producer 2 + 4 x (zero 2 + keep 2) = 20 forwards; bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
from pathlib import Path
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_number_dod_battery_v55 as v55

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/number_family_dod_per_head_keep_v59_result.json"
CANDIDATE_ID = "lexical_number.pp_intervener.dod_per_head_keep_v59"
ONE_DIR, INSTRUMENT_TOL = 0.70, 1e-4
FORWARDS_MAX = 24


def main() -> None:
    rows, _ = v55.build()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, v55.SINGLES, v55.WERE, v55.WAS)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    report = {}
    for comp in v55.SINGLES:
        zero, n = v1._run_arm(fw, rows, components=(comp,), mode="zero"); forwards += n
        keep, n = v1._run_arm(fw, rows, components=(comp,), mode="keep_only"); forwards += n
        zd = L.summarize(rows, native, zero)["target_damage_mean"]; kd = L.summarize(rows, native, keep)["target_damage_mean"]
        report[comp.name] = {"zero_damage": zd, "keep_damage": kd, "retention": 1.0 - kd / zd if zd else None}
        print(comp.name, {k: round(v, 3) for k, v in report[comp.name].items()})
    predictions = {"pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL, "pred_b_at_least_one_head_is_one_directional": any(v["retention"] is not None and v["retention"] >= ONE_DIR for v in report.values()),
                   "pred_c_head_11_3_is_not_one_directional": report["attn11_h3_final"]["retention"] < ONE_DIR}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_family_dod_per_head_keep_result_v59", "candidate_id": CANDIDATE_ID, "instrument_max_abs_error": instrument, "heads": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
