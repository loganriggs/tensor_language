#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_all_cells_capable pred_c_set_fraction_within_015_of_frozen_constant pred_d_set_positive_and_selective_everywhere
"""Temporal will/had DoD battery, step 6 (v33): FROZEN NUMERIC PREDICTION on a fourth lexicon and a new construction.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v28 (S fraction 0.81 on fresh rows), v29
(templates 0.82 / 0.78 / 0.36).

Frozen before access: F* = 0.80, band +-0.15 -> [0.65, 0.95] per construction; the comma-adverb construction
is included with the SAME band (prior: unsure; v29's comma-final construction fell to 0.36, but there the
comma ended the input). Rows: 16 new agents x 16 new places (disjoint from every prior panel) on the two
line constructions plus "Tomorrow, the A near the P" / "Earlier, the A near the P".

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native   <= 1e-4 on 96 rows
    pred_b_all_cells_capable           all six (construction x cue) cells >= 0.85
    pred_c_set_fraction_within_015_of_frozen_constant   |fraction - 0.80| <= 0.15 in every construction
    pred_d_set_positive_and_selective_everywhere   positive >= 0.75 rows and each unrelated reader mean|move|
                                       <= 0.25 x damage (raw gate; the norm-matched null was established in v28/v29)

PRICE (registered maximum): 96 rows in 3 batches; native 3 + producer 3 + S 3 = 9 forwards; 0 backwards; 0 fits.
Bar <= 12.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_frozen_prediction_v33_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_frozen_prediction_v33"
FROZEN, BAND, CAPABILITY_MIN, LIVE_POSITIVE, GATE_RATIO, INSTRUMENT_TOL = 0.80, 0.15, 0.85, 0.75, 0.25, 1e-4
FORWARDS_MAX = 12
AGENTS = ("vendor", "maid", "peasant", "pirate", "robber", "widow", "orphan", "prophet", "emperor", "princess", "count", "general", "colonel", "sergeant", "deputy", "thief")
PLACES = ("canyon", "harbor", "island", "forest", "garden", "tower", "bridge", "cabin", "river", "ocean", "road", "valley", "field", "stable", "chapel", "tavern")
CONSTRUCTIONS = {
    "bare_frame": v28.CONSTRUCTIONS["bare_frame"],
    "report_frame": v28.CONSTRUCTIONS["report_frame"],
    "comma_adverb": (lambda p, a: f"Tomorrow, the {a} near the {p}", lambda p, a: f"Earlier, the {a} near the {p}"),
}


def build():
    prior = tuple(v28.AGENTS) + tuple(v28.PLACES)
    rows = L.build_rows_lexicon(AGENTS, PLACES, CONSTRUCTIONS, "temporal_lexicon4", prior=prior)
    return [L.Row(r.row_id, r.construction, r.group, r.present, r.text, r.ids, " will" if r.present else " had", " had" if r.present else " will",
                  v28.WILL if r.present else v28.HAD, v28.HAD if r.present else v28.WILL, r.final, (), r.reader_ids) for r in rows]


def main() -> None:
    rows = build()
    for r in rows:
        if L.ENCODING.encode(r.text + r.answer) != list(r.ids) + [r.answer_id] or L.ENCODING.encode(r.text + r.foil) != list(r.ids) + [r.foil_id]:
            raise SystemExit(f"joint tokenization failed for {r.text!r}")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "frozen": {"fraction": FROZEN, "band": BAND}, "forwards_max": FORWARDS_MAX,
                          "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, v28.SET, v28.WILL, v28.HAD)
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    ref, n = v1._producer_native(backend, rows); forwards += n
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    arm, n = v1._run_arm(fw, rows, components=v28.SET, mode="project"); forwards += n
    capability, report = {}, {}
    for c in CONSTRUCTIONS:
        idx = [i for i, r in enumerate(rows) if r.construction == c]
        for present in (True, False):
            cell = [1.0 if native[i]["answer"] > native[i]["foil"] else 0.0 for i in idx if rows[i].present == present]
            capability[f"{c}/{'tomorrow' if present else 'earlier'}"] = sum(cell) / len(cell)
        s = L.summarize([rows[i] for i in idx], [native[i] for i in idx], [arm[i] for i in idx])
        gates = {name: s[f"{name}_abs_move_mean"] <= GATE_RATIO * s["target_damage_mean"] for name in L.UNRELATED}
        report[c] = {"set": s, "within_band": abs(s["target_damage_fraction"] - FROZEN) <= BAND, "positive_ok": s["target_damage_positive_fraction"] >= LIVE_POSITIVE, "gates": gates}
        print(c, "fraction", round(s["target_damage_fraction"], 3), "pos", round(s["target_damage_positive_fraction"], 3), "gates", gates)
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_all_cells_capable": all(v >= CAPABILITY_MIN for v in capability.values()),
        "pred_c_set_fraction_within_015_of_frozen_constant": all(r["within_band"] for r in report.values()),
        "pred_d_set_positive_and_selective_everywhere": all(r["positive_ok"] and all(r["gates"].values()) for r in report.values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_frozen_prediction_result_v33", "candidate_id": CANDIDATE_ID, "rows_sha256": L.rows_sha256(rows), "instrument_max_abs_error": instrument,
              "capability": capability, "constructions": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "capability": capability, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
