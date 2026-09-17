#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_tomorrow_rows_shift_toward_had_on_both_corpora pred_c_earlier_rows_shift_toward_will_on_both_corpora pred_d_head_11_3_alone_shifts_the_same_way pred_e_unrelated_readers_within_gate
"""Temporal will/had DoD battery, step 7 (v34): natural rows, FineWeb (in-distribution) and Pile (OOD), frozen bars.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`; rows: `temporal_dod_natural_rows_v34.py` (outcome-blind:
a will/had target with tomorrow/earlier within the previous 10 tokens, stream order, 16 per cell where the corpus
has them; tomorrow/had is rare: 4 FineWeb, 3 Pile rows). Mechanism prediction as in the aspectual v20/v26: removing
S = {11.3, 9.1, 15.5, 9.4} along `O_h^T(u_will - u_had)` lowers will-had on tomorrow-cued rows and raises it on
earlier-cued rows, whatever the true continuation. Readers: was-were, who-which, night-day; the gate uses the mean
ABSOLUTE will-had shift (lesson from v20's mis-designed pooled-signed gate).

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      <= 1e-4 on both panels
    pred_b_tomorrow_rows_shift_toward_had_on_both_corpora   per corpus: mean d(will-had) <= -0.15 and <= 0 on >= 60%
    pred_c_earlier_rows_shift_toward_will_on_both_corpora   per corpus: mean >= +0.15 and >= 0 on >= 60%
    pred_d_head_11_3_alone_shifts_the_same_way   11.3-only removal satisfies both sign conditions on both corpora with
                                          |mean| >= 0.05
    pred_e_unrelated_readers_within_gate  per corpus each unrelated reader mean|move| <= 0.25 x mean|d(will-had)|

PRICE (registered maximum): FineWeb 52 rows (2 batches) + Pile 41 rows (2 batches): per panel native 2 + producer 2 +
S 2 + 11.3 2 = 8 -> 16 forwards; 0 backwards; 0 fits. Bar <= 20.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_temporal_dod_removal_v28 as v28

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "circuits/followups/temporal_auxiliary_dod_natural_rows_v34.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_dod_natural_v34_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.dod_natural_v34"
SHIFT_MIN, DIRECTION_MIN, WEAK_MIN, GATE_RATIO, INSTRUMENT_TOL = 0.15, 0.60, 0.05, 0.25, 1e-4
FORWARDS_MAX = 20
HEAD113 = v28.SINGLES[0]


def to_rows(items, tag):
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for r in items:
        ids = tuple(r["ids"]); present = r["label"] == "will"
        rid = hashlib.sha256(json.dumps([tag, r["doc_index"], r["position"]]).encode()).hexdigest()[:24]
        rows.append(L.Row(rid, f"{tag}_{r['cue']}", r["doc_index"], present, r["text"], ids, " will" if present else " had", " had" if present else " will",
                          v28.WILL if present else v28.HAD, v28.HAD if present else v28.WILL, len(ids) - 1, (), reader_ids))
    return rows


def will_minus_had(row, entry):
    return (entry["answer"] - entry["foil"]) if row.present else (entry["foil"] - entry["answer"])


def main() -> None:
    doc = json.loads(ROWS.read_text())["panels"]
    panels = {}
    for name, panel in doc.items():
        if hashlib.sha256(json.dumps(panel["rows"], sort_keys=True).encode()).hexdigest() != panel["rows_sha256"]:
            raise SystemExit("rows changed")
        panels[name] = to_rows(panel["rows"], name)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": {k: len(v) for k, v in panels.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, v28.SET + (HEAD113,), v28.WILL, v28.HAD)
    forwards, instrument, report = 0, 0.0, {}
    for name, rows in panels.items():
        native, n = v1._run_arm(fw, rows); forwards += n
        ref, n = v1._producer_native(backend, rows); forwards += n
        instrument = max(instrument, max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref)))
        set_arm, n = v1._run_arm(fw, rows, components=v28.SET, mode="project"); forwards += n
        alone, n = v1._run_arm(fw, rows, components=(HEAD113,), mode="project"); forwards += n
        out = {}
        for label, arm in (("set", set_arm), ("head_11_3", alone)):
            for cue in ("tomorrow", "earlier"):
                idx = [i for i, r in enumerate(rows) if r.construction.endswith(cue)]
                d = [will_minus_had(rows[i], arm[i]) - will_minus_had(rows[i], native[i]) for i in idx]
                out[f"{label}/{cue}"] = {"n": len(d), "mean": statistics.mean(d), "fraction_negative": sum(1 for x in d if x <= 0) / len(d), "fraction_nonneg": sum(1 for x in d if x >= 0) / len(d),
                                        "native_accuracy": sum(1 for i in idx if native[i]["answer"] > native[i]["foil"]) / len(idx)}
        abs_shift = statistics.mean(abs(will_minus_had(rows[i], set_arm[i]) - will_minus_had(rows[i], native[i])) for i in range(len(rows)))
        unrelated = {rn: statistics.mean(abs(set_arm[i][rn] - native[i][rn]) for i in range(len(rows))) for rn in L.UNRELATED}
        out["mean_abs_shift"] = abs_shift; out["unrelated"] = unrelated
        report[name] = out
        print(name, json.dumps({k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in out.items()}))
    def ok(name, key, sign, bar):
        s = report[name][key]
        return (s["mean"] <= -bar and s["fraction_negative"] >= DIRECTION_MIN) if sign < 0 else (s["mean"] >= bar and s["fraction_nonneg"] >= DIRECTION_MIN)
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_tomorrow_rows_shift_toward_had_on_both_corpora": all(ok(nm, "set/tomorrow", -1, SHIFT_MIN) for nm in panels),
        "pred_c_earlier_rows_shift_toward_will_on_both_corpora": all(ok(nm, "set/earlier", +1, SHIFT_MIN) for nm in panels),
        "pred_d_head_11_3_alone_shifts_the_same_way": all(ok(nm, "head_11_3/tomorrow", -1, WEAK_MIN) and ok(nm, "head_11_3/earlier", +1, WEAK_MIN) for nm in panels),
        "pred_e_unrelated_readers_within_gate": all(v <= GATE_RATIO * report[nm]["mean_abs_shift"] for nm in panels for v in report[nm]["unrelated"].values()),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "temporal_auxiliary_dod_natural_result_v34", "candidate_id": CANDIDATE_ID, "rows_sha256": {k: v["rows_sha256"] for k, v in doc.items()}, "instrument_max_abs_error": instrument,
              "panels": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
