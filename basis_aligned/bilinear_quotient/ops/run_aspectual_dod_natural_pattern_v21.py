#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_pattern_predicts_shift_on_since_rows pred_c_pattern_predicts_shift_on_by_rows pred_d_high_pattern_by_rows_shift_toward_has pred_e_temporal_by_panel_shifts_toward_has
"""Aspectual has/had definition-of-done battery, step 21: does head 8.1's PATTERN on the cue carry the sense?

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parents: v20 (natural FineWeb rows: the since-
shift transfers; any-sense `by` does not), v10-v15 (8.1's value is token-only; its pattern is the only
context-dependent factor).

WHY. If the block-0 value is sense-blind, the sense must enter through the pattern p_{8.1}(final, cue):
agentive `by` should get little attention from 8.1 at the has/had position, deadline `by` more. Fold per
natural row: p on the cue; edit per row: d(has-had) under the 8.1-only readout removal. Prediction: across
rows of one cue, the shift scales with the pattern (Pearson r with the predicted sign), and the `by` rows in
the top pattern tercile shift toward has. A second, outcome-blind temporal-`by` panel (v21 rows; filter
still imperfect) is scored with the same frozen bar as v20.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      on both panels, <= 1e-4
    pred_b_pattern_predicts_shift_on_since_rows   Pearson r(p, d) <= -0.40 over the 32 since rows (more
                                          attention -> larger drop of has-had when 8.1 is removed)
    pred_c_pattern_predicts_shift_on_by_rows      Pearson r(p, d) >= +0.40 over the 32 any-sense by rows
    pred_d_high_pattern_by_rows_shift_toward_has  the top pattern tercile of the any-sense by rows (11 rows)
                                          has mean d >= +0.10 under the 8.1-only removal
    pred_e_temporal_by_panel_shifts_toward_has    on the v21 temporal-by panel, triple removal mean d >= +0.15
                                          and >= 60% rows non-negative (same bar as v20 pred_c)

PRICE (registered maximum): v20 panel 2 batches x (native + producer + 8.1 removal + pattern fold) = 8;
v21 panel 1 batch x (native + producer + triple + 8.1) = 4 -> 12 forwards; 0 backwards; 0 fits. Bar <= 16.
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
import run_aspectual_dod_natural_v20 as v20

ROOT = Path(__file__).resolve().parent.parent
ROWS21 = ROOT / "circuits/followups/aspectual_anchor_dod_natural_rows_v21.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_natural_pattern_v21_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_natural_pattern_v21"
TOKENS = {"has": 468, "had": 550}
R_MIN, TERCILE_MIN, SHIFT_MIN, DIRECTION_MIN, INSTRUMENT_TOL = 0.40, 0.10, 0.15, 0.60, 1e-4
FORWARDS_MAX = 16
HEAD81 = L.Component("attn8_h1_final", 8, "attn", (1,), "final")
TRIPLE = (HEAD81, L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))


def load_v21():
    doc = json.loads(ROWS21.read_text())
    if hashlib.sha256(json.dumps(doc["rows"], sort_keys=True).encode()).hexdigest() != doc["rows_sha256"]:
        raise SystemExit("v21 rows changed")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows, offsets = [], {}
    for r in doc["rows"]:
        ids = tuple(r["ids"]); present = r["label"] == "has"
        rid = hashlib.sha256(json.dumps(["v21", r["doc_index"], r["position"]]).encode()).hexdigest()[:24]
        rows.append(L.Row(rid, "natural_temporal_by", r["doc_index"], present, r["text"], ids, " has" if present else " had",
                          " had" if present else " has", TOKENS["has"] if present else TOKENS["had"], TOKENS["had"] if present else TOKENS["has"],
                          len(ids) - 1, (), reader_ids))
        offsets[rid] = r["cue_offset"]
    return rows, offsets, doc["rows_sha256"]


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * sx * sy)


def main() -> None:
    rows20, sha20 = v20.load_rows()
    doc20 = json.loads(v20.ROWS.read_text())
    offsets20 = {hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24]: r["cue_offset"] for r in doc20["rows"]}
    rows21, offsets21, sha21 = load_v21()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows20": len(rows20), "rows21": len(rows21), "sha20": sha20, "sha21": sha21,
                          "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                          "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE, TOKENS["has"], TOKENS["had"])
    unit = {1: fw.directions[(HEAD81.name, 1)]}
    forwards = 0
    # panel v20: native, 8.1 removal, pattern fold
    native, n = v1._run_arm(fw, rows20); forwards += n
    ref, n = v1._producer_native(backend, rows20); forwards += n
    alone, n = v1._run_arm(fw, rows20, components=(HEAD81,), mode="project"); forwards += n
    pattern = {}
    for start in range(0, len(rows20), v1.BATCH):
        chunk = rows20[start:start + v1.BATCH]
        out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r: r.final, unit); forwards += 1
        for row, entry in zip(chunk, out):
            pattern[row.row_id] = entry[1]["pattern"][offsets20[row.row_id]]
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    per_cue = {}
    for cue in ("since", "by"):
        idx = [i for i, r in enumerate(rows20) if r.construction == f"natural_{cue}"]
        p = [pattern[rows20[i].row_id] for i in idx]
        d = [alone[i]["target_has_had"] - native[i]["target_has_had"] for i in idx]
        order = sorted(range(len(idx)), key=lambda k: -p[k])
        top = order[:len(idx) // 3 + (1 if len(idx) % 3 else 0)]
        per_cue[cue] = {"n": len(idx), "pearson": pearson(p, d), "pattern_mean": statistics.mean(p), "pattern_median": statistics.median(p),
                        "shift_mean": statistics.mean(d), "top_tercile_shift_mean": statistics.mean(d[k] for k in top),
                        "bottom_tercile_shift_mean": statistics.mean(d[k] for k in order[-len(top):]),
                        "rows": [{"pattern": p[k], "shift": d[k], "text_tail": rows20[idx[k]].text[-60:]} for k in order]}
        print(cue, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in per_cue[cue].items() if k != "rows"})
    # panel v21
    native21, n = v1._run_arm(fw, rows21); forwards += n
    ref21, n = v1._producer_native(backend, rows21); forwards += n
    triple21, n = v1._run_arm(fw, rows21, components=TRIPLE, mode="project"); forwards += n
    alone21, n = v1._run_arm(fw, rows21, components=(HEAD81,), mode="project"); forwards += n
    instrument = max(instrument, max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native21, ref21)))
    d21 = [triple21[i]["target_has_had"] - native21[i]["target_has_had"] for i in range(len(rows21))]
    d21a = [alone21[i]["target_has_had"] - native21[i]["target_has_had"] for i in range(len(rows21))]
    panel21 = {"n": len(rows21), "triple_shift_mean": statistics.mean(d21), "triple_fraction_nonneg": sum(1 for x in d21 if x >= 0) / len(d21),
               "head_8_1_shift_mean": statistics.mean(d21a), "native_accuracy": sum(1 for i in range(len(rows21)) if native21[i]["answer"] > native21[i]["foil"]) / len(rows21)}
    print("temporal_by panel", {k: round(v, 4) for k, v in panel21.items()})
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_pattern_predicts_shift_on_since_rows": per_cue["since"]["pearson"] <= -R_MIN,
        "pred_c_pattern_predicts_shift_on_by_rows": per_cue["by"]["pearson"] >= R_MIN,
        "pred_d_high_pattern_by_rows_shift_toward_has": per_cue["by"]["top_tercile_shift_mean"] >= TERCILE_MIN,
        "pred_e_temporal_by_panel_shifts_toward_has": panel21["triple_shift_mean"] >= SHIFT_MIN and panel21["triple_fraction_nonneg"] >= DIRECTION_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_natural_pattern_result_v21", "candidate_id": CANDIDATE_ID, "rows_sha256": {"v20": sha20, "v21": sha21},
              "instrument_max_abs_error": instrument, "lambda": lamb, "per_cue": per_cue, "temporal_by_panel": panel21,
              "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
