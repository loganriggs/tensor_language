#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_since_rows_shift_toward_had pred_c_temporal_by_rows_shift_toward_has pred_d_head_8_1_alone_shifts_since_rows pred_e_pattern_tracks_shift_on_since_rows
"""Aspectual has/had definition-of-done battery, step 26: OOD corpus (Pile) with the frozen v20/v21 bars.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`; rows: `aspectual_dod_pile_rows_v26.py` (outcome-
blind, monology/pile-uncopyrighted stream order, 8,183 docs scanned; receipt
`circuits/followups/aspectual_anchor_dod_pile_rows_v26.json`). The program labels FineWeb as the training
corpus and the Pile as OOD; v20-v23 were FineWeb (in-distribution natural text). Same mechanism prediction,
same bars, no retuning: on Pile rows with a `since` cue, removing the component lowers has-had; on temporal-
`by` rows it raises it; head 8.1 alone carries the since shift; the shift scales with 8.1's pattern.

PREDICTIONS (scored as written; failures preserved)
    pred_a_instrument_replays_native      <= 1e-4 on both panels
    pred_b_since_rows_shift_toward_had    triple removal: mean d(has-had) <= -0.15 on the 32 any-sense since
                                          rows and <= 0 on >= 60% of them (v20 bar)
    pred_c_temporal_by_rows_shift_toward_has   triple removal on the 32 temporal-by rows: mean >= +0.15 and
                                          >= 60% non-negative (v21 bar)
    pred_d_head_8_1_alone_shifts_since_rows    8.1-only removal: since mean <= -0.05 and >= 60% negative
    pred_e_pattern_tracks_shift_on_since_rows  Pearson r(8.1 pattern on cue, 8.1-only shift) <= -0.40 (v21 bar)

PRICE (registered maximum): any panel 2 batches x (native + producer + triple + 8.1 + pattern fold) = 10;
temporal-by panel 1 batch x (native + producer + triple) = 3 -> 13 forwards; 0 backwards; 0 fits. Bar <= 16.
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
import run_aspectual_dod_natural_pattern_v21 as v21

ROOT = Path(__file__).resolve().parent.parent
ROWS = ROOT / "circuits/followups/aspectual_anchor_dod_pile_rows_v26.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_pile_ood_v26_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_pile_ood_v26"
TOKENS = {"has": 468, "had": 550}
SHIFT_MIN, DIRECTION_MIN, WEAK_MIN, R_MIN, INSTRUMENT_TOL = 0.15, 0.60, 0.05, 0.40, 1e-4
FORWARDS_MAX = 16
HEAD81 = L.Component("attn8_h1_final", 8, "attn", (1,), "final")
TRIPLE = (HEAD81, L.Component("attn9_h1_h4_final", 9, "attn", (1, 4), "final"))


def to_rows(items, tag):
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows, offsets = [], {}
    for r in items:
        ids = tuple(r["ids"]); present = r["label"] == "has"
        rid = hashlib.sha256(json.dumps([tag, r["doc_index"], r["position"]]).encode()).hexdigest()[:24]
        rows.append(L.Row(rid, f"{tag}_{r['cue']}", r["doc_index"], present, r["text"], ids, " has" if present else " had", " had" if present else " has",
                          TOKENS["has"] if present else TOKENS["had"], TOKENS["had"] if present else TOKENS["has"], len(ids) - 1, (), reader_ids))
        offsets[rid] = r["cue_offset"]
    return rows, offsets


def main() -> None:
    doc = json.loads(ROWS.read_text())
    for key in ("any_rows", "temporal_by_rows"):
        if hashlib.sha256(json.dumps(doc[key], sort_keys=True).encode()).hexdigest() != doc[f"{key}_sha256"]:
            raise SystemExit("pile rows changed")
    any_rows, any_off = to_rows(doc["any_rows"], "pile")
    tb_rows, _ = to_rows(doc["temporal_by_rows"], "pile_temporal")
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "any_rows": len(any_rows), "temporal_by_rows": len(tb_rows), "forwards_max": FORWARDS_MAX,
                          "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False,
                          "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, TRIPLE, TOKENS["has"], TOKENS["had"])
    unit = {1: fw.directions[(HEAD81.name, 1)]}
    forwards = 0
    native, n = v1._run_arm(fw, any_rows); forwards += n
    ref, n = v1._producer_native(backend, any_rows); forwards += n
    triple, n = v1._run_arm(fw, any_rows, components=TRIPLE, mode="project"); forwards += n
    alone, n = v1._run_arm(fw, any_rows, components=(HEAD81,), mode="project"); forwards += n
    pattern = {}
    for start in range(0, len(any_rows), v1.BATCH):
        chunk = any_rows[start:start + v1.BATCH]
        out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r: r.final, unit); forwards += 1
        for row, entry in zip(chunk, out):
            pattern[row.row_id] = entry[1]["pattern"][any_off[row.row_id]]
    instrument = max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native, ref))
    native_tb, n = v1._run_arm(fw, tb_rows); forwards += n
    ref_tb, n = v1._producer_native(backend, tb_rows); forwards += n
    triple_tb, n = v1._run_arm(fw, tb_rows, components=TRIPLE, mode="project"); forwards += n
    instrument = max(instrument, max(max(abs(a["answer"] - r[0]), abs(a["foil"] - r[1])) for a, r in zip(native_tb, ref_tb)))

    def shift(arm, base, rows, cue):
        idx = [i for i, r in enumerate(rows) if r.construction.endswith(cue)]
        d = [arm[i]["target_has_had"] - base[i]["target_has_had"] for i in idx]
        return {"n": len(d), "mean": statistics.mean(d), "fraction_negative": sum(1 for x in d if x <= 0) / len(d),
                "fraction_nonneg": sum(1 for x in d if x >= 0) / len(d), "native_accuracy": sum(1 for i in idx if base[i]["answer"] > base[i]["foil"]) / len(idx)}, idx, d
    ts, idx_s, _ = shift(triple, native, any_rows, "since")
    tby, _, _ = shift(triple, native, any_rows, "by")
    hs, _, d_hs = shift(alone, native, any_rows, "since")
    ttb = shift(triple_tb, native_tb, tb_rows, "by")[0]
    r_since = v21.pearson([pattern[any_rows[i].row_id] for i in idx_s], d_hs)
    report = {"triple_since": ts, "triple_any_by": tby, "head_8_1_since": hs, "triple_temporal_by": ttb, "pearson_pattern_shift_since": r_since}
    for k, v in report.items():
        print(k, v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()})
    predictions = {
        "pred_a_instrument_replays_native": instrument <= INSTRUMENT_TOL,
        "pred_b_since_rows_shift_toward_had": ts["mean"] <= -SHIFT_MIN and ts["fraction_negative"] >= DIRECTION_MIN,
        "pred_c_temporal_by_rows_shift_toward_has": ttb["mean"] >= SHIFT_MIN and ttb["fraction_nonneg"] >= DIRECTION_MIN,
        "pred_d_head_8_1_alone_shifts_since_rows": hs["mean"] <= -WEAK_MIN and hs["fraction_negative"] >= DIRECTION_MIN,
        "pred_e_pattern_tracks_shift_on_since_rows": r_since <= -R_MIN,
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_pile_ood_result_v26", "candidate_id": CANDIDATE_ID, "rows_sha256": {"any": doc["any_rows_sha256"], "temporal_by": doc["temporal_by_rows_sha256"]},
              "instrument_max_abs_error": instrument, "report": report, "predictions": predictions, "forwards": forwards,
              "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
