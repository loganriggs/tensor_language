#!/usr/bin/env python3
"""Capability-only gate for the fifth fresh temporal authority."""

# BQGATE: EXPERIMENT pred_a_authority pred_b_joint_capability pred_c_exact_finite_coverage_and_price
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v5 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_v5_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v5.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v5_capability_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_v5_capability_v1"
EXPECTED = {"prior": "2f3e1f222745b1961db2221281db6409e11f1d80c0429c189cf7f6d56105738b",
            "builder": "68b5bc13849e9be97cb596b5cf52ab2ae7a6a6ee3af0937acf472264c63fd647"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    if {"prior": sha(PRIOR), "builder": sha(BUILDER)} != EXPECTED:
        raise RuntimeError("authority hash changed")
    rows = candidate.build_rows()
    family = {name: [row for row in rows if row["transform_id"] == name]
              for name in ("A1", "A2")}
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "model_forwards": 4, "example_evaluations": 128, "records": 128}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records = []
    for panel in ("A1", "A2"):
        for side in ("base", "donor"):
            output = backend.native(das._batch(backend, family[panel], side=side), capture=False)
            for row, values in zip(family[panel], output.answer_foil):
                margin = float(values[0]) - float(values[1])
                records.append({"row_id": row["row_id"], "panel": panel, "side": side,
                                "margin": margin, "correct": margin > 0})
    counts = {panel: {"base_correct": sum(r["correct"] for r in records
                                           if r["panel"] == panel and r["side"] == "base"),
                      "donor_correct": sum(r["correct"] for r in records
                                            if r["panel"] == panel and r["side"] == "donor")}
              for panel in ("A1", "A2")}
    joint_ids = {panel: [row["row_id"] for row in family[panel]
        if all(next(r["correct"] for r in records
                    if r["row_id"] == row["row_id"] and r["side"] == side)
               for side in ("base", "donor"))] for panel in ("A1", "A2")}
    pred_a = len(rows) == 128 and all(len(family[name]) == 32 for name in ("A1", "A2"))
    pred_b = all(counts[p]["base_correct"] >= 28 and counts[p]["donor_correct"] >= 28
                 and len(joint_ids[p]) >= 28 for p in ("A1", "A2"))
    pred_c = (len(records) == 128
              and len({(r["row_id"], r["side"]) for r in records}) == 128
              and all(math.isfinite(r["margin"]) for r in records))
    predictions = {"pred_a_authority": pred_a, "pred_b_joint_capability": pred_b,
                   "pred_c_exact_finite_coverage_and_price": pred_c}
    result = {"schema": "temporal_auxiliary_fresh_v5_capability_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "counts": counts, "jointly_capable_row_ids": joint_ids,
              "predictions": predictions, "records": records,
              "price": {"model_forwards": 4, "example_evaluations": 128,
                        "records": len(records)},
              "terminal": "manifest" if all(predictions.values()) else "invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "counts",
          "jointly_capable_row_ids", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
