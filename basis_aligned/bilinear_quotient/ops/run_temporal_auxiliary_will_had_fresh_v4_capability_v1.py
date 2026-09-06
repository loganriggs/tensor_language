#!/usr/bin/env python3
"""Capability-only gate for the fourth fresh temporal authority."""

# BQGATE: EXPERIMENT pred_a_authority_and_price pred_b_joint_capability
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v4 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_v4_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v4.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v4_capability_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_v4_capability_v1"
EXPECTED = {"prior": "a3a9fea8db93cbe9aca4a3563c06ab07d8a9fdc2b867f0f2e9409cf86fe0175a",
            "builder": "31e40a5e8a8b285ce7afdb6327276c0aa28b4759083586d0310b0857c8b86764"}


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
    joint = {panel: sum(all(next(r["correct"] for r in records
                                 if r["row_id"] == row["row_id"] and r["side"] == side)
                             for side in ("base", "donor")) for row in family[panel])
             for panel in ("A1", "A2")}
    pred_a = (len(records) == 128
              and len({(r["row_id"], r["side"]) for r in records}) == 128
              and all(math.isfinite(r["margin"]) for r in records))
    pred_b = all(counts[p]["base_correct"] >= 28 and counts[p]["donor_correct"] >= 28
                 and joint[p] >= 28 for p in ("A1", "A2"))
    predictions = {"pred_a_authority_and_price": pred_a, "pred_b_joint_capability": pred_b}
    result = {"schema": "temporal_auxiliary_fresh_v4_capability_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "counts": counts, "joint_correct": joint,
              "predictions": predictions, "records": records,
              "price": {"model_forwards": 4, "example_evaluations": 128,
                        "records": len(records)},
              "terminal": "manifest" if all(predictions.values()) else "invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "counts", "joint_correct",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
