#!/usr/bin/env python3
"""Freeze exact native-capability row IDs for fresh temporal A1/A2."""

# BQGATE: EXPERIMENT pred_a_counts_recur pred_b_joint_subset_exists pred_c_price_exact
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_capability_manifest_v1.json"
INVALID_PARENT = ROOT / "circuits/followups/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_capability_manifest_v1"
EXPECTED = {"prior": "8502b434391c05dc46c231d891907ac187f18dbfb4bb210a219105c825c6aa29",
            "invalid_parent": "6ae9a2cd1299f0adb91811cad42076655adb3da35ecee0ba7a7a54cdfb0442cc",
            "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9"}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "invalid_parent": INVALID_PARENT, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(INVALID_PARENT.read_text())
    rows = candidate.build_rows()
    selected = {family: [row for row in rows if row["transform_id"] == family]
                for family in ("A1", "A2")}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "invalid"
            or {key: len(value) for key, value in selected.items()} != {"A1": 32, "A2": 32}):
        raise ExperimentError("population or parent terminal changed")
    return selected


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "model_forwards": 4,
              "example_evaluations": 128, "records": 128,
              "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records = []
    for family in ("A1", "A2"):
        for side in ("base", "donor"):
            batch = das._batch(backend, rows[family], side=side)
            output = backend.native(batch, capture=False)
            for row, (answer, foil) in zip(rows[family], output.answer_foil):
                margin = float(answer) - float(foil)
                records.append({"row_id": row["row_id"], "family": family,
                                "direction": row["direction_id"], "side": side,
                                "answer_minus_foil": margin, "correct": margin > 0})
    counts = {family: {side: sum(record["correct"] for record in records
                                 if record["family"] == family and record["side"] == side)
                       for side in ("base", "donor")} for family in ("A1", "A2")}
    jointly_capable = {family: sorted(row["row_id"] for row in rows[family]
        if all(next(record["correct"] for record in records
                    if record["row_id"] == row["row_id"] and record["side"] == side)
               for side in ("base", "donor"))) for family in ("A1", "A2")}
    pred_a = counts == {"A1": {"base": 30, "donor": 31},
                        "A2": {"base": 31, "donor": 31}}
    pred_b = all(len(values) >= 28 for values in jointly_capable.values())
    pred_c = bool(len(records) == 128
                  and len({(record["row_id"], record["side"]) for record in records}) == 128
                  and all(math.isfinite(record["answer_minus_foil"]) for record in records))
    predictions = {"pred_a_counts_recur": pred_a, "pred_b_joint_subset_exists": pred_b,
                   "pred_c_price_exact": pred_c}
    result = {"schema": "temporal_auxiliary_fresh_capability_manifest_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "counts": counts, "jointly_capable_row_ids": jointly_capable,
              "predictions": predictions, "price": {"model_forwards": 4,
              "example_evaluations": 128, "records": len(records),
              "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": "manifest" if all(predictions.values()) else "invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "counts",
          "jointly_capable_row_ids", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
