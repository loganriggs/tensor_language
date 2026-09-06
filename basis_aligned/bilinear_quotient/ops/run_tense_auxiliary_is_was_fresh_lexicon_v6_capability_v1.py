#!/usr/bin/env python3
"""Capability-only gate for prospective is/was lexicon v6."""

# BQGATE: EXPERIMENT pred_a_authority_and_disjointness pred_b_population_capability pred_c_no_causal_outcome_access pred_d_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v6_capability_v1.json"
V5_INVALID = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v6_capability_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v6_capability_v1"
EXPECTED_PRIOR_SHA256 = "c2809798718d1adb2f8fc21c4ed71038219a934cd850605ca8fb8b28cde03d81"
EXPECTED = {
    V5_INVALID: "6ec496c8ac93a369fe87e7dcb023e225df131f7acb8aa8d5804c1c84dccff0d1",
    BUILDER: "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
}
EXPECTED_ROWS_SHA256 = "4eee90d9f39f6997c4926a0e7f6baecc4134c06535fe307d0a38f936b75defd5"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row):
    return row["direction_id"] if row["family"] in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior, invalid = json.loads(PRIOR.read_text()), json.loads(V5_INVALID.read_text())
    rows = fresh.build_rows()
    if prior.get("candidate_id") != CANDIDATE_ID or invalid.get("terminal") != "invalid" or invalid.get("causal_outcomes_opened") is not False or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256 or len(rows) != 64:
        raise ExperimentError("candidate, v5 boundary, rows, or coverage changed")
    return rows


def main():
    rows = validate_static()
    plan = {
        "schema": "tense_auxiliary_is_was_fresh_lexicon_v6_capability_dryrun_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_only",
        "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256,
        "rows": 64, "native_sides": 128, "model_forwards_exact": 2,
        "example_evaluations_exact": 128, "interventions": 0, "resid10_features": 0,
        "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=False) for side in ("base", "donor")}
    records = []
    for i, row in enumerate(rows):
        direction = direction_for(row)
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[i]
            records.append({"family": row["family"], "direction": direction, "side": side, "row_id": str(row["row_id"]), "correct": float(answer) > float(foil)})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [record for record in records if record["family"] == family and record["direction"] == direction]
            correct = sum(record["correct"] for record in selected)
            threshold = 0.75 if family == "C" else 0.85
            accuracy = correct / len(selected)
            cells.append({"family": family, "direction": direction, "correct": correct, "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    pred_a = fresh.authority_sha256() == EXPECTED_ROWS_SHA256 and len(rows) == 64
    pred_b = all(cell["passed"] for cell in cells)
    pred_c = True
    pred_d = len(records) == 128 and len({(record["row_id"], record["side"]) for record in records}) == 128
    predictions = {"pred_a_authority_and_disjointness": pred_a, "pred_b_population_capability": pred_b, "pred_c_no_causal_outcome_access": pred_c, "pred_d_exact_coverage_and_price": pred_d}
    terminal = "screen" if all(predictions.values()) else "invalid"
    reason = "v6_population_capability_passes" if terminal == "screen" else "authority_population_capability_or_coverage_invalid"
    result = {
        "schema": "tense_auxiliary_is_was_fresh_lexicon_v6_capability_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256,
        "capability_cells": cells, "native_records": records, "causal_outcomes_opened": False,
        "score": {"model_forwards": 2, "example_evaluations": 128, "native_sides": len(records), "interventions": 0, "resid10_features": 0, "grid_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "predictions": predictions, "terminal": terminal, "reason": reason,
        "next_action": "run separately preregistered frozen controller on v6" if terminal == "screen" else "retain invalid and do not open causal outcomes",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability_cells": cells, "causal_outcomes_opened": False, "price": result["score"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
