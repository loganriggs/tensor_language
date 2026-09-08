#!/usr/bin/env python3
"""Capability-only gate for the new-construction, aligned v24 OOD bank."""

# BQGATE: EXPERIMENT
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24.py"
V23_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1_result.json"
V23_CONFIRMATION = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1_result.json"
EXPECTED = {
    "prior": "ffa818d105793a63d8a6f2bedf3e240ae3751e584ad5098d8dfa7e8685e2604a",
    "builder": "240a1685d8e7190ee6803268b1b2e7c910891c29135583f2ae4b14eb3d19f1a0",
    "v23_capability": "35e6cb352c276a4e1b3643cc2f5795bde667c06b8d7987352b198ccce10a4856",
    "v23_confirmation": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
}
ROWS_SHA256 = "870c829290e1791351d0b2b67985aa5700780920b14423906e7b8fd4d35ed2de"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v24_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v24_capability_result_v1"
FAMILIES = ("A1", "A2", "P", "C")
TARGET_FAMILIES = ("A1", "A2")
CONTROL_FAMILIES = ("P", "C")
DIRECTIONS = {
    "A1": ("present_to_past", "past_to_present"),
    "A2": ("present_to_past", "past_to_present"),
    "P": ("primary_to_alternative", "alternative_to_primary"),
    "C": ("studio_to_clinic", "clinic_to_studio"),
}
CELL_BAR = 0.75
JOINT_BAR = 12
PRICE = {"model_forwards": 2, "example_evaluations": 128, "interventions": 0,
         "transformer_backwards": 0, "model_updates": 0}
PREDICTION_KEYS = tuple("pred_" + suffix for suffix in (
    "a_authority_novelty_and_exact_population", "b_native_target_capability",
    "c_native_control_capability", "d_no_causal_outcome_access_and_exact_price"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {"prior": PRIOR, "builder": BUILDER, "v23_capability": V23_CAPABILITY,
             "v23_confirmation": V23_CONFIRMATION}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v24 capability authority changed: {observed}")
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(V23_CAPABILITY.read_text())
    confirmation = json.loads(V23_CONFIRMATION.read_text())
    if (prior.get("candidate_id") != CANDIDATE_ID
            or capability.get("terminal") != "screen"
            or not all(capability.get("predictions", {}).values())
            or capability.get("causal_outcomes_opened") is not False
            or confirmation.get("terminal") != "confirmed_selective_four_head_writer_program"
            or not all(confirmation.get("predictions", {}).values())):
        raise RuntimeError("v23 ancestry or v24 prior changed")
    rows = fresh.build_rows()
    if (fresh.validate_rows(rows) != ROWS_SHA256 or len(rows) != 64
            or any(row["base_semantic_position"] != row["donor_semantic_position"] for row in rows)):
        raise RuntimeError("v24 population or alignment changed")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64, "native_sides": 128,
        "model_forwards_exact": 2, "example_evaluations_exact": 128,
        "interventions": 0, "transformer_backwards": 0, "model_updates": 0,
        "all_families_scored_without_filtering": list(FAMILIES),
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    outputs = {
        side: backend.native(das._batch(backend, rows, side=side), capture=False)
        for side in ("base", "donor")
    }
    records = []
    for index, row in enumerate(rows):
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[index]
            records.append({
                "row_id": row["row_id"], "family": row["family"],
                "direction": row["direction_id"], "side": side,
                "correct": float(answer) > float(foil),
            })
    cells = []
    for family in FAMILIES:
        for direction in DIRECTIONS[family]:
            for side in ("base", "donor"):
                selected = [record for record in records
                            if record["family"] == family
                            and record["direction"] == direction
                            and record["side"] == side]
                if len(selected) != 8:
                    raise RuntimeError(f"v24 cell size changed: {family}/{direction}/{side}")
                accuracy = sum(record["correct"] for record in selected) / len(selected)
                cells.append({"family": family, "direction": direction, "side": side,
                              "count": len(selected), "accuracy": accuracy,
                              "threshold": CELL_BAR, "passed": accuracy >= CELL_BAR})
    joint = {
        family: [row["row_id"] for row in rows if row["family"] == family and all(
            next(record["correct"] for record in records
                 if record["row_id"] == row["row_id"] and record["side"] == side)
            for side in ("base", "donor"))]
        for family in FAMILIES
    }
    family_capable = {
        family: (all(cell["passed"] for cell in cells if cell["family"] == family)
                 and len(joint[family]) >= JOINT_BAR)
        for family in FAMILIES
    }
    predictions = {
        "pred_a_authority_novelty_and_exact_population": (
            fresh.authority_sha256() == ROWS_SHA256
            and len({row["row_id"] for row in rows}) == 64),
        "pred_b_native_target_capability": all(family_capable[family]
                                                for family in TARGET_FAMILIES),
        "pred_c_native_control_capability": all(family_capable[family]
                                                 for family in CONTROL_FAMILIES),
        "pred_d_no_causal_outcome_access_and_exact_price": len(records) == 128,
    }
    if tuple(predictions) != PREDICTION_KEYS:
        raise RuntimeError("v24 prediction inventory changed")
    terminal = ("invalid" if not predictions["pred_a_authority_novelty_and_exact_population"]
                or not predictions["pred_d_no_causal_outcome_access_and_exact_price"]
                else "screen" if all(predictions.values()) else "null")
    result = {
        "schema": RESULT_SCHEMA, "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only_capability_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256,
        "dryrun": dryrun, "capability_cells": cells,
        "jointly_capable_row_ids": joint, "family_capable": family_capable,
        "native_records": records, "causal_outcomes_opened": False,
        "predictions": predictions, "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "capability_cells", "jointly_capable_row_ids",
        "family_capable", "causal_outcomes_opened", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
