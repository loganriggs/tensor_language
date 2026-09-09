#!/usr/bin/env python3
"""Capability-only gate for the eight-structure v25 is/was OOD bank."""

# BQGATE: EXPERIMENT pred_a_hash_authority_history_structure_alignment_population_and_price pred_b_all_eight_target_constructions_are_natively_capable pred_c_all_eight_control_constructions_are_natively_capable pred_d_no_causal_outcome_access_and_exact_two_forward_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_structural_ood_v25 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT.parent / "polynomial_causal/TENSE_AUXILIARY_IS_WAS_STRUCTURAL_OOD_V25_CAPABILITY_PREREGISTRATION.md"
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_structural_ood_v25_capability_v1.json"
BUILDER = Path(fresh.__file__).resolve()
V24_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1_result.json"
V23_CELLS = ROOT / "circuits/followups/temporal_iswas_v23_source_destination_cell_atlas_v4_result.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_structural_ood_v25_capability_v1_result.json"
EXPECTED = {
    "authority": "0a0e0ecc41e4a81a75404afe22c3a37abb3e1c4d180dc79560ada68b33273377",
    "prior": "1f8b76bf8d7e3c67ca9b69ea948e5da54c0b8fec355168c80502494434fe6995",
    "builder": "e1cd4d2917773a7fbdc8c4049ab34484e5687d44ebe81a616dd078b490b7c53e",
    "v24_result": "7c03baa3e57615aacc74459fbbc67595d1a2d2842a58906338dad88f3f01a4ba",
    "v23_cells": "4f32076a076749ff5601201579660949a7099b350025a0d261d4dfdeb18d10f0",
}
ROWS_SHA256 = "69135a7af3cdd594ff7b8171256342ae681b7fff931d3fdc21806d07a478e922"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.structural_ood_v25_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_structural_ood_v25_capability_result_v1"
FAMILIES = ("A1", "A2", "P", "C")
TARGET_FAMILIES = ("A1", "A2")
CONTROL_FAMILIES = ("P", "C")
CELL_BAR = .5
JOINT_BAR = 3
PRICE = {"model_forwards_exact": 2, "sequence_evaluations_exact": 128,
         "interventions": 0, "transformer_backwards": 0, "model_updates": 0}
PREDICTION_KEYS = (
    "pred_a_hash_authority_history_structure_alignment_population_and_price",
    "pred_b_all_eight_target_constructions_are_natively_capable",
    "pred_c_all_eight_control_constructions_are_natively_capable",
    "pred_d_no_causal_outcome_access_and_exact_two_forward_price",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = {"authority": AUTHORITY, "prior": PRIOR, "builder": BUILDER,
             "v24_result": V24_RESULT, "v23_cells": V23_CELLS}
    observed = {name: sha(path) for name, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    v24 = json.loads(V24_RESULT.read_text())
    cells = json.loads(V23_CELLS.read_text())
    rows = fresh.build_rows()
    construction_ids = sorted({row["construction_id"] for row in rows})
    target_constructions = sorted({row["construction_id"] for row in rows
                                   if row["family"] in TARGET_FAMILIES})
    control_constructions = sorted({row["construction_id"] for row in rows
                                    if row["family"] in CONTROL_FAMILIES})
    token_lengths = [len(row[key]) for row in rows for key in ("base_ids", "donor_ids")]
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and v24.get("terminal") == "null" and v24.get("causal_outcomes_opened") is False
        and cells.get("terminal") == "shared_directed_cell_program_screen"
        and all(cells.get("predictions", {}).values())
        and fresh.authority_sha256() == ROWS_SHA256
        and len(rows) == len({row["row_id"] for row in rows}) == 64
        and len(construction_ids) == 16
        and len(target_constructions) == len(control_constructions) == 8
        and min(token_lengths) == 8 and max(token_lengths) == 18
        and all(row["base_semantic_position"] == row["donor_semantic_position"] for row in rows)
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
        "rows": len(rows), "native_sides": 128, "construction_ids": construction_ids,
        "target_constructions": target_constructions, "control_constructions": control_constructions,
        "token_length_range": [min(token_lengths), max(token_lengths)],
        "model_forwards_exact": 2, "sequence_evaluations_exact": 128,
        "causal_outcomes_opened": False, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("v25 structural capability authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=False)
               for side in ("base", "donor")}
    records = []
    for index, row in enumerate(rows):
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[index]
            records.append({
                "row_id": row["row_id"], "family": row["family"],
                "construction_id": row["construction_id"],
                "direction": row["direction_id"], "side": side,
                "correct": float(answer) > float(foil),
                "margin": float(answer) - float(foil),
            })
    capability_cells = []
    keys = sorted({(row["family"], row["construction_id"], row["direction_id"])
                   for row in rows})
    for family, construction_id, direction in keys:
        for side in ("base", "donor"):
            selected = [record for record in records
                        if record["family"] == family
                        and record["construction_id"] == construction_id
                        and record["direction"] == direction and record["side"] == side]
            if len(selected) != 2:
                raise RuntimeError("v25 construction/direction/side cell count changed")
            accuracy = sum(record["correct"] for record in selected) / 2
            capability_cells.append({
                "family": family, "construction_id": construction_id,
                "direction": direction, "side": side, "count": 2,
                "accuracy": accuracy, "threshold": CELL_BAR, "passed": accuracy >= CELL_BAR,
            })
    jointly_capable = {}
    construction_capable = {}
    for construction_id in construction_ids:
        construction_rows = [row for row in rows if row["construction_id"] == construction_id]
        capable_ids = [row["row_id"] for row in construction_rows if all(
            next(record["correct"] for record in records
                 if record["row_id"] == row["row_id"] and record["side"] == side)
            for side in ("base", "donor"))]
        jointly_capable[construction_id] = capable_ids
        construction_capable[construction_id] = bool(
            len(capable_ids) >= JOINT_BAR
            and all(cell["passed"] for cell in capability_cells
                    if cell["construction_id"] == construction_id))
    A = authority_ok
    B = all(construction_capable[name] for name in target_constructions)
    C = all(construction_capable[name] for name in control_constructions)
    D = len(records) == 128
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = "invalid" if not (A and D) else "screen" if B and C else "null"
    result = {
        "schema": RESULT_SCHEMA, "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only_capability_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "rows_sha256": ROWS_SHA256,
        "population": {"rows": 64, "target_constructions": target_constructions,
                       "control_constructions": control_constructions,
                       "token_length_range": [min(token_lengths), max(token_lengths)]},
        "capability_cells": capability_cells, "jointly_capable_row_ids": jointly_capable,
        "construction_capable": construction_capable, "native_records": records,
        "causal_outcomes_opened": False, "predictions": predictions,
        "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "population", "capability_cells", "jointly_capable_row_ids",
        "construction_capable", "causal_outcomes_opened", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
