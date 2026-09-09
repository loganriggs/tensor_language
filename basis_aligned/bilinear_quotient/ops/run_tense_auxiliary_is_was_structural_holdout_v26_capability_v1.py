#!/usr/bin/env python3
"""Capability-only gate for the fresh four-family structural v26 holdout."""

# BQGATE: EXPERIMENT pred_a_hash_v25_history_structure_alignment_population_length_and_price pred_b_all_four_target_constructions_are_natively_capable pred_c_all_four_control_constructions_are_natively_capable pred_d_no_causal_outcome_access_and_exact_two_forward_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_structural_holdout_v26 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT.parent / "polynomial_causal/TENSE_AUXILIARY_IS_WAS_STRUCTURAL_HOLDOUT_V26_CAPABILITY_PREREGISTRATION.md"
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_structural_holdout_v26_capability_v1.json"
BUILDER = Path(fresh.__file__).resolve()
V25_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_structural_ood_v25_capability_v1_result.json"
V23_CELLS = ROOT / "circuits/followups/temporal_iswas_v23_source_destination_cell_atlas_v4_result.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_structural_holdout_v26_capability_v1_result.json"
PATHS = {"authority": AUTHORITY, "prior": PRIOR, "builder": BUILDER,
         "v25_result": V25_RESULT, "v23_cells": V23_CELLS}
EXPECTED = {
    "authority": "9cb3c79d8825ea18c211300a3d574c02b70af8158feaaecb89df0e44db97436d",
    "prior": "4d6fb4719ab37e85b38e9826553bcb946d46c10b3abb4eee67fc0a6ae1d8f9d5",
    "builder": "50a885aff484d1d94cb66a202caa161ce0d7644c55baf1e0f9cdc0a5472cb6d3",
    "v25_result": "2073c4431c9bf03b241361887ee9a4afb07edef7f22ceec28d2f0c596c55eba1",
    "v23_cells": "4f32076a076749ff5601201579660949a7099b350025a0d261d4dfdeb18d10f0",
}
ROWS_SHA256 = "805b734109a275de40f809c639f90ca2a524d82ab39467f10e2bfeb7b5ca9c15"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.structural_holdout_v26_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_structural_holdout_v26_capability_result_v1"
TARGET_FAMILIES, CONTROL_FAMILIES = ("A1", "A2"), ("P", "C")
CELL_BAR, JOINT_BAR = .75, 6
PRICE = {"model_forwards_exact": 2, "sequence_evaluations_exact": 128,
         "interventions": 0, "transformer_backwards": 0, "model_updates": 0}
PREDICTION_KEYS = (
    "pred_a_hash_v25_history_structure_alignment_population_length_and_price",
    "pred_b_all_four_target_constructions_are_natively_capable",
    "pred_c_all_four_control_constructions_are_natively_capable",
    "pred_d_no_causal_outcome_access_and_exact_two_forward_price",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    observed = {name: sha(path) for name, path in PATHS.items()}
    prior = json.loads(PRIOR.read_text())
    v25 = json.loads(V25_RESULT.read_text())
    cells = json.loads(V23_CELLS.read_text())
    rows = fresh.build_rows()
    construction_ids = sorted({row["construction_id"] for row in rows})
    target_constructions = sorted({row["construction_id"] for row in rows
                                   if row["family"] in TARGET_FAMILIES})
    control_constructions = sorted({row["construction_id"] for row in rows
                                    if row["family"] in CONTROL_FAMILIES})
    token_lengths = [len(row[key]) for row in rows for key in ("base_ids", "donor_ids")]
    length_range = [min(token_lengths), max(token_lengths)]
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and v25.get("terminal") == "null" and v25.get("causal_outcomes_opened") is False
        and cells.get("terminal") == "shared_directed_cell_program_screen"
        and all(cells.get("predictions", {}).values())
        and fresh.authority_sha256() == ROWS_SHA256
        and len(rows) == len({row["row_id"] for row in rows}) == 64
        and len(construction_ids) == 8
        and len(target_constructions) == len(control_constructions) == 4
        and length_range == [8, 17]
        and all(row["base_semantic_position"] == row["donor_semantic_position"] for row in rows)
    )
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
        "rows": len(rows), "native_sides": 128, "construction_ids": construction_ids,
        "target_constructions": target_constructions, "control_constructions": control_constructions,
        "token_length_range": length_range, "model_forwards_exact": 2,
        "sequence_evaluations_exact": 128, "causal_outcomes_opened": False, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("v26 structural holdout capability authority changed")
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
                "correct": float(answer) > float(foil), "margin": float(answer) - float(foil),
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
            if len(selected) != 4:
                raise RuntimeError("v26 construction/direction/side cell count changed")
            accuracy = sum(record["correct"] for record in selected) / 4
            capability_cells.append({
                "family": family, "construction_id": construction_id,
                "direction": direction, "side": side, "count": 4,
                "accuracy": accuracy, "threshold": CELL_BAR, "passed": accuracy >= CELL_BAR,
            })
    jointly_capable, construction_capable = {}, {}
    for construction_id in construction_ids:
        construction_rows = [row for row in rows if row["construction_id"] == construction_id]
        capable_ids = [row["row_id"] for row in construction_rows if all(
            next(record["correct"] for record in records
                 if record["row_id"] == row["row_id"] and record["side"] == side)
            for side in ("base", "donor"))]
        jointly_capable[construction_id] = capable_ids
        construction_capable[construction_id] = bool(
            len(capable_ids) >= JOINT_BAR and all(
                cell["passed"] for cell in capability_cells
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
                       "token_length_range": length_range},
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
