#!/usr/bin/env python3
"""Capability-only gate for token-aligned v15 P/C controls."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_and_exact_population pred_b_each_panel_side_cell_capable pred_c_each_panel_joint_population_capable pred_d_no_causal_access_and_exact_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as aligned
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
from capability_population_contract import assert_declared_counts, derive_panel_contract
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_v15_aligned_controls_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
ALIGNMENT_CONTRACT = ROOT / "ops/aligned_full_sequence_patch_contract.py"
POPULATION_CONTRACT = ROOT / "ops/capability_population_contract.py"
BUILDER_TEST = ROOT / "ops/test_circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
ALIGNMENT_TEST = ROOT / "ops/test_aligned_full_sequence_patch_contract.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_v15_aligned_controls_capability_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.v15_aligned_controls_capability_v1"
ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
MINIMUM_FRACTION = 0.875
PANELS = ("P", "C")
EXPECTED = {
    "prior": "21787920152cf2f1f6b4f9315c0bfe5a06bd508357f532ce86f0ef2746e3c2c5",
    "builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "alignment_contract": "7b4b04cec6f28b47b7d22b8e4be32421bbfae6a04f1adc7c0b890939c9e32760",
    "population_contract": "acdf74ffe8eaa729e7704a2f66c0dde7d9543d7dedfdfeffe8ac7f88c6aef288",
    "builder_test": "37a1d33e62fbf448a9a2a6ffbd13edd4e7c202b109a87c8a6ad41a8876102c20",
    "alignment_test": "29452d478ad318af6768c0294c4297c253788d59a863f9a74b265a611eaa193f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {
        "prior": PRIOR,
        "builder": BUILDER,
        "alignment_contract": ALIGNMENT_CONTRACT,
        "population_contract": POPULATION_CONTRACT,
        "builder_test": BUILDER_TEST,
        "alignment_test": ALIGNMENT_TEST,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"aligned-control capability authority changed: {observed}")
    prior = json.loads(PRIOR.read_text())
    all_rows = aligned.build_rows()
    rows = [row for row in all_rows if row["transform_id"] in PANELS]
    alignment = derive_full_sequence_alignment_contract(rows, required_panels=PANELS)
    population = derive_panel_contract(
        rows, panels=PANELS, minimum_fraction=MINIMUM_FRACTION
    )
    assert_declared_counts(
        population,
        declared_denominators={"P": 16, "C": 16},
        declared_required_counts={"P": 14, "C": 14},
    )
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or aligned.validate_rows(all_rows) != ROWS_SHA256
        or len(all_rows) != 64
        or len(rows) != 32
        or alignment["panel_counts"] != {"P": 16, "C": 16}
    ):
        raise RuntimeError("aligned-control capability population changed")

    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "all_authority_rows": 64,
        "executed_control_rows": 32,
        "panels": list(PANELS),
        "native_sides": 64,
        "model_forwards_exact": 2,
        "example_evaluations_exact": 64,
        "interventions": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
        "a1_a2_outcomes_opened": False,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    model_forwards = 0
    outputs = {}
    for side in ("base", "donor"):
        outputs[side] = backend.native(das._batch(backend, rows, side=side), capture=False)
        model_forwards += 1

    records = []
    for index, row in enumerate(rows):
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[index]
            answer_logit, foil_logit = float(answer), float(foil)
            records.append(
                {
                    "row_id": row["row_id"],
                    "panel": row["transform_id"],
                    "construction_id": row["construction_id"],
                    "direction_id": row["direction_id"],
                    "side": side,
                    "answer_logit": answer_logit,
                    "foil_logit": foil_logit,
                    "margin": answer_logit - foil_logit,
                    "correct": answer_logit > foil_logit,
                }
            )

    cells = []
    for panel in PANELS:
        for side in ("base", "donor"):
            selected = [
                record for record in records
                if record["panel"] == panel and record["side"] == side
            ]
            accuracy = sum(record["correct"] for record in selected) / len(selected)
            cells.append(
                {
                    "panel": panel,
                    "side": side,
                    "count": len(selected),
                    "correct_count": sum(record["correct"] for record in selected),
                    "accuracy": accuracy,
                    "threshold": MINIMUM_FRACTION,
                    "passed": accuracy >= MINIMUM_FRACTION,
                }
            )
    joint = {
        panel: [
            row["row_id"] for row in rows if row["transform_id"] == panel
            and all(
                next(
                    record["correct"] for record in records
                    if record["row_id"] == row["row_id"] and record["side"] == side
                )
                for side in ("base", "donor")
            )
        ]
        for panel in PANELS
    }

    pred_a = bool(
        aligned.authority_sha256() == ROWS_SHA256
        and len({row["row_id"] for row in rows}) == 32
        and alignment["panel_counts"] == population["panel_counts"] == {"P": 16, "C": 16}
        and population["required_jointly_capable_counts"] == {"P": 14, "C": 14}
    )
    pred_b = all(cell["passed"] for cell in cells)
    pred_c = all(
        len(joint[panel]) >= population["required_jointly_capable_counts"][panel]
        for panel in PANELS
    )
    example_evaluations = len(records)
    pred_d = bool(
        model_forwards == 2
        and example_evaluations == 64
        and len(records) == 64
        and all(math.isfinite(record["margin"]) for record in records)
    )
    predictions = {
        "pred_a_authority_alignment_and_exact_population": pred_a,
        "pred_b_each_panel_side_cell_capable": pred_b,
        "pred_c_each_panel_joint_population_capable": pred_c,
        "pred_d_no_causal_access_and_exact_price": pred_d,
    }
    terminal = "invalid" if not pred_a or not pred_d else "manifest" if pred_b and pred_c else "null"
    result = {
        "schema": "tense_auxiliary_is_was_v15_aligned_controls_capability_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only_capability_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": ROWS_SHA256,
        "alignment_contract": alignment,
        "population_contract": population,
        "dryrun": dryrun,
        "capability_cells": cells,
        "jointly_capable_row_ids": joint,
        "native_records": records,
        "causal_outcomes_opened": False,
        "a1_a2_outcomes_opened": False,
        "predictions": predictions,
        "terminal": terminal,
        "price": {
            "model_forwards": model_forwards,
            "example_evaluations": example_evaluations,
            "interventions": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
    }
    atomic_create_json(OUT, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "candidate_id", "capability_cells", "jointly_capable_row_ids",
                    "causal_outcomes_opened", "a1_a2_outcomes_opened", "predictions",
                    "terminal", "price",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
