#!/usr/bin/env python3
"""Cross-task predictive validation of the causally admitted rank640 program."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

import tensor_attention_projection_frontier as frontier
import tensor_bilin18_rank512_cross_task_validation as cross
import tensor_bilin18_shared_qk_whole_program as base


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
PREREG = HERE / "TENSOR_BILIN18_RANK640_PREDICTIVE_VALIDATION_PREREGISTRATION.md"
RANK512_PARENT = HERE / "tensor_bilin18_rank512_cross_task_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
RANK = 640
EXPECTED_TOTAL = 516_707_766
DENSE_TOTAL = 545_904_054
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_rank512_cross_task_validation.py",
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "test_tensor_bilin18_rank640_predictive_validation.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("rank640 predictive publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prediction_gates(
    cost: dict[str, Any], built: dict[str, Any], comparisons: dict[str, Any],
    rank512_parent: dict[str, Any], causal_parent: dict[str, Any],
) -> dict[str, bool]:
    harms = [row["covered_ce_harm"] for row in comparisons.values()]
    return {
        "A_complete_price_and_ownership": bool(
            cost["total_stored_values"] == EXPECTED_TOTAL
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"]
            and built["storage_disjoint"]
            and not built["native_module_references"]
        ),
        "B_predictive_harm": all(
            row["all_ce_harm"] <= 0.020
            and row["covered_ce_harm"] <= 0.020
            and row["unseen_ce_harm"] <= 0.025
            for row in comparisons.values()
        ),
        "C_no_material_degradation_from_rank512": all(
            row["all_ce_harm"]
            <= rank512_parent["comparisons"][role]["all_ce_harm"] + 0.002
            and row["covered_ce_harm"]
            <= rank512_parent["comparisons"][role]["covered_ce_harm"] + 0.002
            for role, row in comparisons.items()
        ),
        "D_role_harm_replication_within_0.01": max(harms) - min(harms) <= 0.010,
        "E_rank640_causal_parent_pass": bool(
            causal_parent["status"] == "rank640_robust_pass"
            and causal_parent["candidates"]["640"]["robust_gate"]
        ),
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("rank640 predictive result is create-only and already exists")
    started = time.time()
    role_receipts = {
        name: cross.validate_role(path) for name, path in cross.ROLE_PATHS.items()
    }
    rank512_parent = json.loads(RANK512_PARENT.read_text())
    causal_parent = json.loads(CAUSAL_PARENT.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_roles = frontier.EVAL_ROLES
    old_rank = base.RANK
    try:
        frontier.EVAL_ROLES = cross.ROLE_PATHS
        base.RANK = RANK
        built = base.build_reference_and_program(device)
    finally:
        frontier.EVAL_ROLES = old_roles
        base.RANK = old_rank

    program = built.pop("program")
    model_reference = built.pop("model_reference")
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives rank640 predictive boundary")
    program_roles = {
        name: base.score_program(program, rows, built["seen"])
        for name, rows in built["role_rows"].items()
    }
    comparisons: dict[str, Any] = {}
    for role, native in built["native_roles"].items():
        measured = program_roles[role]
        comparisons[role] = {
            "all_ce_harm": measured["all"]["ce"] - native["all"]["ce"],
            "covered_ce_harm": (
                measured["seen_current"]["ce"] - native["seen_current"]["ce"]
            ),
            "unseen_ce_harm": (
                measured["unseen_current"]["ce"] - native["unseen_current"]["ce"]
            ),
        }
    cost = program.cost_receipt()
    predictions = prediction_gates(
        cost, built, comparisons, rank512_parent, causal_parent,
    )
    result = {
        "status": "pass" if all(predictions.values()) else "measured_gate_failure",
        "scope": "rank640 complete standalone cross-task predictive validation",
        "rank": RANK,
        "checkpoint": asdict(built["checkpoint"]),
        "roles": {
            name: {"native": built["native_roles"][name], "program": program_roles[name]}
            for name in built["native_roles"]
        },
        "comparisons": comparisons,
        "cost": {
            **cost,
            "dense_reference_stored_values": DENSE_TOTAL,
            "stored_values_saved": DENSE_TOTAL - int(cost["total_stored_values"]),
            "stored_fraction_of_dense": int(cost["total_stored_values"]) / DENSE_TOTAL,
        },
        "operations_production_forward": program.operation_receipt(
            batch=base.BATCH, sequence=frontier.T,
        ),
        "execution": {
            "checkpoint_model_collected_before_scoring": True,
            "native_program_storage_disjoint": built["storage_disjoint"],
            "native_module_references": built["native_module_references"],
            "program_native_calls": 0,
        },
        "predictions": predictions,
        "fit": built["fit_receipt"],
        "causal_parent_summary": causal_parent["candidates"]["640"]["summary"],
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parents": {
                "rank512_cross_task": sha256_file(RANK512_PARENT),
                "causal_bank": sha256_file(CAUSAL_PARENT),
            },
            "roles": role_receipts,
            "fit": sha256_file(frontier.FIT_ROWS),
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "status": outcome["status"],
        "comparisons": outcome["comparisons"],
        "cost": {
            key: outcome["cost"][key] for key in (
                "total_stored_values", "stored_values_saved", "stored_fraction_of_dense",
            )
        },
        "predictions": outcome["predictions"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
