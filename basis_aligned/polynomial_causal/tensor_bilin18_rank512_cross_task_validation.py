#!/usr/bin/env python3
"""Cross-task heldout validation of the complete standalone rank512 program."""

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

import early_mlp_suffix_transport_v1_rows as row_authority
import tensor_attention_projection_frontier as frontier
import tensor_bilin18_shared_qk_rank512 as rank512
import tensor_bilin18_shared_qk_whole_program as base


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
OUTPUT = HERE / "tensor_bilin18_rank512_cross_task_validation_results.json"
PREREG = HERE / "TENSOR_BILIN18_RANK512_CROSS_TASK_VALIDATION_PREREGISTRATION.md"
RANK512_PARENT = HERE / "tensor_bilin18_shared_qk_rank512_results.json"
ROLE_PATHS = {
    "cross_task_skip31000": BQ / ".rowcache_compiler_v2/fineweb_n192_skip31000.pt",
    "cross_task_skip35000": BQ / ".rowcache_compiler_v2/fineweb_n192_skip35000.pt",
}
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_shared_qk_rank512.py",
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "early_mlp_suffix_transport_v1_rows.py",
    HERE / "test_tensor_bilin18_rank512_cross_task_validation.py",
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
                raise OSError("cross-task validation publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_role(path: Path) -> dict[str, Any]:
    expected_file, expected_raw, payload_key = row_authority.CANONICAL_ROW_TENSORS[path]
    if payload_key is not None:
        raise RuntimeError("cross-task role unexpectedly uses a nested tensor payload")
    observed_file = row_authority.file_sha256(path)
    rows = torch.load(path, map_location="cpu", weights_only=True)
    observed_raw = row_authority.tensor_raw_sha256(rows)
    if observed_file != expected_file or observed_raw != expected_raw or tuple(
        rows.shape
    ) != (192, 513) or rows.dtype != torch.long:
        raise RuntimeError(f"cross-task row authority failed for {path}")
    return {
        "path": str(path.resolve()), "shape": list(rows.shape),
        "serialized_sha256": observed_file, "tensor_raw_sha256": observed_raw,
    }


def fresh_deterministic_tokens(device: torch.device) -> torch.Tensor:
    index = torch.arange(4 * frontier.T, device=device, dtype=torch.long)
    return ((index * 104_729 + 8_191) % 50_257).reshape(4, frontier.T)


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("cross-task validation result is create-only and already exists")
    started = time.time()
    role_receipts = {name: validate_role(path) for name, path in ROLE_PATHS.items()}
    parent = json.loads(RANK512_PARENT.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_roles = frontier.EVAL_ROLES
    old_fixture = base.deterministic_tokens
    old_rank = base.RANK
    try:
        frontier.EVAL_ROLES = ROLE_PATHS
        base.deterministic_tokens = fresh_deterministic_tokens
        base.RANK = rank512.RANK
        built = base.build_reference_and_program(device)
    finally:
        frontier.EVAL_ROLES = old_roles
        base.deterministic_tokens = old_fixture
        base.RANK = old_rank

    program = built.pop("program")
    model_reference = built.pop("model_reference")
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives cross-task validation boundary")

    program_roles = {
        name: base.score_program(program, rows, built["seen"])
        for name, rows in built["role_rows"].items()
    }
    program_base = program(built["tokens"])
    program_changed = program(built["changed"])
    context = base.context_metrics(
        built["native_base"], built["native_changed"], program_base, program_changed,
    )
    comparisons: dict[str, Any] = {}
    for role in built["native_roles"]:
        native = built["native_roles"][role]
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
    parent_max_all = max(
        row["all_ce_harm"] for row in parent["comparisons"].values()
    )
    parent_max_covered = max(
        row["covered_ce_harm"] for row in parent["comparisons"].values()
    )
    harms = [row["covered_ce_harm"] for row in comparisons.values()]
    cost = program.cost_receipt()
    predictions = {
        "A_complete_price_and_ownership": (
            cost["total_stored_values"] == rank512.EXPECTED_TOTAL
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"]
            and built["storage_disjoint"]
            and not built["native_module_references"]
        ),
        "B_cross_task_predictive_harm": all(
            row["all_ce_harm"] <= 0.025
            and row["covered_ce_harm"] <= 0.025
            and row["unseen_ce_harm"] <= 0.03
            for row in comparisons.values()
        ),
        "C_no_material_degradation_from_opened_roles": all(
            row["all_ce_harm"] <= parent_max_all + 0.01
            and row["covered_ce_harm"] <= parent_max_covered + 0.01
            for row in comparisons.values()
        ),
        "D_fresh_fixture_context_transport": (
            context["program_max_abs"] > 0
            and context["context_delta_recovery"] >= 0.90
            and context["context_delta_cosine"] >= 0.95
        ),
        "E_role_harm_replication_within_0.01": max(harms) - min(harms) <= 0.01,
    }
    result = {
        "status": "pass" if all(predictions.values()) else "measured_gate_failure",
        "scope": "cross-task heldout rows plus fresh synthetic context fixture",
        "checkpoint": asdict(built["checkpoint"]),
        "roles": {
            name: {"native": built["native_roles"][name], "program": program_roles[name]}
            for name in built["native_roles"]
        },
        "comparisons": comparisons,
        "context_gate": context,
        "cost": cost,
        "execution": {
            "checkpoint_model_collected_before_scoring": True,
            "native_program_storage_disjoint": built["storage_disjoint"],
            "native_module_references": built["native_module_references"],
            "program_native_calls": 0,
        },
        "predictions": predictions,
        "fit": built["fit_receipt"],
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parent_rank512_sha256": sha256_file(RANK512_PARENT),
            "roles": role_receipts,
            "fit": sha256_file(frontier.FIT_ROWS),
            "mask": sha256_file(frontier.MASK_ROWS),
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
        "context_gate": outcome["context_gate"],
        "predictions": outcome["predictions"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
