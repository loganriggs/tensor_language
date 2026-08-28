#!/usr/bin/env python3
"""Shared-QK rank-512 complete-program causal discriminator."""

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
import tensor_bilin18_shared_qk_whole_program as base


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_shared_qk_rank512_results.json"
PREREG = HERE / "TENSOR_BILIN18_SHARED_QK_RANK512_PREREGISTRATION.md"
RANK384_PARENT = HERE / "tensor_bilin18_shared_qk_whole_program_results.json"
EXACT_PARENT = HERE / "tensor_bilin18_standalone_identity_results.json"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "test_tensor_bilin18_shared_qk_rank512.py",
)
RANK = 512
EXPECTED_TOTAL = 503_436_726
DENSE_TOTAL = 545_904_054


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
                raise OSError("rank512 publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("rank512 result is create-only and already exists")
    started = time.time()
    base.RANK = RANK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = base.build_reference_and_program(device)
    program = built.pop("program")
    model_reference = built.pop("model_reference")
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives rank512 construction boundary")

    program_roles = {
        name: base.score_program(program, rows, built["seen"])
        for name, rows in built["role_rows"].items()
    }
    program_base = program(built["tokens"])
    program_changed = program(built["changed"])
    context = base.context_metrics(
        built["native_base"], built["native_changed"], program_base, program_changed,
    )
    rank384 = json.loads(RANK384_PARENT.read_text())
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
            "covered_harm_minus_rank384": (
                measured["seen_current"]["ce"] - native["seen_current"]["ce"]
                - rank384["comparisons"][role]["covered_ce_harm"]
            ),
            "all_harm_minus_rank384": (
                measured["all"]["ce"] - native["all"]["ce"]
                - rank384["comparisons"][role]["all_ce_harm"]
            ),
        }
    cost = program.cost_receipt()
    harms = [row["covered_ce_harm"] for row in comparisons.values()]
    recovery_gain = (
        context["context_delta_recovery"]
        - rank384["context_gate"]["context_delta_recovery"]
    )
    cosine_gain = (
        context["context_delta_cosine"]
        - rank384["context_gate"]["context_delta_cosine"]
    )
    predictions = {
        "A_complete_price_and_ownership": (
            cost["total_stored_values"] == EXPECTED_TOTAL
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"]
            and built["storage_disjoint"]
            and not built["native_module_references"]
        ),
        "B_predictive_harm_and_rank384_dominance": all(
            row["covered_ce_harm"] <= 0.025 and row["all_ce_harm"] <= 0.025
            and row["covered_harm_minus_rank384"] <= 0.001
            and row["all_harm_minus_rank384"] <= 0.001
            for row in comparisons.values()
        ),
        "C_context_gate": (
            context["context_delta_recovery"] >= 0.90
            and context["context_delta_cosine"] >= 0.95
        ),
        "D_rank_was_limiting": recovery_gain >= 0.03 and cosine_gain >= 0.02,
        "E_role_harm_replication_within_0.01": max(harms) - min(harms) <= 0.01,
    }
    result = {
        "status": "pass" if all(predictions.values()) else "measured_gate_failure",
        "scope": "shared-QK-512 attention plus exact MLPs in complete standalone program",
        "rank": RANK,
        "checkpoint": asdict(built["checkpoint"]),
        "roles": {
            name: {"native": built["native_roles"][name], "program": program_roles[name]}
            for name in built["native_roles"]
        },
        "comparisons": comparisons,
        "context_gate": {
            **context,
            "recovery_gain_vs_rank384": recovery_gain,
            "cosine_gain_vs_rank384": cosine_gain,
        },
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
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parents": {
                "rank384": sha256_file(RANK384_PARENT),
                "exact_complete_program": sha256_file(EXACT_PARENT),
            },
            "roles": {
                "fit": sha256_file(frontier.FIT_ROWS),
                "mask": sha256_file(frontier.MASK_ROWS),
                **{name: sha256_file(path) for name, path in frontier.EVAL_ROLES.items()},
            },
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
        "cost": {
            key: outcome["cost"][key] for key in (
                "total_stored_values", "stored_values_saved", "stored_fraction_of_dense",
            )
        },
        "predictions": outcome["predictions"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
