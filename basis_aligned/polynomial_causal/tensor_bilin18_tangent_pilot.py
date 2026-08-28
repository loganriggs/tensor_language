#!/usr/bin/env python3
"""Create-only stage-1 finite-horizon tangent pilot on admitted rank640 bilin18."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import weakref

import torch

import bilin18_observed_model_facade as facade
import finite_horizon_tangent_realization as realization
from finite_horizon_tangent_response_bank import (
    TangentResponseBankTransaction,
    TangentResponsePlan,
    allocate_whole_document_splits,
    analyze_bank,
)
import freeze_finite_horizon_tangent_plan as frozen
import tensor_attention_projection_frontier as frontier
from tensor_bilin18_program import TensorBilin18Program
from tensor_bilin18_tangent_collector import (
    PRODUCTION_BATCH,
    TensorBilin18TangentTransaction,
    collect_write_geometry_bank,
)
import tensor_bilin18_shared_qk_whole_program as shared
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_tangent_pilot_results.json"
PREREG = HERE / "FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md"
PLAN_RESULT = HERE / "finite_horizon_tangent_plan.json"
RANK640_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
RANK = 640
CUTS = (1, 2, 3)
EXPECTED_PLAN_FINGERPRINT = "062ad87d552112bd2064726848a5f3d1a1e1ee13118e01cf3a4b462c2c8e0141"
SOURCES = (
    Path(__file__).resolve(), PREREG, PLAN_RESULT,
    HERE / "tensor_bilin18_tangent_collector.py",
    HERE / "finite_horizon_tangent_response_bank.py",
    HERE / "finite_horizon_tangent_realization.py",
    HERE / "freeze_finite_horizon_tangent_plan.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "test_tensor_bilin18_tangent_pilot.py",
    HERE / "test_tensor_bilin18_tangent_collector.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(value: MappingLike) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("tangent pilot publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


MappingLike = dict[str, Any]


def load_plan_and_rows() -> tuple[TangentResponsePlan, torch.Tensor, dict[str, Any]]:
    rows = torch.load(frozen.ROWS, map_location="cpu", weights_only=True)
    authority = json.loads(frozen.AUTHORITY.read_text())
    provenance = authority["document_provenance"]["sets"]["n96_skip80"]
    raw_hash = frozen.tensor_raw_sha256(rows)
    if frozen.file_sha256(frozen.ROWS) != frozen.EXPECTED_FILE_SHA256 or (
        raw_hash != frozen.EXPECTED_RAW_SHA256
    ) or authority["entries"]["n96_skip80"]["tensor_raw_sha256"] != raw_hash or (
        len(provenance) != len(rows)
    ):
        raise RuntimeError("tangent pilot row authority changed")
    row_ids = tuple(f"n96_skip80:{index}" for index in range(len(rows)))
    document_ids = tuple(record["document_id"] for record in provenance)
    plan = TangentResponsePlan(
        experiment_id="bilin18-early-mlp-finite-horizon-tangent-v1",
        row_artifact_sha256=raw_hash,
        row_ids=row_ids,
        document_ids=document_ids,
        splits=allocate_whole_document_splits(document_ids),
        scored_positions=tuple(frozen.scored_position(row_id) for row_id in row_ids),
        input_dims=((0, 32), (1, 32), (2, 32)), target_site=3,
        probes_per_row=16, direction_seed=2026082801,
        probe_seed=2026082802, position_seed=2026082803,
    )
    frozen_result = json.loads(PLAN_RESULT.read_text())
    if plan.fingerprint != EXPECTED_PLAN_FINGERPRINT or frozen_result[
        "plan_fingerprint"
    ] != plan.fingerprint:
        raise RuntimeError("tangent plan fingerprint changed")
    return plan, rows, authority


@torch.no_grad()
def build_rank640_program(device: torch.device) -> tuple[TensorBilin18Program, dict[str, Any]]:
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    fit_rows = frontier.load_rows(frontier.FIT_ROWS, 480)
    previous_rank = shared.RANK
    try:
        shared.RANK = RANK
        attention_bank, fit_receipt = shared.compile_shared_bank(model, fit_rows)
    finally:
        shared.RANK = previous_rank
    blocks = tuple(model.transformer.h)
    program = TensorBilin18Program(
        token_embedding=model.transformer.wte.weight.detach(),
        residual_lambdas=torch.stack([block.lambdas.detach() for block in blocks]),
        unembedding=model.lm_head.weight.detach(),
        attention_bank=attention_bank,
        mlp_bank=TensorMLPBank.from_model(model),
    )
    model_storage = {
        value.untyped_storage().data_ptr()
        for value in tuple(model.parameters()) + tuple(model.buffers()) if value.numel()
    }
    program_storage = {
        value.untyped_storage().data_ptr()
        for value in tuple(program.parameters()) + tuple(program.buffers()) if value.numel()
    }
    if not model_storage.isdisjoint(program_storage):
        raise RuntimeError("rank640 tangent program aliases checkpoint storage")
    reference = weakref.ref(model)
    del blocks, model, fit_rows
    gc.collect()
    if reference() is not None:
        raise RuntimeError("checkpoint survives rank640 tangent construction")
    cost = program.cost_receipt()
    if int(cost["total_stored_values"]) != 516_707_766:
        raise RuntimeError("rank640 tangent program price changed")
    return program, {
        "checkpoint": asdict(checkpoint),
        "attention_fit": fit_receipt,
        "cost": cost,
        "checkpoint_collected": True,
        "checkpoint_storage_disjoint": True,
    }


def stage1_passes(stability: dict[str, Any]) -> bool:
    return set(stability) == {"1", "2", "3"} and all(
        row["passes"] for row in stability.values()
    )


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("tangent pilot result is create-only and already exists")
    started = time.time()
    plan, rows, authority = load_plan_and_rows()
    rank640_parent = json.loads(RANK640_PARENT.read_text())
    causal_parent = json.loads(CAUSAL_PARENT.read_text())
    if rank640_parent["status"] != "pass" or causal_parent[
        "status"
    ] != "rank640_robust_pass":
        raise RuntimeError("admitted rank640 parent certificates changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_started = time.time()
    program, program_receipt = build_rank640_program(device)
    build_seconds = time.time() - build_started
    geometry_started = time.time()
    geometry_bank = collect_write_geometry_bank(program, rows, plan, production=True)
    geometry_seconds = time.time() - geometry_started
    response_started = time.time()
    bank_transaction = TangentResponseBankTransaction(plan)
    batch_receipts = []
    target_hashes = set()
    for start in range(0, len(rows), PRODUCTION_BATCH):
        stop = start + PRODUCTION_BATCH
        row_ids = plan.row_ids[start:stop]
        transaction = TensorBilin18TangentTransaction(
            program=program, plan=plan, row_ids=row_ids,
            tokens=rows[start:stop, :256].to(device).contiguous(),
            geometries=geometry_bank.geometries, production=True,
        )
        batch_result = transaction.consume()
        if not transaction.aliases_revoked or batch_result.receipt[
            "plan_fingerprint"
        ] != plan.fingerprint:
            raise RuntimeError("tangent batch transaction did not close")
        target_hash = str(batch_result.receipt["target_ids_sha256"])
        if target_hash in target_hashes:
            raise RuntimeError("categorical target bank replayed across tangent batches")
        target_hashes.add(target_hash)
        for row_id in row_ids:
            bank_transaction.add_row(row_id, batch_result.responses[row_id])
        batch_receipts.append(dict(batch_result.receipt))
        del transaction, batch_result
        torch.cuda.empty_cache()
        print(f"tangent response batch {stop // PRODUCTION_BATCH}/24", flush=True)
    if len(target_hashes) != 24:
        raise RuntimeError("tangent target hash ledger is incomplete")
    bank = bank_transaction.seal()
    response_seconds = time.time() - response_started
    if not bank_transaction.aliases_revoked:
        raise RuntimeError("tangent response bank did not revoke aliases")
    analyses = analyze_bank(bank, CUTS)
    contextwise = {
        split: realization.analyze_contextwise_cuts(
            blocks, bank.input_dims, bank.output_dims_by_split[split], CUTS,
            probes_per_context=plan.probes_per_row,
        )
        for split, blocks in bank.split_blocks.items()
    }
    stability = realization.compare_split_cuts(
        bank.split_blocks["primary"], bank.split_blocks["replication"],
        bank.input_dims, bank.output_dims_by_split["primary"], CUTS,
    )
    passes = stage1_passes(stability)
    result = {
        "status": "stable_cut_state_candidate" if passes else "measured_no_stable_cut_knee",
        "scope": "stage-1 final-output Fisher tangent realization at MLP0-2",
        "plan_fingerprint": plan.fingerprint,
        "program": program_receipt,
        "geometry": geometry_bank.receipt,
        "response_bank": bank.receipt,
        "analyses": analyses,
        "contextwise_analyses": contextwise,
        "split_stability": stability,
        "stage1_pass": passes,
        "consequence_stage_authorized": passes,
        "execution": {
            "batches": len(batch_receipts),
            "unique_target_hashes": len(target_hashes),
            "batch_receipts": batch_receipts,
            "raw_logits_returned": False,
            "raw_vjps_returned": False,
            "raw_write_codes_returned": False,
            "checkpoint_collected_before_response_measurement": True,
        },
        "parents": {
            "rank640_predictive_sha256": sha256_file(RANK640_PARENT),
            "rank640_causal_sha256": sha256_file(CAUSAL_PARENT),
        },
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "rows": {
                "authority": authority["authority"],
                "receipt_sha256": sha256_file(frozen.AUTHORITY),
                "file_sha256": frozen.file_sha256(frozen.ROWS),
                "tensor_raw_sha256": frozen.tensor_raw_sha256(rows),
            },
        },
        "runtime_s": {
            "program_build": build_seconds,
            "write_geometry": geometry_seconds,
            "fisher_responses": response_seconds,
            "total": time.time() - started,
        },
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "status": outcome["status"],
        "stage1_pass": outcome["stage1_pass"],
        "split_stability": outcome["split_stability"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
