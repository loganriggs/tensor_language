#!/usr/bin/env python3
"""Create-only stage-1 finite-horizon tangent pilot on admitted rank640 bilin18."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import weakref

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    SCORE_STOP,
    TensorBilin18TangentTransaction,
    WriteCovarianceGeometry,
    WriteGeometryBank,
    _json_sha256,
    _tensor_sha256,
    collect_write_geometry_bank,
)
import tensor_bilin18_shared_qk_whole_program as shared
import tensor_bilin18_tangent_authority as tangent_authority
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_tangent_pilot_results.json"
AUTHORITY_RECEIPT = HERE / "tensor_bilin18_tangent_authority_receipt.json"
GEOMETRY_ARTIFACT = HERE / "tensor_bilin18_tangent_geometry.pt"
GEOMETRY_RECEIPT = HERE / "tensor_bilin18_tangent_geometry_receipt.json"
RUN_LOCK = HERE / ".tensor_bilin18_tangent_pilot.lock"
PREREG = HERE / "FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md"
PLAN_RESULT = HERE / "finite_horizon_tangent_plan.json"
RANK640_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
RANK = 640
CUTS = (1, 2, 3)
EXPECTED_PLAN_FINGERPRINT = "b9caa7ce2ecbd63a197262098931541c32dce27ed31b35454753b773f8cf4e20"
SOURCES = (
    Path(__file__).resolve(), PREREG, PLAN_RESULT,
    HERE / "tensor_bilin18_tangent_authority.py",
    HERE / "tensor_bilin18_tangent_collector.py",
    HERE / "finite_horizon_tangent_response_bank.py",
    HERE / "finite_horizon_tangent_realization.py",
    HERE / "freeze_finite_horizon_tangent_plan.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_attention_identity.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "bilin18_observed_model_facade.py",
    ROOT / "jacclust/tt_model.py",
    HERE / "test_tensor_bilin18_program.py",
    HERE / "test_tensor_bilin18_tangent_pilot.py",
    HERE / "test_tensor_bilin18_tangent_collector.py",
    HERE / "test_finite_horizon_tangent_response_bank.py",
    HERE / "test_finite_horizon_tangent_realization.py",
    HERE / "test_freeze_finite_horizon_tangent_plan.py",
)


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
    attention_bank, fit_receipt = shared.compile_shared_bank(model, fit_rows, rank=RANK)
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


def freeze_program_authority(
    device: torch.device, run_lock: tangent_authority.RunLock,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a no-outcome receipt for the exact rebuilt rank640 tensor program."""
    if not isinstance(run_lock, tangent_authority.RunLock):
        raise TypeError("program authority requires an owned run lock")
    run_lock.assert_owned()
    if AUTHORITY_RECEIPT.exists():
        raise RuntimeError("tangent program authority is create-only and already exists")
    before = tangent_authority.protected_snapshot(SOURCES)
    program, receipt = build_rank640_program(device)
    tangent_authority.validate_program_receipt(receipt)
    manifest = tangent_authority.program_buffer_manifest(program)
    after = tangent_authority.protected_snapshot(SOURCES)
    if after != before:
        raise RuntimeError("protected inputs changed while freezing program authority")
    result = {
        "status": "rank640_program_authority_frozen_no_outcomes",
        "rank": RANK,
        "protected_snapshot": before,
        "program_receipt": receipt,
        "program_buffers": manifest,
        "outcomes_computed": False,
        "geometry_computed": False,
        "runtime_environment": dict(runtime_environment),
    }
    tangent_authority.publish_json_create_only(
        AUTHORITY_RECEIPT, result, ownership_check=run_lock.assert_owned,
    )
    return result


def validate_frozen_program_authority(
    value: Any, *, protected_snapshot: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> None:
    required = {
        "status", "rank", "protected_snapshot", "program_receipt", "program_buffers",
        "outcomes_computed", "geometry_computed", "runtime_environment",
    }
    if not isinstance(value, dict) or set(value) != required or value["status"] != (
        "rank640_program_authority_frozen_no_outcomes"
    ) or value["rank"] != RANK or value["outcomes_computed"] is not False or (
        value["geometry_computed"] is not False
    ) or value["protected_snapshot"] != dict(protected_snapshot) or (
        value["runtime_environment"] != dict(runtime_environment)
    ):
        raise RuntimeError("frozen tangent program authority schema is invalid")
    tangent_authority.validate_program_receipt(value["program_receipt"])
    manifest = value["program_buffers"]
    if not isinstance(manifest, dict) or set(manifest) != {
        "entries", "buffers", "total_values", "total_bytes", "tree_sha256",
        "manifest_sha256",
    } or manifest["manifest_sha256"] != tangent_authority.canonical_sha256({
        key: manifest[key] for key in manifest if key != "manifest_sha256"
    }):
        raise RuntimeError("frozen tangent program buffer manifest is invalid")


def geometry_payload(bank: WriteGeometryBank) -> dict[str, Any]:
    return {
        "geometries": {
            str(site): {
                "site": geometry.site, "count": geometry.count,
                "mean": geometry.mean.clone(), "covariance": geometry.covariance.clone(),
                "support_rank": geometry.support_rank,
                "eigenvalues": geometry.eigenvalues.clone(),
                "directions": geometry.directions.clone(),
                "covariance_sha256": geometry.covariance_sha256,
                "directions_sha256": geometry.directions_sha256,
                "psd_rtol": geometry.psd_rtol, "support_rtol": geometry.support_rtol,
            }
            for site, geometry in bank.geometries.items()
        },
        "receipt": dict(bank.receipt),
    }


def load_frozen_geometry() -> WriteGeometryBank:
    if not GEOMETRY_ARTIFACT.exists() or not GEOMETRY_RECEIPT.exists():
        raise RuntimeError("freeze and audit tangent geometry before outcomes")
    receipt = json.loads(GEOMETRY_RECEIPT.read_text())
    required_receipt = {
        "status", "protected_snapshot_fingerprint", "program_tree_sha256",
        "program_authority_sha256",
        "plan_fingerprint", "artifact_sha256", "geometry_receipt_sha256",
        "geometry_receipt", "score_targets_sampled", "score_gradients_computed",
        "runtime_environment",
    }
    if set(receipt) != required_receipt or receipt["status"] != (
        "tangent_geometry_frozen_no_score_outcomes"
    ) or receipt["score_targets_sampled"] is not False or (
        receipt["score_gradients_computed"] is not False
    ):
        raise RuntimeError("frozen tangent geometry authority schema is invalid")
    if tangent_authority.sha256_file(GEOMETRY_ARTIFACT) != receipt.get("artifact_sha256"):
        raise RuntimeError("frozen tangent geometry artifact bytes changed")
    payload = torch.load(GEOMETRY_ARTIFACT, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"geometries", "receipt"} or (
        payload["receipt"] != receipt["geometry_receipt"]
    ) or _json_sha256(payload["receipt"]) != receipt["geometry_receipt_sha256"] or (
        set(payload["geometries"]) != {"0", "1", "2"}
    ):
        raise RuntimeError("frozen tangent geometry payload and authority disagree")
    geometries = {}
    for site_text, row in payload["geometries"].items():
        site = int(site_text)
        required_geometry = {
            "site", "count", "mean", "covariance", "support_rank", "eigenvalues",
            "directions", "covariance_sha256", "directions_sha256", "psd_rtol",
            "support_rtol",
        }
        if set(row) != required_geometry or row["site"] != site or row["count"] != 18_432:
            raise RuntimeError("frozen tangent geometry site schema is invalid")
        geometry = WriteCovarianceGeometry(**row)
        if tuple(geometry.mean.shape) != (1152,) or tuple(geometry.covariance.shape) != (
            1152, 1152,
        ) or tuple(geometry.eigenvalues.shape) != (1152,) or tuple(
            geometry.directions.shape
        ) != (32, 1152) or any(
            value.device.type != "cpu" or value.dtype != torch.float64
            or value.requires_grad or not bool(torch.isfinite(value).all())
            for value in (
                geometry.mean, geometry.covariance, geometry.eigenvalues,
                geometry.directions,
            )
        ) or geometry.psd_rtol != 1e-10 or geometry.support_rtol != 1e-12 or not (
            1 <= geometry.support_rank <= 1152
        ):
            raise RuntimeError("frozen tangent geometry tensor contract is invalid")
        if _tensor_sha256(geometry.covariance) != geometry.covariance_sha256 or (
            _tensor_sha256(geometry.directions) != geometry.directions_sha256
        ):
            raise RuntimeError("frozen tangent geometry tensor hash changed")
        geometries[site] = geometry
    bank = WriteGeometryBank(geometries=geometries, receipt=payload["receipt"])
    sites = bank.receipt.get("sites", {})
    if bank.receipt.get("status") != "complete" or set(sites) != {"0", "1", "2"} or (
        bank.receipt.get("write_samples_per_site") != 18_432
    ) or bank.receipt.get("score_support") != [64, 256] or any(
        sites[str(site)] != {
            "count": geometry.count, "support_rank": geometry.support_rank,
            "covariance_sha256": geometry.covariance_sha256,
            "directions_sha256": geometry.directions_sha256,
        }
        for site, geometry in geometries.items()
    ) or bank.receipt.get("geometry_manifest_sha256") != _json_sha256(sites
    ):
        raise RuntimeError("frozen tangent geometry manifest is invalid")
    return bank


def require_geometry_replay_identity(
    frozen_bank: WriteGeometryBank, replayed_bank: WriteGeometryBank,
) -> None:
    if frozen_bank.receipt != replayed_bank.receipt:
        raise RuntimeError("frozen tangent geometry receipt differs from exact replay")
    for site in (0, 1, 2):
        frozen_geometry = frozen_bank.geometries[site]
        replayed_geometry = replayed_bank.geometries[site]
        if any(_tensor_sha256(getattr(frozen_geometry, field)) != _tensor_sha256(
            getattr(replayed_geometry, field)
        ) for field in ("mean", "covariance", "eigenvalues", "directions")) or any(
            getattr(frozen_geometry, field) != getattr(replayed_geometry, field)
            for field in (
                "site", "count", "support_rank", "covariance_sha256",
                "directions_sha256", "psd_rtol", "support_rtol",
            )
        ):
            raise RuntimeError("frozen tangent geometry differs from exact program/row replay")


def freeze_geometry_authority(
    device: torch.device, run_lock: tangent_authority.RunLock,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze natural-write directions without sampling or differentiating scores."""
    if not isinstance(run_lock, tangent_authority.RunLock):
        raise TypeError("geometry authority requires an owned run lock")
    run_lock.assert_owned()
    if GEOMETRY_ARTIFACT.exists() or GEOMETRY_RECEIPT.exists():
        raise RuntimeError("tangent geometry authority is create-only and already exists")
    if not AUTHORITY_RECEIPT.exists():
        raise RuntimeError("freeze program authority before tangent geometry")
    protected = tangent_authority.protected_snapshot(SOURCES)
    program_authority_sha256 = tangent_authority.sha256_file(AUTHORITY_RECEIPT)
    program_authority = json.loads(AUTHORITY_RECEIPT.read_text())
    validate_frozen_program_authority(
        program_authority, protected_snapshot=protected,
        runtime_environment=runtime_environment,
    )
    plan, rows, _ = load_plan_and_rows()
    tangent_authority.validate_loaded_rows(rows)
    program, program_receipt = build_rank640_program(device)
    tangent_authority.validate_program_receipt(program_receipt)
    manifest = tangent_authority.program_buffer_manifest(program)
    if manifest != program_authority.get("program_buffers"):
        raise RuntimeError("geometry program differs from frozen program authority")
    bank = collect_write_geometry_bank(program, rows, plan, production=True)
    if tangent_authority.protected_snapshot(SOURCES) != protected or (
        tangent_authority.program_buffer_manifest(program) != manifest
    ) or tangent_authority.sha256_file(AUTHORITY_RECEIPT) != program_authority_sha256:
        raise RuntimeError("protected inputs or program changed while freezing geometry")
    tangent_authority.publish_torch_create_only(
        GEOMETRY_ARTIFACT, geometry_payload(bank), ownership_check=run_lock.assert_owned,
    )
    receipt = {
        "status": "tangent_geometry_frozen_no_score_outcomes",
        "protected_snapshot_fingerprint": protected["fingerprint"],
        "program_tree_sha256": manifest["tree_sha256"],
        "program_authority_sha256": program_authority_sha256,
        "plan_fingerprint": plan.fingerprint,
        "artifact_sha256": tangent_authority.sha256_file(GEOMETRY_ARTIFACT),
        "geometry_receipt_sha256": _json_sha256(bank.receipt),
        "geometry_receipt": bank.receipt,
        "score_targets_sampled": False,
        "score_gradients_computed": False,
        "runtime_environment": dict(runtime_environment),
    }
    tangent_authority.publish_json_create_only(
        GEOMETRY_RECEIPT, receipt, ownership_check=run_lock.assert_owned,
    )
    return receipt


def stage1_passes(stability: dict[str, Any]) -> bool:
    return set(stability) == {"1", "2", "3"} and all(
        row["passes"] for row in stability.values()
    )


def run(
    run_lock: tangent_authority.RunLock, runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(run_lock, tangent_authority.RunLock):
        raise TypeError("tangent outcomes require an owned run lock")
    run_lock.assert_owned()
    if OUTPUT.exists():
        raise RuntimeError("tangent pilot result is create-only and already exists")
    if not AUTHORITY_RECEIPT.exists() or not GEOMETRY_RECEIPT.exists():
        raise RuntimeError("freeze and audit program plus geometry authority before outcomes")
    started = time.time()
    protected_before = tangent_authority.protected_snapshot(SOURCES)
    program_authority_sha256 = tangent_authority.sha256_file(AUTHORITY_RECEIPT)
    geometry_authority_sha256 = tangent_authority.sha256_file(GEOMETRY_RECEIPT)
    geometry_artifact_sha256 = tangent_authority.sha256_file(GEOMETRY_ARTIFACT)
    frozen_authority = json.loads(AUTHORITY_RECEIPT.read_text())
    validate_frozen_program_authority(
        frozen_authority, protected_snapshot=protected_before,
        runtime_environment=runtime_environment,
    )
    plan, rows, row_authority = load_plan_and_rows()
    tangent_authority.validate_loaded_rows(rows)
    rank640_parent = json.loads(RANK640_PARENT.read_text())
    causal_parent = json.loads(CAUSAL_PARENT.read_text())
    if rank640_parent["status"] != "pass" or causal_parent[
        "status"
    ] != "rank640_robust_pass":
        raise RuntimeError("admitted rank640 parent certificates changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_started = time.time()
    program, program_receipt = build_rank640_program(device)
    tangent_authority.validate_program_receipt(program_receipt)
    program_manifest = tangent_authority.program_buffer_manifest(program)
    if program_manifest != frozen_authority.get("program_buffers"):
        raise RuntimeError("rebuilt rank640 program differs from frozen authority")
    build_seconds = time.time() - build_started
    geometry_started = time.time()
    geometry_bank = load_frozen_geometry()
    replayed_geometry = collect_write_geometry_bank(program, rows, plan, production=True)
    require_geometry_replay_identity(geometry_bank, replayed_geometry)
    del replayed_geometry
    geometry_seconds = time.time() - geometry_started
    frozen_geometry_receipt = json.loads(GEOMETRY_RECEIPT.read_text())
    if frozen_geometry_receipt.get("protected_snapshot_fingerprint") != protected_before[
        "fingerprint"
    ] or frozen_geometry_receipt.get("program_tree_sha256") != program_manifest[
        "tree_sha256"
    ] or frozen_geometry_receipt.get("plan_fingerprint") != plan.fingerprint:
        raise RuntimeError("frozen tangent geometry authority does not match this run")
    if frozen_geometry_receipt.get("program_authority_sha256") != program_authority_sha256:
        raise RuntimeError("frozen geometry references a different program authority")
    if frozen_geometry_receipt.get("runtime_environment") != dict(runtime_environment):
        raise RuntimeError("frozen tangent geometry runtime environment changed")
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
        primary_exposure=sum(
            SCORE_STOP - position for position, split in zip(
                plan.scored_positions, plan.splits, strict=True,
            ) if split == "primary"
        ) * plan.probes_per_row,
        replication_exposure=sum(
            SCORE_STOP - position for position, split in zip(
                plan.scored_positions, plan.splits, strict=True,
            ) if split == "replication"
        ) * plan.probes_per_row,
    )
    passes = stage1_passes(stability)
    result = {
        "status": (
            "stable_shared_linear_encoder_candidate"
            if passes else "measured_no_stable_shared_linear_cut_knee"
        ),
        "scope": (
            "stage-1 final-output Monte-Carlo Fisher-sum tangent realization at "
            "MLP0-2 in frozen direction-coefficient gauge"
        ),
        "plan_fingerprint": plan.fingerprint,
        "program": program_receipt,
        "program_buffers": program_manifest,
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
            "rank640_predictive_sha256": tangent_authority.sha256_file(RANK640_PARENT),
            "rank640_causal_sha256": tangent_authority.sha256_file(CAUSAL_PARENT),
        },
        "provenance": {
            "protected_snapshot": protected_before,
            "program_authority_sha256": program_authority_sha256,
            "geometry_authority_sha256": geometry_authority_sha256,
            "geometry_artifact_sha256": geometry_artifact_sha256,
            "rows": {
                "authority": row_authority["authority"],
                "receipt_sha256": tangent_authority.sha256_file(frozen.AUTHORITY),
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
        "runtime_environment": dict(runtime_environment),
    }
    protected_after = tangent_authority.protected_snapshot(SOURCES)
    if protected_after != protected_before or (
        tangent_authority.program_buffer_manifest(program) != program_manifest
    ) or tangent_authority.validate_loaded_rows(rows)["tensor_raw_sha256"] != plan.row_artifact_sha256 or (
        tangent_authority.sha256_file(AUTHORITY_RECEIPT) != program_authority_sha256
    ) or tangent_authority.sha256_file(GEOMETRY_RECEIPT) != geometry_authority_sha256 or (
        tangent_authority.sha256_file(GEOMETRY_ARTIFACT) != geometry_artifact_sha256
    ):
        raise RuntimeError("protected input, program, row, or geometry changed before publication")
    tangent_authority.publish_json_create_only(
        OUTPUT, result, ownership_check=run_lock.assert_owned,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-authority", action="store_true")
    parser.add_argument("--freeze-geometry", action="store_true")
    arguments = parser.parse_args()
    runtime_environment = tangent_authority.configure_production_runtime()
    with tangent_authority.exclusive_run_lock(RUN_LOCK) as run_lock:
        if arguments.freeze_authority and arguments.freeze_geometry:
            raise RuntimeError("freeze one lifecycle stage per invocation")
        outcome = (
            freeze_program_authority(
                torch.device("cuda"), run_lock, runtime_environment,
            )
            if arguments.freeze_authority
            else freeze_geometry_authority(
                torch.device("cuda"), run_lock, runtime_environment,
            )
            if arguments.freeze_geometry else run(run_lock, runtime_environment)
        )
    if arguments.freeze_authority or arguments.freeze_geometry:
        print(json.dumps({
            "status": outcome["status"],
            "program_tree_sha256": (
                outcome["program_buffers"]["tree_sha256"]
                if arguments.freeze_authority else outcome["program_tree_sha256"]
            ),
        }, indent=2, sort_keys=True), flush=True)
        print(f"wrote {AUTHORITY_RECEIPT if arguments.freeze_authority else GEOMETRY_RECEIPT}", flush=True)
        raise SystemExit(0)
    print(json.dumps({
        "status": outcome["status"],
        "stage1_pass": outcome["stage1_pass"],
        "split_stability": outcome["split_stability"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
