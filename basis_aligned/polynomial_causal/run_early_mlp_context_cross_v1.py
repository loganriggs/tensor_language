#!/usr/bin/env python3
"""Receipt-last two-role transaction for early-MLP/context cross v1.

This file owns transaction ordering only.  The backend owns model execution, the
measurement module owns ordered aggregation, and the statistics module owns staged
scientific access.  No partial outcome is published.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import secrets
from typing import Any, Mapping, Protocol

import torch

import compilation_mask_cut_rank_v1_gpu_adapter as inherited
import early_mlp_context_cross_v1_bilin18_backend as backend_module
import early_mlp_context_cross_v1_lifecycle as lifecycle
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics


SCHEMA_VERSION = 1
ROLE_ORDER = statistics.ROLE_NAMES


class MeasurementBackend(Protocol):
    batch_size: int
    source_paths: tuple[str, ...]

    def prepare(
        self, role_rows: Mapping[str, torch.Tensor],
        requests: tuple[measurement.MeasurementRequest, ...],
    ) -> backend_module.PreparedBank: ...

    def execute_cell(
        self, role: str, request: measurement.MeasurementRequest,
        rows: torch.Tensor, descriptor: backend_module.ProgramDescriptor,
    ) -> backend_module.BackendCellResult: ...

    def verify_pre_outcome(
        self, bank: backend_module.PreparedBank,
    ) -> tuple[str, str]: ...

    def close(self) -> str: ...


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


def _stage_payload(stage: statistics.StageStatistics) -> dict[str, Any]:
    return {
        "role": stage.role,
        "stage": stage.stage,
        "authority_sha256": stage.authority_sha256,
        "ordered_document_ids_sha256": stage.ordered_document_ids_sha256,
        "document_token_count": stage.document_token_count,
        "top1_correct": stage.top1_correct,
        "ce_sum": stage.ce_sum,
        "stage_statistics_sha256": stage.sha256,
    }


def _tree_equal(observed: Any, expected: Any) -> bool:
    if torch.is_tensor(expected):
        return torch.is_tensor(observed) and torch.equal(observed, expected)
    if isinstance(expected, dict):
        return isinstance(observed, dict) and set(observed) == set(expected) and all(
            _tree_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        return type(observed) is type(expected) and len(observed) == len(expected) and all(
            _tree_equal(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return type(observed) is type(expected) and observed == expected


def _validate_payload(path: Path, expected: dict[str, Any]) -> None:
    observed = torch.load(path, map_location="cpu", weights_only=True)
    if not _tree_equal(observed, expected):
        raise RuntimeError("installed two-role payload changed")


def _verify_backend_surface(
    backend: MeasurementBackend, *, require_production: bool,
) -> None:
    if type(backend.batch_size) is not int or backend.batch_size <= 0 or not isinstance(
        backend.source_paths, tuple
    ) or not backend.source_paths or len(backend.source_paths) != len(
        set(backend.source_paths)
    ) or any(not isinstance(path, str) or not path for path in backend.source_paths) or not all(
        callable(getattr(backend, name, None))
        for name in ("prepare", "verify_pre_outcome", "execute_cell", "close")
    ) or backend.source_paths != backend_module.SOURCE_PATHS:
        raise RuntimeError("two-role backend surface is malformed")
    if require_production and (
        type(backend) is not backend_module.Bilin18ContextCrossBackend
        or backend.dimensions != backend_module.parent.PRODUCTION_DIMENSIONS
        or backend.device != torch.device("cuda")
        or backend._model_loader is not None
        or backend._fit_wave_loader is not None
        or backend._program_builder is not None
        or backend.expected_shared_program_sha256 != measurement.SHARED_PROGRAM_SHA256
        or backend._closed
    ):
        raise RuntimeError("canonical publication requires the exact production backend")


def _role_authority(
    *, role_rows: lifecycle.RoleRows, source: inherited.SourceClosure,
    bank: backend_module.PreparedBank, disjointness_sha256: str,
    batch_count: int, nonce: str,
) -> measurement.RoleAuthority:
    wave = role_rows.wave
    row_to_document, token_count = wave.clone_mapping_and_counts()
    return measurement.RoleAuthority(
        role=role_rows.role,
        source_commit=source.source_commit,
        source_closure_sha256=source.sha256,
        source_receipt_file_sha256=lifecycle.FROZEN_FILE_SHA256[
            lifecycle.ROW_RECEIPT
        ],
        row_file_sha256=role_rows.row_file_sha256,
        row_raw_sha256=lifecycle.FROZEN_RAW_ROW_SHA256[role_rows.role],
        row_provenance_sha256=wave.row_provenance_sha256,
        ordered_row_identity_sha256=wave.ordered_row_identity_sha256,
        ordered_row_to_document_sha256=statistics.tensor_sha256(row_to_document),
        ordered_document_ids_sha256=wave.ordered_document_ids_sha256,
        row_token_count_sha256=statistics.tensor_sha256(token_count),
        document_identity_set_sha256=role_rows.document_identity_set_sha256,
        cross_role_disjointness_sha256=disjointness_sha256,
        wave_nonce_sha256=measurement._logical_sha256({
            "nonce": nonce, "role": role_rows.role,
            "row_wave_sha256": wave.sha256,
            "program_bank_sha256": bank.sha256,
            "source_closure_sha256": source.sha256,
        }),
        row_count=wave.row_count,
        document_count=wave.document_count,
        total_scored_token_count=int(token_count.sum()),
        batch_count=batch_count,
        model_realization_sha256=bank.model.model_realization_sha256,
        component_tree_sha256=bank.model.component_tree_sha256,
        shared_program_sha256=bank.shared_program_sha256,
    )


def _authority_payload(
    *, source: inherited.SourceClosure, roles: lifecycle.TwoRoleRows,
    bank: backend_module.PreparedBank,
    authorities: Mapping[str, measurement.RoleAuthority],
    authoritative_measurement: bool,
) -> dict[str, Any]:
    row_bindings = {}
    for role in ROLE_ORDER:
        role_rows = getattr(roles, role)
        row_bindings[role] = role_rows.wave.receipt()
    authority_hashes = {
        role: authorities[role].sha256 for role in ROLE_ORDER
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "frozen_before_any_measurement_outcome"
            if authoritative_measurement else "test_only_non_authoritative_authority"
        ),
        "authoritative_measurement": authoritative_measurement,
        "authorized_for_final_role": False,
        "source_closure": asdict(source),
        "source_closure_sha256": source.sha256,
        "role_authorities": {
            role: asdict(authorities[role]) for role in ROLE_ORDER
        },
        "role_authority_sha256s": authority_hashes,
        "two_role_authority_sha256": measurement._logical_sha256(authority_hashes),
        "row_bindings": row_bindings,
        "cross_role_disjointness_sha256": roles.disjointness_sha256,
        "model_binding": asdict(bank.model),
        "shared_program_sha256": bank.shared_program_sha256,
        "program_descriptors": [asdict(value) for value in bank.programs],
        "program_bank_sha256": bank.sha256,
        "request_plan_sha256": measurement.REQUEST_PLAN_SHA256,
        "role_order": list(ROLE_ORDER),
        "cell_order": list(range(64)),
    }


def run_transaction(
    *, backend: MeasurementBackend,
    paths: lifecycle.OutputPaths | None = None,
) -> dict[str, Any]:
    """Execute authority -> 128 cells -> close -> payload -> manifest -> receipt."""

    paths = lifecycle.output_paths() if paths is None else paths
    canonical = lifecycle.output_paths()
    canonical_publication = all(
        left.resolve() == right.resolve()
        for left, right in zip(paths.all_paths(), canonical.all_paths(), strict=True)
    )
    _verify_backend_surface(backend, require_production=canonical_publication)
    paths.require_pristine()
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    existing = [
        str(path) for path in (
            paths.authority, paths.payload, paths.manifest, paths.receipt,
            paths.failure,
        ) if path.exists()
    ]
    if existing:
        lock.release()
        raise RuntimeError(f"cross namespace raced after lock acquisition: {existing}")
    phase = "verify_source_and_rows"
    role: str | None = None
    ordinal: int | None = None
    backend_closed = False
    try:
        source = lifecycle.committed_source_closure()
        roles = lifecycle.load_two_roles()
        role_rows = {
            name: getattr(roles, name).wave.clone_rows() for name in ROLE_ORDER
        }
        phase = "prepare_shared_program"
        bank = backend.prepare(role_rows, measurement.REQUESTS)
        if not isinstance(bank, backend_module.PreparedBank) or (
            bank.model.model_realization_sha256 != measurement.MODEL_REALIZATION_SHA256
        ) or bank.model.component_tree_sha256 != measurement.COMPONENT_TREE_SHA256 or (
            bank.shared_program_sha256 != measurement.SHARED_PROGRAM_SHA256
        ):
            raise RuntimeError("backend prepared an unauthorized realization")
        if dict(bank.evaluation_role_row_sha256s) != {
            name: statistics.tensor_sha256(rows) for name, rows in role_rows.items()
        }:
            raise RuntimeError("prepared bank role support changed")
        batch_count = (measurement.ROW_COUNT + backend.batch_size - 1) // backend.batch_size
        if batch_count != measurement.BATCH_COUNT:
            raise RuntimeError("backend batch count differs from authority")
        nonce = secrets.token_hex(32)
        authorities = {
            name: _role_authority(
                role_rows=getattr(roles, name), source=source, bank=bank,
                disjointness_sha256=roles.disjointness_sha256,
                batch_count=batch_count, nonce=nonce,
            )
            for name in ROLE_ORDER
        }
        authority_value = _authority_payload(
            source=source, roles=roles, bank=bank, authorities=authorities,
            authoritative_measurement=canonical_publication,
        )
        authority_bytes = _json_bytes(authority_value)
        phase = "publish_pre_outcome_authority"
        lock.require_owned()
        raced = [
            str(path) for path in (
                paths.authority, paths.payload, paths.manifest, paths.receipt,
                paths.failure,
            ) if path.exists()
        ]
        if raced:
            raise RuntimeError(
                f"output appeared during pre-outcome preparation: {raced}"
            )
        lifecycle.verify_inherited_files()
        lifecycle.verify_source_closure(source)
        for name in ROLE_ORDER:
            if getattr(roles, name).wave.sha256 != authority_value[
                "row_bindings"
            ][name]["row_wave_sha256"]:
                raise RuntimeError("role wave changed before authority publication")
        component, program = backend.verify_pre_outcome(bank)
        if component != measurement.COMPONENT_TREE_SHA256 or program != (
            measurement.SHARED_PROGRAM_SHA256
        ):
            raise RuntimeError("pre-outcome model/program revalidation changed")
        # The checks above are intentionally outcome-blind but can be slow.  Close
        # the race again at the last possible instant before the authority link.
        lock.require_owned()
        raced = [
            str(path) for path in (
                paths.authority, paths.payload, paths.manifest, paths.receipt,
                paths.failure,
            ) if path.exists()
        ]
        if raced:
            raise RuntimeError(
                f"output appeared immediately before authority publication: {raced}"
            )
        lifecycle.publish_json_create_only(paths.authority, authority_value, lock)
        if paths.authority.read_bytes() != authority_bytes:
            raise RuntimeError("published authority bytes do not replay")

        collectors: dict[str, measurement.RoleCollector] = {}
        cell_audit_records: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ROLE_ORDER
        }
        for name in ROLE_ORDER:
            mapping, counts = getattr(roles, name).wave.clone_mapping_and_counts()
            collectors[name] = measurement.RoleCollector(
                authority=authorities[name], row_to_document=mapping,
                row_token_count=counts,
            )
        phase = "measure_both_roles"
        for role in ROLE_ORDER:
            for request, descriptor in zip(
                measurement.REQUESTS, bank.programs, strict=True,
            ):
                ordinal = request.ordinal
                result = backend.execute_cell(
                    role, request, role_rows[role].clone(), descriptor,
                )
                if not isinstance(result, backend_module.BackendCellResult):
                    raise RuntimeError("backend returned an untyped cell result")
                result.call_ledger.validate(descriptor)
                receipt = measurement.CellReceipt(
                    authority_sha256=authorities[role].sha256,
                    request_sha256=request.sha256,
                    ordinal=request.ordinal,
                    cell=request.cell,
                    statistics_sha256=result.statistics.sha256,
                    top1_correct_sha256=result.statistics.top1_correct_sha256,
                    ce_sum_sha256=result.statistics.ce_sum_sha256,
                    row_token_count_sha256=authorities[role].row_token_count_sha256,
                    call_ledger_sha256=result.call_ledger.sha256,
                    source_closure_sha256=source.sha256,
                    model_tree_before_sha256=result.component_tree_before_sha256,
                    model_tree_after_sha256=result.component_tree_after_sha256,
                    shared_program_before_sha256=result.shared_program_before_sha256,
                    shared_program_after_sha256=result.shared_program_after_sha256,
                    outer_forward_count=result.call_ledger.outer_forward_count,
                    batch_count=result.call_ledger.batch_count,
                )
                collectors[role].add_cell(
                    request=request, values=result.statistics, receipt=receipt,
                )
                cell_audit_records[role].append({
                    "ordinal": request.ordinal,
                    "program_descriptor_sha256": descriptor.sha256,
                    "cell_receipt": asdict(receipt),
                    "cell_receipt_sha256": receipt.sha256,
                    "call_ledger": asdict(result.call_ledger),
                    "call_ledger_sha256": result.call_ledger.sha256,
                })

        phase = "close_and_reverify"
        closed_tree = backend.close()
        backend_closed = True
        if closed_tree != measurement.COMPONENT_TREE_SHA256:
            raise RuntimeError("backend close did not preserve the model tree")
        lifecycle.verify_inherited_files()
        lifecycle.verify_source_closure(source)
        for name in ROLE_ORDER:
            if getattr(roles, name).wave.sha256 != authority_value[
                "row_bindings"
            ][name]["row_wave_sha256"]:
                raise RuntimeError("role wave changed during measurement")
        bundles = {name: collectors[name].finalize() for name in ROLE_ORDER}
        payload_value = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_two_role_staged_sufficient_statistics"
                if canonical_publication
                else "test_only_non_authoritative_sufficient_statistics"
            ),
            "authoritative_measurement": canonical_publication,
            "two_role_authority_sha256": authority_value[
                "two_role_authority_sha256"
            ],
            "roles": {
                name: {
                    "role_receipt": asdict(bundles[name].receipt),
                    "role_receipt_sha256": bundles[name].receipt.sha256,
                    "stages": {
                        stage: _stage_payload(getattr(bundles[name], stage))
                        for stage in ("discovery", "validation", "heldout")
                    },
                }
                for name in ROLE_ORDER
            },
        }
        phase = "publish_terminal_payload"
        lifecycle.publish_torch_create_only(paths.payload, payload_value, lock)
        _validate_payload(paths.payload, payload_value)
        payload_file_sha256 = lifecycle.file_sha256(paths.payload)
        manifest_value = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "terminal_payload_verified"
                if canonical_publication else "test_only_non_authoritative_manifest"
            ),
            "authoritative_measurement": canonical_publication,
            "authorized_for_final_role": False,
            "source_commit": source.source_commit,
            "source_closure_sha256": source.sha256,
            "authority_file_sha256": lifecycle.file_sha256(paths.authority),
            "payload_file_sha256": payload_file_sha256,
            "two_role_authority_sha256": authority_value[
                "two_role_authority_sha256"
            ],
            "program_bank_sha256": bank.sha256,
            "role_receipt_sha256s": {
                name: bundles[name].receipt.sha256 for name in ROLE_ORDER
            },
            "stage_payload_sha256s": {
                name: {
                    stage: getattr(bundles[name], stage).sha256
                    for stage in ("discovery", "validation", "heldout")
                }
                for name in ROLE_ORDER
            },
            # These are tensor-free preimages, not opaque hashes.  They make the
            # physical native/substitution census independently replayable after
            # the in-memory backend and hooks are gone.
            "cell_audit_records": cell_audit_records,
            "role_order": list(ROLE_ORDER),
            "cell_order": list(range(64)),
        }
        phase = "publish_manifest"
        lifecycle.publish_json_create_only(paths.manifest, manifest_value, lock)
        manifest_bytes = _json_bytes(manifest_value)
        if paths.manifest.read_bytes() != manifest_bytes:
            raise RuntimeError("published manifest bytes do not replay")
        manifest_file_sha256 = lifecycle.file_sha256(paths.manifest)

        receipt_value = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_two_role_measurement_receipt_last"
                if canonical_publication
                else "test_only_non_authoritative_receipt_last"
            ),
            "authoritative_measurement": canonical_publication,
            "authorized_for_final_role": False,
            "authority_path": str(paths.authority.resolve()),
            "authority_file_sha256": lifecycle.file_sha256(paths.authority),
            "payload_path": str(paths.payload.resolve()),
            "payload_file_sha256": payload_file_sha256,
            "manifest_path": str(paths.manifest.resolve()),
            "manifest_file_sha256": manifest_file_sha256,
            "two_role_authority_sha256": authority_value[
                "two_role_authority_sha256"
            ],
            "source_closure_sha256": source.sha256,
            "program_bank_sha256": bank.sha256,
            "role_receipt_sha256s": manifest_value["role_receipt_sha256s"],
        }
        phase = "publish_receipt_last"
        lock.require_owned()
        lifecycle.verify_inherited_files()
        lifecycle.verify_source_closure(source)
        _validate_payload(paths.payload, payload_value)
        if paths.authority.read_bytes() != authority_bytes or (
            lifecycle.file_sha256(paths.payload) != payload_file_sha256
        ) or paths.manifest.read_bytes() != manifest_bytes or (
            lifecycle.file_sha256(paths.manifest) != manifest_file_sha256
        ):
            raise RuntimeError("a predecessor changed before receipt publication")
        lifecycle.publish_json_create_only(paths.receipt, receipt_value, lock)
        receipt_bytes = _json_bytes(receipt_value)
        if paths.receipt.read_bytes() != receipt_bytes:
            raise RuntimeError("published receipt bytes do not replay")
        return receipt_value
    except Exception as error:
        if not paths.failure.exists():
            failure_value = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed_no_scientific_interpretation",
                "authorized_for_final_role": False,
                "phase": phase,
                "role": role,
                "ordinal": ordinal,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "authority_file_sha256": (
                    lifecycle.file_sha256(paths.authority)
                    if paths.authority.exists() else None
                ),
            }
            try:
                lifecycle.publish_json_create_only(paths.failure, failure_value, lock)
            except Exception:
                pass
        raise
    finally:
        try:
            if not backend_closed:
                backend.close()
        finally:
            if lock.inode is not None:
                lock.release()


def main() -> None:
    receipt = run_transaction(backend=backend_module.create_backend())
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
