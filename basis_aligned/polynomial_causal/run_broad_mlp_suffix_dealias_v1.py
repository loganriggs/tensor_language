#!/usr/bin/env python3
"""Receipt-last two-role transaction for broad-MLP suffix de-alias v1."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import secrets
from typing import Any, Mapping, Protocol

import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_bilin18_backend as backend_module
import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle
import broad_mlp_suffix_dealias_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as parent_statistics


SCHEMA_VERSION = 1
ROLE_ORDER = assay.ROLE_NAMES


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
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _tree_equal(observed: Any, expected: Any) -> bool:
    if torch.is_tensor(expected):
        return torch.is_tensor(observed) and torch.equal(observed, expected)
    if isinstance(expected, dict):
        return isinstance(observed, dict) and set(observed) == set(expected) and all(
            _tree_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        return type(observed) is type(expected) and len(observed) == len(expected) and all(
            _tree_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return type(observed) is type(expected) and observed == expected


def _validate_payload(path: Path, expected: dict[str, Any]) -> None:
    if not _tree_equal(torch.load(path, map_location="cpu", weights_only=True), expected):
        raise RuntimeError("installed broad-MLP payload changed")


def _verify_backend_surface(backend: MeasurementBackend, *, require_production: bool) -> None:
    if type(backend.batch_size) is not int or backend.batch_size <= 0 or (
        backend.source_paths != backend_module.SOURCE_PATHS
    ) or not all(callable(getattr(backend, name, None)) for name in (
        "prepare", "verify_pre_outcome", "execute_cell", "close",
    )):
        raise RuntimeError("broad-MLP backend surface is malformed")
    if require_production and (
        type(backend) is not backend_module.Bilin18BroadMLPSuffixBackend
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
    *, role_rows, source, bank: backend_module.PreparedBank,
    parent_authority: dict[str, Any], batch_count: int, nonce: str,
) -> measurement.RoleAuthority:
    role = role_rows.role
    wave = role_rows.wave
    row_to_document, token_count = wave.clone_mapping_and_counts()
    parent_role = parent_authority["role_authorities"][role]
    parent_binding = parent_authority["row_bindings"][role]
    return measurement.RoleAuthority(
        role=role,
        source_commit=source.source_commit,
        source_closure_sha256=source.sha256,
        row_file_sha256=role_rows.row_file_sha256,
        row_raw_sha256=lifecycle.parent.FROZEN_RAW_ROW_SHA256[role],
        ordered_row_identity_sha256=wave.ordered_row_identity_sha256,
        ordered_row_to_document_sha256=parent_statistics.tensor_sha256(row_to_document),
        ordered_document_ids_sha256=wave.ordered_document_ids_sha256,
        row_token_count_sha256=parent_statistics.tensor_sha256(token_count),
        common_support_sha256=parent_binding["common_support_sha256"],
        parent_role_authority_sha256=measurement._logical_sha256(parent_role),
        parent_measurement_receipt_sha256=lifecycle.file_sha256(
            lifecycle.PARENT_PATHS.receipt
        ),
        wave_nonce_sha256=measurement._logical_sha256({
            "nonce": nonce,
            "role": role,
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
    *, source, roles, bank: backend_module.PreparedBank,
    authorities: Mapping[str, measurement.RoleAuthority],
    parent_authority: dict[str, Any], authoritative: bool,
) -> dict[str, Any]:
    hashes = {role: authorities[role].sha256 for role in ROLE_ORDER}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_any_measurement_outcome" if authoritative else (
            "test_only_non_authoritative_authority"
        ),
        "authoritative_measurement": authoritative,
        "authorized_for_final_role": False,
        "source_closure": asdict(source),
        "source_closure_sha256": source.sha256,
        "role_authorities": {role: asdict(authorities[role]) for role in ROLE_ORDER},
        "role_authority_sha256s": hashes,
        "two_role_authority_sha256": measurement._logical_sha256(hashes),
        "parent_two_role_authority_sha256": parent_authority[
            "two_role_authority_sha256"
        ],
        "row_bindings": {
            role: getattr(roles, role).wave.receipt() for role in ROLE_ORDER
        },
        "model_binding": asdict(bank.model),
        "shared_program_sha256": bank.shared_program_sha256,
        "program_descriptors": [asdict(value) for value in bank.programs],
        "program_bank_sha256": bank.sha256,
        "request_plan_sha256": measurement.REQUEST_PLAN_SHA256,
        "role_order": list(ROLE_ORDER),
        "cell_order": list(range(assay.CELL_COUNT)),
    }


def run_transaction(
    *, backend: MeasurementBackend, paths: lifecycle.OutputPaths | None = None,
) -> dict[str, Any]:
    """Publish authority, run sixteen role-cells, then publish receipt last."""

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
    phase, role, ordinal = "verify_source_and_rows", None, None
    backend_closed = False
    try:
        if any(path.exists() for path in paths.all_paths() if path != paths.lock):
            raise RuntimeError("broad-MLP namespace raced after lock acquisition")
        lifecycle.verify_protected_files()
        parent_authority = lifecycle.parent_authority()
        source = lifecycle.committed_source_closure()
        roles = lifecycle.load_two_roles()
        role_rows = {
            name: getattr(roles, name).wave.clone_rows() for name in ROLE_ORDER
        }
        phase = "prepare_shared_program"
        bank = backend.prepare(role_rows, measurement.REQUESTS)
        if bank.model.model_realization_sha256 != measurement.MODEL_REALIZATION_SHA256 or (
            bank.model.component_tree_sha256 != measurement.COMPONENT_TREE_SHA256
        ) or bank.shared_program_sha256 != measurement.SHARED_PROGRAM_SHA256:
            raise RuntimeError("backend prepared an unauthorized realization")
        batch_count = (measurement.ROW_COUNT + backend.batch_size - 1) // backend.batch_size
        if batch_count != measurement.BATCH_COUNT:
            raise RuntimeError("backend batch count differs from authority")
        nonce = secrets.token_hex(32)
        authorities = {
            name: _role_authority(
                role_rows=getattr(roles, name), source=source, bank=bank,
                parent_authority=parent_authority, batch_count=batch_count, nonce=nonce,
            ) for name in ROLE_ORDER
        }
        authority_value = _authority_payload(
            source=source, roles=roles, bank=bank, authorities=authorities,
            parent_authority=parent_authority, authoritative=canonical_publication,
        )
        authority_bytes = _json_bytes(authority_value)
        phase = "publish_pre_outcome_authority"
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        component, program = backend.verify_pre_outcome(bank)
        if component != measurement.COMPONENT_TREE_SHA256 or program != (
            measurement.SHARED_PROGRAM_SHA256
        ):
            raise RuntimeError("pre-outcome realization changed")
        if any(path.exists() for path in paths.all_paths() if path != paths.lock):
            raise RuntimeError("output appeared immediately before authority publication")
        lifecycle.publish_json_create_only(paths.authority, authority_value, lock)
        if paths.authority.read_bytes() != authority_bytes:
            raise RuntimeError("published authority bytes do not replay")

        collectors, audits = {}, {name: [] for name in ROLE_ORDER}
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
                result = backend.execute_cell(role, request, role_rows[role].clone(), descriptor)
                result.call_ledger.validate(descriptor)
                receipt = measurement.CellReceipt(
                    authority_sha256=authorities[role].sha256,
                    request_sha256=request.sha256,
                    ordinal=ordinal,
                    prefix_index=request.prefix_index,
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
                collectors[role].add(result.statistics, receipt)
                audits[role].append({
                    "ordinal": ordinal,
                    "program_descriptor_sha256": descriptor.sha256,
                    "cell_receipt": asdict(receipt),
                    "cell_receipt_sha256": receipt.sha256,
                    "call_ledger": asdict(result.call_ledger),
                    "call_ledger_sha256": result.call_ledger.sha256,
                })
            # The implementation amendment requires a full immutable-input
            # checkpoint after each complete role, not only after both roles.
            lifecycle.verify_protected_files()
            lifecycle.verify_source_closure(source)

        phase = "close_and_reverify"
        closed_tree = backend.close()
        backend_closed = True
        if closed_tree != measurement.COMPONENT_TREE_SHA256:
            raise RuntimeError("backend close did not preserve model tree")
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        bundles = {name: collectors[name].finalize() for name in ROLE_ORDER}
        payload_value = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_two_role_sufficient_statistics" if canonical_publication else (
                "test_only_non_authoritative_sufficient_statistics"
            ),
            "authoritative_measurement": canonical_publication,
            "two_role_authority_sha256": authority_value["two_role_authority_sha256"],
            "roles": {
                name: {
                    "role_receipt": asdict(bundles[name].receipt),
                    "role_receipt_sha256": bundles[name].receipt.sha256,
                    "statistics": {
                        "role": bundles[name].statistics.role,
                        "authority_sha256": bundles[name].statistics.authority_sha256,
                        "ordered_document_ids_sha256": bundles[name].statistics.ordered_document_ids_sha256,
                        "document_token_count": bundles[name].statistics.document_token_count,
                        "top1_correct": bundles[name].statistics.top1_correct,
                        "ce_sum": bundles[name].statistics.ce_sum,
                        "statistics_sha256": bundles[name].statistics.sha256,
                    },
                } for name in ROLE_ORDER
            },
        }
        phase = "publish_terminal_payload"
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        lifecycle.publish_torch_create_only(paths.payload, payload_value, lock)
        _validate_payload(paths.payload, payload_value)
        payload_sha = lifecycle.file_sha256(paths.payload)
        manifest_value = {
            "schema_version": SCHEMA_VERSION,
            "status": "terminal_payload_verified" if canonical_publication else (
                "test_only_non_authoritative_manifest"
            ),
            "authoritative_measurement": canonical_publication,
            "authorized_for_final_role": False,
            "source_commit": source.source_commit,
            "source_closure_sha256": source.sha256,
            "authority_file_sha256": lifecycle.file_sha256(paths.authority),
            "payload_file_sha256": payload_sha,
            "two_role_authority_sha256": authority_value["two_role_authority_sha256"],
            "program_bank_sha256": bank.sha256,
            "role_receipt_sha256s": {
                name: bundles[name].receipt.sha256 for name in ROLE_ORDER
            },
            "cell_audit_records": audits,
            "role_order": list(ROLE_ORDER),
            "cell_order": list(range(assay.CELL_COUNT)),
        }
        phase = "publish_manifest"
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        lifecycle.publish_json_create_only(paths.manifest, manifest_value, lock)
        manifest_bytes = _json_bytes(manifest_value)
        manifest_sha = lifecycle.file_sha256(paths.manifest)
        if paths.manifest.read_bytes() != manifest_bytes:
            raise RuntimeError("published manifest bytes do not replay")
        receipt_value = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_two_role_measurement_receipt_last" if canonical_publication else (
                "test_only_non_authoritative_receipt_last"
            ),
            "authoritative_measurement": canonical_publication,
            "authorized_for_final_role": False,
            "authority_file_sha256": lifecycle.file_sha256(paths.authority),
            "payload_file_sha256": payload_sha,
            "manifest_file_sha256": manifest_sha,
            "two_role_authority_sha256": authority_value["two_role_authority_sha256"],
            "source_closure_sha256": source.sha256,
            "program_bank_sha256": bank.sha256,
            "role_receipt_sha256s": manifest_value["role_receipt_sha256s"],
        }
        phase = "publish_receipt_last"
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        _validate_payload(paths.payload, payload_value)
        if paths.authority.read_bytes() != authority_bytes or (
            lifecycle.file_sha256(paths.payload) != payload_sha
        ) or paths.manifest.read_bytes() != manifest_bytes or (
            lifecycle.file_sha256(paths.manifest) != manifest_sha
        ):
            raise RuntimeError("a predecessor changed before receipt publication")
        lifecycle.publish_json_create_only(paths.receipt, receipt_value, lock)
        return receipt_value
    except Exception as error:
        if not paths.failure.exists():
            try:
                lifecycle.publish_json_create_only(paths.failure, {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed_no_scientific_interpretation",
                    "authorized_for_final_role": False,
                    "phase": phase,
                    "role": role,
                    "ordinal": ordinal,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }, lock)
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
    print(json.dumps(
        run_transaction(backend=backend_module.create_backend()),
        sort_keys=True, indent=2,
    ))


if __name__ == "__main__":
    main()
