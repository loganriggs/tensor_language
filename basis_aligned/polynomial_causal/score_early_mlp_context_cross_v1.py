#!/usr/bin/env python3
"""Capability-separated CPU scorer for early-MLP/context cross v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

import early_mlp_context_cross_v1_bilin18_backend as backend_module
import early_mlp_context_cross_v1_lifecycle as lifecycle
import early_mlp_context_cross_v1_measurements as measurement
import early_mlp_context_cross_v1_statistics as statistics


SCHEMA_VERSION = 1
DEFAULT_SCORE_NAMESPACE = "early_mlp_context_cross_v1_score_v1"


@dataclass(frozen=True, slots=True)
class ScorePaths:
    results: Path
    receipt: Path
    failure: Path
    lock: Path

    def require_pristine(self) -> None:
        existing = [
            str(path) for path in (self.results, self.receipt, self.failure, self.lock)
            if path.exists()
        ]
        if existing:
            raise RuntimeError(f"score namespace is already spent: {existing}")


def score_paths(
    directory: Path = lifecycle.HERE, namespace: str = DEFAULT_SCORE_NAMESPACE,
) -> ScorePaths:
    if not namespace or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in namespace
    ):
        raise ValueError("score namespace is not a safe lowercase identifier")
    root = directory.resolve()
    return ScorePaths(
        results=root / f"{namespace}_results.json",
        receipt=root / f"{namespace}_receipt.json",
        failure=root / f"{namespace}_failure.json",
        lock=root / f".{namespace}.lock",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("terminal JSON artifact is not an object")
    return value


def _load_stage(value: dict[str, Any]) -> statistics.StageStatistics:
    required = {
        "role", "stage", "authority_sha256", "ordered_document_ids_sha256",
        "document_token_count", "top1_correct", "ce_sum",
        "stage_statistics_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("stage payload schema changed")
    stage = statistics.StageStatistics(
        role=value["role"], stage=value["stage"],
        authority_sha256=value["authority_sha256"],
        ordered_document_ids_sha256=value["ordered_document_ids_sha256"],
        document_token_count=value["document_token_count"],
        top1_correct=value["top1_correct"], ce_sum=value["ce_sum"],
    )
    if stage.sha256 != value["stage_statistics_sha256"]:
        raise RuntimeError("stage payload hash differs")
    return stage


def _site(value: Any) -> tuple[str, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError("serialized site changed")
    return str(value[0]), int(value[1])


def _load_descriptor(value: dict[str, Any]) -> backend_module.ProgramDescriptor:
    copied = dict(value)
    copied["installed_compiled_sites"] = tuple(
        _site(site) for site in copied["installed_compiled_sites"]
    )
    return backend_module.ProgramDescriptor(**copied)


def _load_cell_receipt(value: dict[str, Any]) -> measurement.CellReceipt:
    copied = dict(value)
    copied["cell"] = tuple(copied["cell"])
    return measurement.CellReceipt(**copied)


def _load_call_ledger(value: dict[str, Any]) -> backend_module.CellCallLedger:
    copied = dict(value)
    for name in ("native_module_calls", "substitution_calls"):
        copied[name] = tuple(
            (_site(site), int(count)) for site, count in copied[name]
        )
    return backend_module.CellCallLedger(**copied)


def load_terminal_bundles(
    paths: lifecycle.OutputPaths, *, require_authoritative: bool = True,
) -> tuple[dict[str, measurement.StagedRoleBundle], dict[str, Any]]:
    """Open outcomes only after verifying the last-write terminal receipt."""

    if not paths.receipt.is_file() or paths.failure.exists():
        raise RuntimeError("measurement lacks a unique successful terminal receipt")
    receipt = _read_json(paths.receipt)
    authoritative = receipt.get("authoritative_measurement") is True
    expected_receipt_status = (
        "complete_two_role_measurement_receipt_last"
        if authoritative else "test_only_non_authoritative_receipt_last"
    )
    if require_authoritative and not authoritative:
        raise RuntimeError("canonical score requires an authoritative measurement")
    if receipt.get("status") != expected_receipt_status or (
        receipt.get("authorized_for_final_role") is not False
    ) or receipt.get("authority_path") != str(paths.authority.resolve()) or (
        receipt.get("payload_path") != str(paths.payload.resolve())
    ) or receipt.get("manifest_path") != str(paths.manifest.resolve()) or (
        not isinstance(receipt.get("authoritative_measurement"), bool)
    ) or lifecycle.file_sha256(paths.authority) != receipt.get(
        "authority_file_sha256"
    ) or lifecycle.file_sha256(paths.payload) != receipt.get(
        "payload_file_sha256"
    ) or lifecycle.file_sha256(paths.manifest) != receipt.get(
        "manifest_file_sha256"
    ):
        raise RuntimeError("measurement terminal receipt does not bind its predecessors")
    manifest = _read_json(paths.manifest)
    authority = _read_json(paths.authority)
    expected_manifest_status = (
        "terminal_payload_verified"
        if authoritative else "test_only_non_authoritative_manifest"
    )
    expected_authority_status = (
        "frozen_before_any_measurement_outcome"
        if authoritative else "test_only_non_authoritative_authority"
    )
    if manifest.get("status") != expected_manifest_status or manifest.get(
        "authoritative_measurement"
    ) is not authoritative or authority.get("status") != expected_authority_status or (
        authority.get("authoritative_measurement") is not authoritative
    ) or manifest.get(
        "authorized_for_final_role"
    ) is not False or manifest.get("source_closure_sha256") != receipt.get(
        "source_closure_sha256"
    ) or manifest.get("program_bank_sha256") != receipt.get(
        "program_bank_sha256"
    ) or manifest.get("authority_file_sha256") != receipt.get(
        "authority_file_sha256"
    ) or manifest.get("payload_file_sha256") != receipt.get(
        "payload_file_sha256"
    ) or manifest.get("two_role_authority_sha256") != receipt.get(
        "two_role_authority_sha256"
    ) or authority.get("two_role_authority_sha256") != receipt.get(
        "two_role_authority_sha256"
    ):
        raise RuntimeError("measurement authority/manifest/receipt disagree")
    raw_descriptors = authority.get("program_descriptors")
    if not isinstance(raw_descriptors, list) or len(raw_descriptors) != 64:
        raise RuntimeError("authority program descriptors changed")
    descriptors = tuple(_load_descriptor(value) for value in raw_descriptors)
    audit_by_role = manifest.get("cell_audit_records")
    if not isinstance(audit_by_role, dict) or set(audit_by_role) != set(
        statistics.ROLE_NAMES
    ):
        raise RuntimeError("manifest lacks the two-role cell audit records")
    raw_role_authorities = authority.get("role_authorities")
    raw_role_hashes = authority.get("role_authority_sha256s")
    if not isinstance(raw_role_authorities, dict) or not isinstance(
        raw_role_hashes, dict
    ) or set(raw_role_authorities) != set(statistics.ROLE_NAMES) or set(
        raw_role_hashes
    ) != set(statistics.ROLE_NAMES):
        raise RuntimeError("two-role authority preimages are incomplete")
    role_authorities = {
        role: measurement.RoleAuthority(**raw_role_authorities[role])
        for role in statistics.ROLE_NAMES
    }
    observed_authority_hashes = {
        role: role_authorities[role].sha256 for role in statistics.ROLE_NAMES
    }
    if observed_authority_hashes != raw_role_hashes or measurement._logical_sha256(
        observed_authority_hashes
    ) != authority["two_role_authority_sha256"]:
        raise RuntimeError("two-role authority hash chain differs")
    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "status", "authoritative_measurement",
        "two_role_authority_sha256", "roles",
    } or payload["schema_version"] != SCHEMA_VERSION or payload["status"] != (
        "complete_two_role_staged_sufficient_statistics"
        if authoritative else "test_only_non_authoritative_sufficient_statistics"
    ) or payload["authoritative_measurement"] is not authoritative or payload[
        "two_role_authority_sha256"
    ] != receipt[
        "two_role_authority_sha256"
    ] or not isinstance(payload["roles"], dict) or tuple(payload["roles"]) != (
        statistics.ROLE_NAMES
    ):
        raise RuntimeError("two-role terminal payload schema changed")
    bundles: dict[str, measurement.StagedRoleBundle] = {}
    for role in statistics.ROLE_NAMES:
        value = payload["roles"][role]
        if not isinstance(value, dict) or set(value) != {
            "role_receipt", "role_receipt_sha256", "stages",
        } or not isinstance(value["stages"], dict) or tuple(value["stages"]) != (
            "discovery", "validation", "heldout"
        ):
            raise RuntimeError("role payload schema changed")
        role_receipt_value = dict(value["role_receipt"])
        role_receipt_value["cell_receipt_sha256s"] = tuple(
            role_receipt_value["cell_receipt_sha256s"]
        )
        role_receipt_value["stage_payload_sha256s"] = tuple(
            role_receipt_value["stage_payload_sha256s"]
        )
        role_receipt = measurement.RoleReceipt(**role_receipt_value)
        role_authority = role_authorities[role]
        stages = {
            name: _load_stage(value["stages"][name])
            for name in ("discovery", "validation", "heldout")
        }
        bundle = measurement.StagedRoleBundle(
            discovery=stages["discovery"], validation=stages["validation"],
            heldout=stages["heldout"], receipt=role_receipt,
        )
        if role_receipt.sha256 != value["role_receipt_sha256"] or (
            role_receipt.sha256 != receipt["role_receipt_sha256s"][role]
        ) or role_receipt.sha256 != manifest["role_receipt_sha256s"][role]:
            raise RuntimeError("role receipt hash differs")
        if role_receipt.authority_sha256 != role_authority.sha256 or (
            role_receipt.source_commit != role_authority.source_commit
        ) or role_receipt.source_closure_sha256 != (
            role_authority.source_closure_sha256
        ) or role_receipt.row_file_sha256 != role_authority.row_file_sha256 or (
            role_receipt.row_raw_sha256 != role_authority.row_raw_sha256
        ) or role_receipt.ordered_document_ids_sha256 != (
            role_authority.ordered_document_ids_sha256
        ) or role_receipt.shared_program_sha256 != (
            role_authority.shared_program_sha256
        ) or manifest.get("stage_payload_sha256s", {}).get(role) != {
            name: stages[name].sha256
            for name in ("discovery", "validation", "heldout")
        }:
            raise RuntimeError("role authority/receipt/stage hash chain differs")
        audit_records = audit_by_role[role]
        if not isinstance(audit_records, list) or len(audit_records) != 64:
            raise RuntimeError("role cell audit record count changed")
        for ordinal, (record, descriptor) in enumerate(zip(
            audit_records, descriptors, strict=True,
        )):
            if not isinstance(record, dict) or set(record) != {
                "ordinal", "program_descriptor_sha256", "cell_receipt",
                "cell_receipt_sha256", "call_ledger", "call_ledger_sha256",
            } or record["ordinal"] != ordinal:
                raise RuntimeError("cell audit record order/schema changed")
            cell_receipt = _load_cell_receipt(record["cell_receipt"])
            call_ledger = _load_call_ledger(record["call_ledger"])
            call_ledger.validate(descriptor)
            request = measurement.REQUESTS[ordinal]
            if descriptor.sha256 != record["program_descriptor_sha256"] or (
                cell_receipt.sha256 != record["cell_receipt_sha256"]
            ) or call_ledger.sha256 != record["call_ledger_sha256"] or (
                cell_receipt.call_ledger_sha256 != call_ledger.sha256
            ) or cell_receipt.sha256 != role_receipt.cell_receipt_sha256s[ordinal] or (
                cell_receipt.authority_sha256 != role_authority.sha256
            ) or cell_receipt.request_sha256 != request.sha256 or (
                cell_receipt.source_closure_sha256 != receipt["source_closure_sha256"]
            ) or cell_receipt.model_tree_before_sha256 != (
                measurement.COMPONENT_TREE_SHA256
            ) or cell_receipt.model_tree_after_sha256 != (
                measurement.COMPONENT_TREE_SHA256
            ) or cell_receipt.shared_program_before_sha256 != (
                measurement.SHARED_PROGRAM_SHA256
            ) or cell_receipt.shared_program_after_sha256 != (
                measurement.SHARED_PROGRAM_SHA256
            ):
                raise RuntimeError("cell receipt/ledger physical binding differs")
        bundles[role] = bundle
    return bundles, receipt


def score_transaction(
    *, measurement_paths: lifecycle.OutputPaths | None = None,
    paths: ScorePaths | None = None,
) -> dict[str, Any]:
    measurement_paths = (
        lifecycle.output_paths() if measurement_paths is None else measurement_paths
    )
    paths = score_paths() if paths is None else paths
    canonical_score = all(
        left.resolve() == right.resolve()
        for left, right in zip(
            (paths.results, paths.receipt, paths.failure, paths.lock),
            (
                score_paths().results, score_paths().receipt,
                score_paths().failure, score_paths().lock,
            ),
            strict=True,
        )
    )
    canonical_measurement = lifecycle.output_paths()
    if canonical_score and any(
        left.resolve() != right.resolve()
        for left, right in zip(
            measurement_paths.all_paths(), canonical_measurement.all_paths(), strict=True,
        )
    ):
        raise RuntimeError("canonical score requires the canonical measurement namespace")
    paths.require_pristine()
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    existing = [
        str(path) for path in (paths.results, paths.receipt, paths.failure)
        if path.exists()
    ]
    if existing:
        lock.release()
        raise RuntimeError(f"score namespace raced after lock acquisition: {existing}")
    phase = "verify_terminal_measurement"
    try:
        source = lifecycle.committed_source_closure()
        bundles, measurement_receipt = load_terminal_bundles(
            measurement_paths, require_authoritative=canonical_score,
        )
        if source.sha256 != measurement_receipt["source_closure_sha256"]:
            raise RuntimeError("scorer source differs from measurement authority")
        phase = "score_capability_separated_roles"
        scores: dict[str, dict[str, Any]] = {}
        for role in statistics.ROLE_NAMES:
            bundle = bundles[role]
            rank3 = statistics.score_rank(
                bundle.discovery, bundle.validation, None, rank=3,
            )
            rank4 = statistics.score_rank(
                bundle.discovery, bundle.validation, bundle.heldout, rank=4,
            )
            scores[role] = {"rank3": rank3, "rank4": rank4}
        rank3_pass = all(
            scores[role]["rank3"]["ce_useful_pass"]
            for role in statistics.ROLE_NAMES
        )
        rank4_pass = all(
            scores[role]["rank4"]["ce_useful_pass"]
            for role in statistics.ROLE_NAMES
        )
        # Frozen selection rule: a passing rank three is the minimal model and the
        # later rank-four score cannot replace it.  Rank four is selected only when
        # rank three fails and rank four passes.
        selected = 3 if rank3_pass else (4 if rank4_pass else None)
        results = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_capability_separated_two_role_score"
                if canonical_score else "test_only_non_authoritative_score"
            ),
            "authoritative_score": canonical_score,
            "source_commit": source.source_commit,
            "source_closure_sha256": source.sha256,
            "measurement_receipt_file_sha256": lifecycle.file_sha256(
                measurement_paths.receipt
            ),
            "measurement_two_role_authority_sha256": measurement_receipt[
                "two_role_authority_sha256"
            ],
            "roles": scores,
            "two_role_ce_rank3_pass": rank3_pass,
            "two_role_ce_rank4_pass": rank4_pass,
            "ce_any_registered_pass": rank3_pass or rank4_pass,
            "selected_minimal_rank": selected,
            "top1_broad_behavior_pass": None,
        }
        phase = "publish_results"
        lifecycle.publish_json_create_only(paths.results, results, lock)
        results_sha256 = lifecycle.file_sha256(paths.results)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_score_receipt_last"
                if canonical_score else "test_only_non_authoritative_score_receipt_last"
            ),
            "authoritative_score": canonical_score,
            "source_closure_sha256": source.sha256,
            "measurement_receipt_file_sha256": results[
                "measurement_receipt_file_sha256"
            ],
            "results_file_sha256": results_sha256,
            "selected_minimal_rank": selected,
            "ce_any_registered_pass": rank3_pass or rank4_pass,
            "top1_broad_behavior_pass": None,
        }
        phase = "publish_receipt_last"
        lock.require_owned()
        lifecycle.verify_source_closure(source)
        if lifecycle.file_sha256(measurement_paths.receipt) != results[
            "measurement_receipt_file_sha256"
        ] or lifecycle.file_sha256(paths.results) != results_sha256:
            raise RuntimeError("score predecessor changed before receipt publication")
        lifecycle.publish_json_create_only(paths.receipt, receipt, lock)
        return receipt
    except Exception as error:
        if not paths.failure.exists():
            failure = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed_no_scientific_interpretation",
                "phase": phase,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
            try:
                lifecycle.publish_json_create_only(paths.failure, failure, lock)
            except Exception:
                pass
        raise
    finally:
        if lock.inode is not None:
            lock.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    receipt = score_transaction()
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
