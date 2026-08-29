#!/usr/bin/env python3
"""Receipt-last CPU scorer for broad-MLP suffix de-alias v1.

The scorer opens the eight new M-only cells only after their measurement receipt is
terminal.  It separately replays the protected parent E/A/AM transaction, joins the
two transactions at document sufficient-statistic granularity, and delegates all
scientific arithmetic to :mod:`broad_mlp_suffix_dealias_v1`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle
import broad_mlp_suffix_dealias_v1_measurements as measurement
import compilation_mask_cut_rank_v1_gpu_adapter as inherited
import early_mlp_context_cross_v1 as parent_registry
import early_mlp_context_cross_v1_measurements as parent_measurement
import early_mlp_context_cross_v1_statistics as parent_statistics
import score_early_mlp_context_cross_v1 as parent_scorer


SCHEMA_VERSION = 1
DEFAULT_SCORE_NAMESPACE = "broad_mlp_suffix_dealias_v1_score_v1"
CLAIM_BOUNDARY = (
    "same-corpus new-mask finite-replacement composition evidence only; no OOD, "
    "semantic-circuit, zero-native-call, executable-compression, or whole-model "
    "explained-fraction credit"
)
OLD_SUFFIX_COLUMNS = {"e": 0, "a": 4, "am": 5}


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
            raise RuntimeError(f"broad-MLP score namespace is already spent: {existing}")


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


def _site(value: Any) -> tuple[str, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or type(
        value[0]
    ) is not str or type(value[1]) is not int:
        raise RuntimeError("serialized physical site changed")
    return value[0], value[1]


def _load_source_closure(value: Any) -> inherited.SourceClosure:
    if not isinstance(value, dict) or set(value) != {
        "source_commit", "path_sha256s",
    } or not isinstance(value["path_sha256s"], (tuple, list)):
        raise RuntimeError("measurement source closure schema changed")
    source = inherited.SourceClosure(
        source_commit=value["source_commit"],
        path_sha256s=tuple(tuple(item) for item in value["path_sha256s"]),
    )
    return source


def _descriptor_sha256(value: Any, ordinal: int) -> str:
    expected_keys = {
        "ordinal", "request_sha256", "installed_compiled_sites",
        "shared_program_sha256", "execution_mode", "gain_policy",
    }
    if not isinstance(value, dict) or set(value) != expected_keys or value[
        "ordinal"
    ] != ordinal or value["request_sha256"] != measurement.REQUESTS[ordinal].sha256 or (
        value["shared_program_sha256"] != measurement.SHARED_PROGRAM_SHA256
    ) or value["execution_mode"] != (
        "native_module_executes_then_exact_output_substitution"
    ) or value["gain_policy"] != "identity_gains_no_mask_specific_refitting":
        raise RuntimeError("new program descriptor schema or binding changed")
    installed = tuple(_site(site) for site in value["installed_compiled_sites"])
    expected = tuple(
        site for site in inherited.ALL_NATIVE_SITES
        if site in set(measurement.REQUESTS[ordinal].sites)
    )
    if installed != expected:
        raise RuntimeError("new program descriptor materializes the wrong mask")
    copied = dict(value)
    copied["installed_compiled_sites"] = installed
    return measurement._logical_sha256(copied)


def _validate_call_ledger(
    value: Any, *, ordinal: int, descriptor_sha256: str,
) -> str:
    expected_keys = {
        "ordinal", "request_sha256", "program_sha256", "row_count",
        "scored_token_count", "batch_count", "outer_forward_count",
        "outer_returned_count", "native_module_calls", "substitution_calls",
        "hook_order", "execution_mode", "fitter_calls", "retained_logits",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("new call-ledger schema changed")
    request = measurement.REQUESTS[ordinal]
    batch_count = measurement.BATCH_COUNT
    native = tuple((_site(site), int(count)) for site, count in value["native_module_calls"])
    substituted = tuple(
        (_site(site), int(count)) for site, count in value["substitution_calls"]
    )
    installed = tuple(
        site for site in inherited.ALL_NATIVE_SITES if site in set(request.sites)
    )
    if value["ordinal"] != ordinal or value["request_sha256"] != request.sha256 or (
        value["program_sha256"] != descriptor_sha256
    ) or value["row_count"] != measurement.ROW_COUNT or value[
        "scored_token_count"
    ] != measurement.ROW_COUNT * measurement.SCORED_TOKENS_PER_ROW or value[
        "batch_count"
    ] != batch_count or value["outer_forward_count"] != batch_count or value[
        "outer_returned_count"
    ] != batch_count or native != tuple(
        (site, batch_count) for site in inherited.ALL_NATIVE_SITES
    ) or substituted != tuple((site, batch_count) for site in installed) or value[
        "hook_order"
    ] != "native_count_registered_before_substitution" or value[
        "execution_mode"
    ] != "native_module_executes_then_exact_output_substitution" or value[
        "fitter_calls"
    ] != 0 or value["retained_logits"] != 0:
        raise RuntimeError("new call ledger fails the physical census")
    copied = dict(value)
    copied["native_module_calls"] = native
    copied["substitution_calls"] = substituted
    return measurement._logical_sha256(copied)


def _load_new_statistics(value: Any) -> measurement.RoleStatistics:
    expected = {
        "role", "authority_sha256", "ordered_document_ids_sha256",
        "document_token_count", "top1_correct", "ce_sum", "statistics_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("new role-statistics payload schema changed")
    statistics = measurement.RoleStatistics(
        role=value["role"], authority_sha256=value["authority_sha256"],
        ordered_document_ids_sha256=value["ordered_document_ids_sha256"],
        document_token_count=value["document_token_count"],
        top1_correct=value["top1_correct"], ce_sum=value["ce_sum"],
    )
    if statistics.sha256 != value["statistics_sha256"]:
        raise RuntimeError("new role-statistics hash changed")
    return statistics


def load_new_terminal_bundles(
    paths: lifecycle.OutputPaths, *, require_authoritative: bool = True,
) -> tuple[
    dict[str, measurement.RoleBundle],
    dict[str, measurement.RoleAuthority],
    dict[str, Any],
]:
    """Validate the new terminal transaction before returning any M outcomes."""

    if not paths.receipt.is_file() or paths.failure.exists():
        raise RuntimeError("new measurement lacks a unique successful terminal receipt")
    receipt = _read_json(paths.receipt)
    receipt_keys = {
        "schema_version", "status", "authoritative_measurement",
        "authorized_for_final_role", "authority_file_sha256",
        "payload_file_sha256", "manifest_file_sha256",
        "two_role_authority_sha256", "source_closure_sha256",
        "program_bank_sha256", "role_receipt_sha256s",
    }
    if set(receipt) != receipt_keys:
        raise RuntimeError("new terminal receipt schema changed")
    authoritative = receipt["authoritative_measurement"]
    expected_status = (
        "complete_two_role_measurement_receipt_last"
        if authoritative else "test_only_non_authoritative_receipt_last"
    )
    if type(authoritative) is not bool or (require_authoritative and not authoritative) or (
        receipt["status"] != expected_status
    ) or receipt["schema_version"] != SCHEMA_VERSION or receipt[
        "authorized_for_final_role"
    ] is not False or lifecycle.file_sha256(paths.authority) != receipt[
        "authority_file_sha256"
    ] or lifecycle.file_sha256(paths.payload) != receipt[
        "payload_file_sha256"
    ] or lifecycle.file_sha256(paths.manifest) != receipt["manifest_file_sha256"]:
        raise RuntimeError("new terminal receipt does not bind its predecessors")

    authority = _read_json(paths.authority)
    authority_keys = {
        "schema_version", "status", "authoritative_measurement",
        "authorized_for_final_role", "source_closure", "source_closure_sha256",
        "role_authorities", "role_authority_sha256s",
        "two_role_authority_sha256", "parent_two_role_authority_sha256",
        "row_bindings", "model_binding", "shared_program_sha256",
        "program_descriptors", "program_bank_sha256", "request_plan_sha256",
        "role_order", "cell_order",
    }
    manifest = _read_json(paths.manifest)
    manifest_keys = {
        "schema_version", "status", "authoritative_measurement",
        "authorized_for_final_role", "source_commit", "source_closure_sha256",
        "authority_file_sha256", "payload_file_sha256",
        "two_role_authority_sha256", "program_bank_sha256",
        "role_receipt_sha256s", "cell_audit_records", "role_order", "cell_order",
    }
    if set(authority) != authority_keys or set(manifest) != manifest_keys:
        raise RuntimeError("new authority or manifest schema changed")
    source = _load_source_closure(authority["source_closure"])
    expected_authority_status = (
        "frozen_before_any_measurement_outcome"
        if authoritative else "test_only_non_authoritative_authority"
    )
    expected_manifest_status = (
        "terminal_payload_verified"
        if authoritative else "test_only_non_authoritative_manifest"
    )
    common = (
        authority["source_closure_sha256"] == source.sha256
        and source.sha256 == receipt["source_closure_sha256"]
        and manifest["source_closure_sha256"] == source.sha256
        and authority["two_role_authority_sha256"]
        == receipt["two_role_authority_sha256"]
        == manifest["two_role_authority_sha256"]
        and authority["program_bank_sha256"]
        == receipt["program_bank_sha256"]
        == manifest["program_bank_sha256"]
        and manifest["source_commit"] == source.source_commit
        and manifest["authority_file_sha256"] == receipt["authority_file_sha256"]
        and manifest["payload_file_sha256"] == receipt["payload_file_sha256"]
        and authority["status"] == expected_authority_status
        and manifest["status"] == expected_manifest_status
        and authority["authoritative_measurement"] is authoritative
        and manifest["authoritative_measurement"] is authoritative
        and authority["authorized_for_final_role"] is False
        and manifest["authorized_for_final_role"] is False
        and authority["shared_program_sha256"] == measurement.SHARED_PROGRAM_SHA256
        and authority["request_plan_sha256"] == measurement.REQUEST_PLAN_SHA256
        and tuple(authority["role_order"]) == assay.ROLE_NAMES
        and tuple(manifest["role_order"]) == assay.ROLE_NAMES
        and authority["cell_order"] == list(range(assay.CELL_COUNT))
        and manifest["cell_order"] == list(range(assay.CELL_COUNT))
    )
    if not common:
        raise RuntimeError("new authority/manifest/receipt chain disagrees")
    parent_authority = lifecycle.parent_authority()
    if authority["parent_two_role_authority_sha256"] != parent_authority[
        "two_role_authority_sha256"
    ] or authority["model_binding"].get("model_realization_sha256") != (
        measurement.MODEL_REALIZATION_SHA256
    ) or authority["model_binding"].get("component_tree_sha256") != (
        measurement.COMPONENT_TREE_SHA256
    ):
        raise RuntimeError("new authority differs from its protected parent realization")

    raw_authorities = authority["role_authorities"]
    raw_authority_hashes = authority["role_authority_sha256s"]
    descriptors = authority["program_descriptors"]
    if not isinstance(raw_authorities, dict) or set(raw_authorities) != set(
        assay.ROLE_NAMES
    ) or not isinstance(raw_authority_hashes, dict) or set(
        raw_authority_hashes
    ) != set(assay.ROLE_NAMES) or not isinstance(descriptors, list) or len(
        descriptors
    ) != assay.CELL_COUNT:
        raise RuntimeError("new two-role authority preimages are incomplete")
    descriptor_hashes = tuple(
        _descriptor_sha256(value, ordinal) for ordinal, value in enumerate(descriptors)
    )
    role_authorities = {
        role: measurement.RoleAuthority(**raw_authorities[role])
        for role in assay.ROLE_NAMES
    }
    observed_authority_hashes = {
        role: role_authorities[role].sha256 for role in assay.ROLE_NAMES
    }
    if observed_authority_hashes != raw_authority_hashes or measurement._logical_sha256(
        observed_authority_hashes
    ) != authority["two_role_authority_sha256"]:
        raise RuntimeError("new two-role authority hash chain changed")

    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "status", "authoritative_measurement",
        "two_role_authority_sha256", "roles",
    } or payload["schema_version"] != SCHEMA_VERSION or payload[
        "status"
    ] != (
        "complete_two_role_sufficient_statistics"
        if authoritative else "test_only_non_authoritative_sufficient_statistics"
    ) or payload["authoritative_measurement"] is not authoritative or payload[
        "two_role_authority_sha256"
    ] != authority["two_role_authority_sha256"] or not isinstance(
        payload["roles"], dict
    ) or tuple(payload["roles"]) != assay.ROLE_NAMES:
        raise RuntimeError("new terminal payload schema changed")

    audit = manifest["cell_audit_records"]
    if not isinstance(audit, dict) or set(audit) != set(assay.ROLE_NAMES):
        raise RuntimeError("new manifest lacks both role audit ledgers")
    bundles: dict[str, measurement.RoleBundle] = {}
    for role in assay.ROLE_NAMES:
        value = payload["roles"][role]
        if not isinstance(value, dict) or set(value) != {
            "role_receipt", "role_receipt_sha256", "statistics",
        }:
            raise RuntimeError("new role payload schema changed")
        raw_receipt = dict(value["role_receipt"])
        raw_receipt["cell_receipt_sha256s"] = tuple(
            raw_receipt["cell_receipt_sha256s"]
        )
        role_receipt = measurement.RoleReceipt(**raw_receipt)
        role_statistics = _load_new_statistics(value["statistics"])
        bundle = measurement.RoleBundle(
            statistics=role_statistics, receipt=role_receipt,
        )
        role_authority = role_authorities[role]
        if role_receipt.sha256 != value["role_receipt_sha256"] or role_receipt.sha256 != (
            receipt["role_receipt_sha256s"].get(role)
        ) or role_receipt.sha256 != manifest["role_receipt_sha256s"].get(role) or (
            role_receipt.authority_sha256 != role_authority.sha256
        ) or role_receipt.source_commit != role_authority.source_commit or (
            role_receipt.source_closure_sha256 != role_authority.source_closure_sha256
        ) or role_receipt.row_file_sha256 != role_authority.row_file_sha256 or (
            role_receipt.row_raw_sha256 != role_authority.row_raw_sha256
        ) or role_receipt.ordered_document_ids_sha256 != (
            role_authority.ordered_document_ids_sha256
        ) or role_receipt.shared_program_sha256 != role_authority.shared_program_sha256:
            raise RuntimeError("new role authority/receipt/statistics chain changed")
        records = audit[role]
        if not isinstance(records, list) or len(records) != assay.CELL_COUNT:
            raise RuntimeError("new role physical audit is incomplete")
        for ordinal, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {
                "ordinal", "program_descriptor_sha256", "cell_receipt",
                "cell_receipt_sha256", "call_ledger", "call_ledger_sha256",
            } or record["ordinal"] != ordinal or record[
                "program_descriptor_sha256"
            ] != descriptor_hashes[ordinal]:
                raise RuntimeError("new cell audit order or descriptor changed")
            raw_cell = dict(record["cell_receipt"])
            cell = measurement.CellReceipt(**raw_cell)
            ledger_sha = _validate_call_ledger(
                record["call_ledger"], ordinal=ordinal,
                descriptor_sha256=descriptor_hashes[ordinal],
            )
            if cell.sha256 != record["cell_receipt_sha256"] or ledger_sha != record[
                "call_ledger_sha256"
            ] or cell.call_ledger_sha256 != ledger_sha or cell.sha256 != (
                role_receipt.cell_receipt_sha256s[ordinal]
            ) or cell.authority_sha256 != role_authority.sha256 or (
                cell.source_closure_sha256 != source.sha256
            ) or cell.model_tree_before_sha256 != measurement.COMPONENT_TREE_SHA256 or (
                cell.model_tree_after_sha256 != measurement.COMPONENT_TREE_SHA256
            ) or cell.shared_program_before_sha256 != measurement.SHARED_PROGRAM_SHA256 or (
                cell.shared_program_after_sha256 != measurement.SHARED_PROGRAM_SHA256
            ):
                raise RuntimeError("new cell receipt or physical ledger changed")
        bundles[role] = bundle
    return bundles, role_authorities, receipt


def _old_grid(
    bundle: parent_measurement.StagedRoleBundle, field: str,
) -> torch.Tensor:
    if not isinstance(bundle, parent_measurement.StagedRoleBundle) or field not in {
        "ce_sum", "top1_correct",
    }:
        raise TypeError("old grid requires a sealed staged role bundle")
    document_count = bundle.discovery.document_count
    dtype = torch.float64 if field == "ce_sum" else torch.long
    grid = torch.empty((document_count, 8, 8), dtype=dtype)
    seen: set[tuple[int, int]] = set()
    for stage in (bundle.discovery, bundle.validation, bundle.heldout):
        values = getattr(stage, field)
        cells = parent_statistics.STAGE_CELLS[stage.stage]
        for column, cell in enumerate(cells):
            if cell in seen:
                raise RuntimeError("protected parent cell appears twice")
            grid[:, cell[0], cell[1]] = values[:, column]
            seen.add(cell)
    if seen != set(parent_registry.ALL_CELLS):
        raise RuntimeError("protected parent grid is incomplete")
    return grid.contiguous()


def join_ce_role_arrays(
    *, role: str, old_bundle: parent_measurement.StagedRoleBundle,
    new_bundle: measurement.RoleBundle,
    new_authority: measurement.RoleAuthority,
    parent_authority: Mapping[str, Any], parent_receipt_file_sha256: str,
) -> assay.RoleArrays:
    """Exact old/new document-level join; no role pooling or scalar rebinding."""

    if role not in assay.ROLE_NAMES or old_bundle.receipt.role != role or (
        new_bundle.receipt.role != role
    ) or new_authority.role != role or new_bundle.receipt.authority_sha256 != (
        new_authority.sha256
    ) or new_bundle.statistics.sha256 != new_bundle.receipt.statistics_sha256:
        raise RuntimeError("old/new role ownership changed")
    raw_parent_role = parent_authority.get("role_authorities", {}).get(role)
    raw_parent_binding = parent_authority.get("row_bindings", {}).get(role)
    if not isinstance(raw_parent_role, Mapping) or not isinstance(
        raw_parent_binding, Mapping
    ) or measurement._logical_sha256(raw_parent_role) != (
        new_authority.parent_role_authority_sha256
    ) or new_authority.parent_measurement_receipt_sha256 != (
        parent_receipt_file_sha256
    ) or new_authority.common_support_sha256 != raw_parent_binding.get(
        "common_support_sha256"
    ) or new_authority.ordered_document_ids_sha256 != (
        old_bundle.discovery.ordered_document_ids_sha256
    ) or new_bundle.statistics.ordered_document_ids_sha256 != (
        old_bundle.discovery.ordered_document_ids_sha256
    ) or new_authority.ordered_document_ids_sha256 != raw_parent_role.get(
        "ordered_document_ids_sha256"
    ) or new_authority.ordered_row_identity_sha256 != raw_parent_role.get(
        "ordered_row_identity_sha256"
    ) or new_authority.ordered_row_to_document_sha256 != raw_parent_role.get(
        "ordered_row_to_document_sha256"
    ) or new_authority.row_token_count_sha256 != raw_parent_role.get(
        "row_token_count_sha256"
    ) or new_authority.row_file_sha256 != raw_parent_role.get("row_file_sha256") or (
        new_authority.row_raw_sha256 != raw_parent_role.get("row_raw_sha256")
    ) or new_authority.shared_program_sha256 != old_bundle.receipt.shared_program_sha256:
        raise RuntimeError("old/new role provenance or physical support does not join")
    old_tokens = old_bundle.discovery.document_token_count
    if not torch.equal(old_tokens, old_bundle.validation.document_token_count) or not (
        torch.equal(old_tokens, old_bundle.heldout.document_token_count)
    ) or not torch.equal(old_tokens, new_bundle.statistics.document_token_count):
        raise RuntimeError("old/new per-document token denominators differ")
    grid = _old_grid(old_bundle, "ce_sum")
    return assay.RoleArrays(
        role=role,
        e=grid[:, :, OLD_SUFFIX_COLUMNS["e"]].numpy(),
        a=grid[:, :, OLD_SUFFIX_COLUMNS["a"]].numpy(),
        m=new_bundle.statistics.ce_sum.numpy(),
        am=grid[:, :, OLD_SUFFIX_COLUMNS["am"]].numpy(),
        token_count=old_tokens.numpy(),
    )


def join_top1_role_arrays(
    *, role: str, old_bundle: parent_measurement.StagedRoleBundle,
    new_bundle: measurement.RoleBundle,
    new_authority: measurement.RoleAuthority,
    parent_authority: Mapping[str, Any], parent_receipt_file_sha256: str,
) -> assay.RoleArrays:
    """Reuse the exact CE join proof, then bind correct-count sufficient statistics."""

    # This call is deliberately retained rather than factoring only the final tensor
    # selection: it proves every old/new provenance and token-denominator equality
    # before the secondary outcome is opened.
    joined_ce = join_ce_role_arrays(
        role=role, old_bundle=old_bundle, new_bundle=new_bundle,
        new_authority=new_authority, parent_authority=parent_authority,
        parent_receipt_file_sha256=parent_receipt_file_sha256,
    )
    grid = _old_grid(old_bundle, "top1_correct")
    return assay.RoleArrays(
        role=role,
        e=grid[:, :, OLD_SUFFIX_COLUMNS["e"]].numpy(),
        a=grid[:, :, OLD_SUFFIX_COLUMNS["a"]].numpy(),
        m=new_bundle.statistics.top1_correct.numpy(),
        am=grid[:, :, OLD_SUFFIX_COLUMNS["am"]].numpy(),
        token_count=joined_ce.token_count,
    )


def score_top1_secondary(data: assay.RoleArrays) -> dict[str, Any]:
    """Report point top-1 effects in percentage points with no decision gate."""

    cost = assay.aggregate(data)
    contrast = assay.contrasts(cost)
    return {
        "role": data.role,
        "unit": "percentage_points",
        "decision_role": "mandatory_secondary_no_gate",
        "cell_top1_percent": {
            name: (100.0 * value).tolist() for name, value in cost.items()
        },
        "contrasts_percentage_points": {
            name: (
                (100.0 * value).tolist()
                if isinstance(value, np.ndarray) else 100.0 * float(value)
            ) for name, value in contrast.items()
        },
    }


def score_transaction(
    *, measurement_paths: lifecycle.OutputPaths | None = None,
    paths: ScorePaths | None = None,
) -> dict[str, Any]:
    measurement_paths = lifecycle.output_paths() if measurement_paths is None else measurement_paths
    paths = score_paths() if paths is None else paths
    canonical_score = paths == score_paths()
    if canonical_score and measurement_paths != lifecycle.output_paths():
        raise RuntimeError("canonical score requires the canonical new measurement namespace")
    paths.require_pristine()
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    phase = "verify_terminal_measurements"
    try:
        if any(path.exists() for path in (paths.results, paths.receipt, paths.failure)):
            raise RuntimeError("broad-MLP score namespace raced after lock acquisition")
        lifecycle.verify_protected_files()
        source = lifecycle.committed_source_closure()
        old_bundles, old_receipt = parent_scorer.load_terminal_bundles(
            lifecycle.PARENT_PATHS, require_authoritative=True,
        )
        new_bundles, new_authorities, new_receipt = load_new_terminal_bundles(
            measurement_paths, require_authoritative=canonical_score,
        )
        if source.sha256 != new_receipt["source_closure_sha256"]:
            raise RuntimeError("scorer source differs from new measurement authority")
        parent_authority = lifecycle.parent_authority()
        parent_receipt_sha = lifecycle.file_sha256(lifecycle.PARENT_PATHS.receipt)
        if parent_receipt_sha != lifecycle.PROTECTED_SHA256[lifecycle.PARENT_PATHS.receipt]:
            raise RuntimeError("protected parent receipt binding changed")

        phase = "join_and_score_two_roles"
        arrays = {
            role: join_ce_role_arrays(
                role=role, old_bundle=old_bundles[role],
                new_bundle=new_bundles[role], new_authority=new_authorities[role],
                parent_authority=parent_authority,
                parent_receipt_file_sha256=parent_receipt_sha,
            ) for role in assay.ROLE_NAMES
        }
        top1_arrays = {
            role: join_top1_role_arrays(
                role=role, old_bundle=old_bundles[role],
                new_bundle=new_bundles[role], new_authority=new_authorities[role],
                parent_authority=parent_authority,
                parent_receipt_file_sha256=parent_receipt_sha,
            ) for role in assay.ROLE_NAMES
        }
        role_scores = {
            role: assay.score_role(arrays[role]) for role in assay.ROLE_NAMES
        }
        cross_scores = {
            f"{source_role}_to_{target_role}": assay.score_cross_role(
                arrays[source_role], arrays[target_role],
            )
            for source_role, target_role in (
                (assay.ROLE_NAMES[0], assay.ROLE_NAMES[1]),
                (assay.ROLE_NAMES[1], assay.ROLE_NAMES[0]),
            )
        }
        within_pass = all(value["useful_pass"] for value in role_scores.values())
        cross_pass = all(value["useful_pass"] for value in cross_scores.values())
        useful_pass = within_pass and cross_pass
        results = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_two_role_broad_mlp_suffix_score"
                if canonical_score else "test_only_non_authoritative_score"
            ),
            "authoritative_score": canonical_score,
            "authorized_for_final_role": False,
            "authorized_for_global_ledger_credit": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_commit": source.source_commit,
            "source_closure_sha256": source.sha256,
            "parent_measurement_receipt_file_sha256": parent_receipt_sha,
            "parent_two_role_authority_sha256": old_receipt[
                "two_role_authority_sha256"
            ],
            "new_measurement_receipt_file_sha256": lifecycle.file_sha256(
                measurement_paths.receipt
            ),
            "new_two_role_authority_sha256": new_receipt[
                "two_role_authority_sha256"
            ],
            "target": "token_weighted_ce_nats",
            "roles": role_scores,
            "conditional_cross_role": cross_scores,
            "top1_secondary": {
                role: score_top1_secondary(top1_arrays[role])
                for role in assay.ROLE_NAMES
            },
            "two_role_within_role_pass": within_pass,
            "two_direction_conditional_cross_role_pass": cross_pass,
            "attention_invariance_useful_pass": useful_pass,
        }
        phase = "publish_results"
        # Bootstrap scoring can be long enough for a source or protected parent
        # to change. Recheck at the last instant before any outcome publication.
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        lifecycle.publish_json_create_only(paths.results, results, lock)
        result_sha = lifecycle.file_sha256(paths.results)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_score_receipt_last"
                if canonical_score else "test_only_non_authoritative_score_receipt_last"
            ),
            "authoritative_score": canonical_score,
            "authorized_for_final_role": False,
            "authorized_for_global_ledger_credit": False,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_closure_sha256": source.sha256,
            "parent_measurement_receipt_file_sha256": parent_receipt_sha,
            "new_measurement_receipt_file_sha256": results[
                "new_measurement_receipt_file_sha256"
            ],
            "results_file_sha256": result_sha,
            "attention_invariance_useful_pass": useful_pass,
        }
        phase = "publish_receipt_last"
        lifecycle.verify_protected_files()
        lifecycle.verify_source_closure(source)
        if lifecycle.file_sha256(lifecycle.PARENT_PATHS.receipt) != parent_receipt_sha or (
            lifecycle.file_sha256(measurement_paths.receipt) != results[
                "new_measurement_receipt_file_sha256"
            ]
        ) or lifecycle.file_sha256(paths.results) != result_sha:
            raise RuntimeError("score predecessor changed before terminal receipt")
        lifecycle.publish_json_create_only(paths.receipt, receipt, lock)
        return receipt
    except Exception as error:
        if not paths.failure.exists():
            try:
                lifecycle.publish_json_create_only(paths.failure, {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed_no_scientific_interpretation",
                    "authorized_for_global_ledger_credit": False,
                    "phase": phase,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }, lock)
            except Exception:
                pass
        raise
    finally:
        if lock.inode is not None:
            lock.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(score_transaction(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
