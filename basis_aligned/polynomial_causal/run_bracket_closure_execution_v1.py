#!/usr/bin/env python3
"""Concrete source-closed adapter for the L13H8 bracket execution.

This module cannot mint its authority, ruling, independent audit, or fresh rows.
Import is I/O-free.  The public transaction validates all metadata before loading a
row tensor or checkpoint, never forwards FIT, owns the exact SELECT/OOD arm loop,
recomputes the scorer from sufficient statistics, and links the receipt last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import bracket_closure_canary_v1 as canary
import bracket_closure_execution_lifecycle_v1 as lifecycle
import bracket_closure_execution_v1 as execution
import bracket_closure_masks_v1 as masks_module
import bracket_closure_rows_v1 as rows_contract
import bracket_closure_tensor_v1 as tensor_program


AUTHORITY_SCHEMA = "bracket_closure_execution_v1_authority"
AUDIT_SCHEMA = "bracket_closure_execution_v1_independent_audit"
RESULT_SCHEMA = "bracket_closure_execution_v1_result_bundle"


def _sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = lifecycle.file_sha256(path)
    raw = path.read_bytes()
    if lifecycle.file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON changed during read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value, before


def _registry(value: Mapping[str, Any]) -> masks_module.DelimiterRegistry:
    if set(value) != {"families", "quote_control_ids", "punctuation_control_ids"} or not (
        isinstance(value["families"], list)
    ):
        raise RuntimeError("execution delimiter registry schema changed")
    families = tuple(masks_module.DelimiterFamily(
        item["name"], tuple(item["opener_ids"]), tuple(item["closer_ids"]),
    ) for item in value["families"])
    return masks_module.DelimiterRegistry(
        families, tuple(value["quote_control_ids"]), tuple(value["punctuation_control_ids"]),
    )


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RuntimeError(f"execution {label} path must be absolute")
    return Path(value)


def validate_authority_payload(
    payload: Mapping[str, Any], *, audit_sha256: str,
) -> execution.ExecutionAuthority:
    keys = {
        "schema", "status", "outcome_access", "source_commit", "source_hashes",
        "row_receipt", "row_cache", "roles", "delimiter_registry", "model",
        "derangement", "programs", "outputs",
    }
    if set(payload) != keys or payload.get("schema") != AUTHORITY_SCHEMA or payload.get(
        "status"
    ) != "frozen_before_any_bracket_model_forward" or payload.get("outcome_access") is not False:
        raise RuntimeError("execution authority schema/status changed")
    source_hashes = payload["source_hashes"]
    if not _sha(payload["source_commit"], 40) or not isinstance(source_hashes, Mapping) or tuple(
        source_hashes
    ) != lifecycle.SOURCE_CLOSURE or any(not _sha(value) for value in source_hashes.values()):
        raise RuntimeError("execution authority source closure changed")
    row_receipt = payload["row_receipt"]
    if not isinstance(row_receipt, Mapping) or set(row_receipt) != {"path", "sha256"} or not (
        _sha(row_receipt["sha256"])
    ):
        raise RuntimeError("execution row receipt binding changed")
    _path(row_receipt["path"], "row receipt"); _path(payload["row_cache"], "row cache")
    roles = payload["roles"]
    role_keys = {
        "filename", "file_sha256", "rows_sha256", "records_sha256",
        "support_sha256", "document_ids_sha256",
    }
    if not isinstance(roles, Mapping) or tuple(roles) != ("fit", "select", "ood") or any(
        not isinstance(value, Mapping) or set(value) != role_keys
        or not isinstance(value["filename"], str) or not value["filename"]
        or Path(value["filename"]).name != value["filename"]
        or any(not _sha(value[key]) for key in role_keys - {"filename"})
        for value in roles.values()
    ) or len({value["filename"] for value in roles.values()}) != 3:
        raise RuntimeError("execution role bindings changed")
    registry = _registry(payload["delimiter_registry"])
    model = payload["model"]
    if not isinstance(model, Mapping) or set(model) != {
        "snapshot", "config_sha256", "weights_sha256",
    } or _path(model["snapshot"], "model snapshot") != facade.DEFAULT_SNAPSHOT or (
        model["config_sha256"] != facade.CONFIG_SHA256
        or model["weights_sha256"] != facade.WEIGHTS_SHA256
    ):
        raise RuntimeError("execution model binding changed")
    derangement = payload["derangement"]
    if not isinstance(derangement, Mapping) or set(derangement) != {
        "path", "file_sha256", "tensor_sha256",
    } or not _sha(derangement["file_sha256"]) or not _sha(derangement["tensor_sha256"]):
        raise RuntimeError("execution derangement binding changed")
    _path(derangement["path"], "derangement")
    programs = payload["programs"]
    if not isinstance(programs, list) or len(programs) != 3:
        raise RuntimeError("execution program bindings changed")
    program_keys = {
        "arm", "state_sha256", "stored_values", "native_calls_per_forward",
        "token_table_values", "total_input_support",
    }
    if any(not isinstance(item, Mapping) or set(item) != program_keys for item in programs):
        raise RuntimeError("execution program binding schema changed")
    program_bindings = tuple(execution.ProgramAuthority(
        item["arm"], item["state_sha256"], item["stored_values"],
        item["native_calls_per_forward"], item["token_table_values"],
        item["total_input_support"],
    ) for item in programs)
    outputs = payload["outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "result", "receipt", "failure", "lock",
    }:
        raise RuntimeError("execution output namespace changed")
    output_paths = tuple(_path(outputs[key], key) for key in ("result", "receipt", "failure", "lock"))
    if len(set(output_paths)) != 4:
        raise RuntimeError("execution output paths overlap")
    family_names = tuple(family.name for family in registry.families)
    return execution.ExecutionAuthority(
        payload["source_commit"], tuple(source_hashes.items()), row_receipt["sha256"],
        tuple((role, roles[role]["file_sha256"]) for role in roles),
        tuple((role, roles[role]["support_sha256"]) for role in roles),
        tuple((role, roles[role]["document_ids_sha256"]) for role in roles),
        family_names, model["config_sha256"], model["weights_sha256"],
        derangement["tensor_sha256"], program_bindings, True,
        source_hashes[
            "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_AMENDMENT.md"
        ], audit_sha256,
    )


def validate_independent_audit(
    audit: Mapping[str, Any], *, authority_sha256: str,
    authority_payload: Mapping[str, Any],
) -> None:
    if set(audit) != {
        "schema", "status", "outcome_access", "authority_sha256", "source_commit",
        "source_hashes", "tests_passed", "reviewer",
    } or audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "GO" or audit.get(
        "outcome_access"
    ) is not False or audit.get("authority_sha256") != authority_sha256 or audit.get(
        "source_commit"
    ) != authority_payload["source_commit"] or audit.get("source_hashes") != authority_payload[
        "source_hashes"
    ] or type(audit.get("tests_passed")) is not int or audit["tests_passed"] < 1 or not (
        isinstance(audit.get("reviewer"), str) and audit["reviewer"]
    ):
        raise RuntimeError("execution independent audit is not an exact outcome-blind GO")


def _record(item: Mapping[str, Any]) -> rows_contract.CandidateRecord:
    if set(item) != {
        "document_id", "source_document_index", "source_file", "source_revision",
        "source_blob_sha256", "domain", "license_id", "normalized_python_sha256",
    }:
        raise RuntimeError("execution role provenance schema changed")
    return rows_contract.CandidateRecord(
        item["document_id"], item["source_document_index"], item["source_file"],
        item["source_revision"], item["source_blob_sha256"],
        masks_module.BracketDomain(item["domain"]), item["license_id"],
        item["normalized_python_sha256"],
    )


def _load_role(
    path: Path, role: str, binding: Mapping[str, Any],
    registry: masks_module.DelimiterRegistry,
) -> tuple[execution.RoleMaterialization, tuple[rows_contract.CandidateRecord, ...]]:
    before = lifecycle.file_sha256(path)
    if before != binding["file_sha256"]:
        raise RuntimeError(f"execution {role} role file hash changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if lifecycle.file_sha256(path) != before or not isinstance(payload, dict) or set(payload) != {
        "schema", "role", "rows", "records", "masks", "support",
    } or payload["schema"] != "bracket_closure_rows_v1_role" or payload["role"] != role:
        raise RuntimeError(f"execution {role} role schema changed")
    rows = payload["rows"]
    if not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long or (
        tuple(rows.shape) != (320, 257) or not rows.is_contiguous()
    ):
        raise RuntimeError(f"execution {role} rows changed currency")
    records = tuple(_record(item) for item in payload["records"])
    if len(records) != rows.shape[0]:
        raise RuntimeError(f"execution {role} provenance count changed")
    masks = masks_module.build_bracket_masks(
        rows, registry, tuple(record.domain for record in records), first_prediction=64,
    )
    mask_payload = {
        **dict(masks.named_cells()), "family_index": masks.family_index,
        "depth": masks.depth, "distance": masks.distance, "domain_index": masks.domain_index,
    }
    if set(payload["masks"]) != set(mask_payload) or any(
        not torch.equal(payload["masks"][name], value) for name, value in mask_payload.items()
    ):
        raise RuntimeError(f"execution {role} masks do not replay from rows")
    support = {
        domain.value: rows_contract.support_census(masks, domain)
        for domain in masks_module.BracketDomain
    }
    record_sha = _logical_sha(payload["records"])
    materialization = execution.RoleMaterialization(
        role, rows, tuple(record.document_id for record in records), masks,
    )
    observed = {
        "filename": path.name, "file_sha256": before,
        "rows_sha256": rows_contract.tensor_sha256(rows), "records_sha256": record_sha,
        "support_sha256": canary.support_sha256(rows, masks),
        "document_ids_sha256": _logical_sha(list(materialization.document_ids)),
    }
    if payload["support"] != support or observed != dict(binding):
        raise RuntimeError(f"execution {role} role semantic binding changed")
    return materialization, records


def load_bound_roles(
    payload: Mapping[str, Any], authority: execution.ExecutionAuthority,
) -> tuple[execution.RoleMaterialization, execution.RoleMaterialization]:
    """Load all role metadata for disjointness, but return only forwardable roles."""
    receipt_path = _path(payload["row_receipt"]["path"], "row receipt")
    receipt, receipt_sha = _stable_json(receipt_path)
    if receipt_sha != authority.row_receipt_sha256 or set(receipt) != {
        "schema", "status", "authority_sha256", "audit_sha256", "source_commit",
        "source_hashes", "candidate_sha256", "candidate_source_identity_sha256",
        "delimiter_registry_sha256", "historical_registry_hashes",
        "historical_exclusion_counts", "entries", "outcome_access",
    } or receipt["schema"] != "bracket_closure_rows_v1_receipt" or receipt[
        "status"
    ] != "frozen_before_any_model_forward_receipt_last" or receipt["outcome_access"] is not False:
        raise RuntimeError("execution row receipt changed")
    registry = _registry(payload["delimiter_registry"])
    if receipt["delimiter_registry_sha256"] != rows_contract.registry_sha256(registry) or (
        receipt["entries"] != {
            role: {key: payload["roles"][role][key] for key in (
                "filename", "file_sha256", "rows_sha256", "records_sha256",
            )} for role in ("fit", "select", "ood")
        }
    ):
        raise RuntimeError("execution row parent join changed")
    cache = _path(payload["row_cache"], "row cache")
    loaded = {}
    for role in ("fit", "select", "ood"):
        loaded[role] = _load_role(
            cache / payload["roles"][role]["filename"], role,
            payload["roles"][role], registry,
        )
    identity_sets = []
    for role in ("fit", "select", "ood"):
        materialization, records = loaded[role]
        identity_sets.append({
            "documents": set(materialization.document_ids),
            "code_files": {record.source_file for record in records
                           if record.domain is masks_module.BracketDomain.CODE},
            "rows": {rows_contract.row_sha256(row) for row in materialization.rows},
            "prefixes": {rows_contract.prefix32_sha256(row) for row in materialization.rows},
            "python": {record.normalized_python_sha256 for record in records
                       if record.normalized_python_sha256 is not None},
        })
    for left in range(3):
        for right in range(left + 1, 3):
            if any(identity_sets[left][key] & identity_sets[right][key]
                   for key in identity_sets[left]):
                raise RuntimeError("execution FIT/SELECT/OOD identities overlap")
    return loaded["select"][0], loaded["ood"][0]


def load_derangement(payload: Mapping[str, Any], authority: execution.ExecutionAuthority) -> torch.Tensor:
    path = _path(payload["derangement"]["path"], "derangement")
    before = lifecycle.file_sha256(path)
    if before != payload["derangement"]["file_sha256"]:
        raise RuntimeError("execution derangement file hash changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if lifecycle.file_sha256(path) != before:
        raise RuntimeError("execution derangement file changed during read")
    if isinstance(value, Mapping) and set(value) == {"permutation"}:
        value = value["permutation"]
    execution.validate_derangement_realization(value, authority)
    return value


def _stats_payload(stats: execution.RoleSufficientStatistics) -> dict[str, Any]:
    stats.validate()
    return {
        "role": stats.role, "document_ids": list(stats.document_ids),
        "coordinate_names": list(stats.coordinate_names), "counts": stats.counts.tolist(),
        "ce_sums": stats.ce_sums.tolist(),
        "teacher_kl_sums": stats.teacher_kl_sums.tolist(),
        "correct_sums": stats.correct_sums.tolist(),
        "replay_max_abs_logit": stats.replay_max_abs_logit,
    }


def _stats_from_payload(value: Mapping[str, Any]) -> execution.RoleSufficientStatistics:
    if set(value) != {
        "role", "document_ids", "coordinate_names", "counts", "ce_sums",
        "teacher_kl_sums", "correct_sums", "replay_max_abs_logit",
    }:
        raise RuntimeError("execution sufficient-statistic schema changed")
    result = execution.RoleSufficientStatistics(
        value["role"], tuple(value["document_ids"]), tuple(value["coordinate_names"]),
        torch.tensor(value["counts"], dtype=torch.int64),
        torch.tensor(value["ce_sums"], dtype=torch.float64),
        torch.tensor(value["teacher_kl_sums"], dtype=torch.float64),
        torch.tensor(value["correct_sums"], dtype=torch.float64),
        float(value["replay_max_abs_logit"]),
    )
    result.validate(); return result


def closure_summary(
    closures: Mapping[tuple[str, str], tuple[Any, ...]],
    expected_documents: Mapping[str, int],
) -> dict[str, Any]:
    execution.validate_execution_ledgers(closures, expected_documents)
    output = {}
    for role in execution.ROLE_ORDER:
        output[role] = {}
        for arm in canary.ARM_NAMES:
            entries = closures[(role, arm)]
            target = [entry.sites[tensor_program.TARGET_SITE] for entry in entries]
            output[role][arm] = {
                "batches": len(entries), "documents": sum(entry.document_count for entry in entries),
                "native_l13_calls": sum(item.native_attention_calls for item in target),
                "replacement_l13_calls": sum(item.replacement_attention_calls for item in target),
            }
    return output


def _top1_payload(stats: execution.RoleSufficientStatistics) -> dict[str, Any]:
    denominator = stats.counts.double()
    output = {}
    for arm_index, arm in enumerate(canary.ARM_NAMES):
        document_values = (stats.correct_sums[arm_index] / denominator.clamp_min(1)).masked_fill(
            denominator == 0, float("nan"),
        )
        output[arm] = {
            name: float(torch.nanmean(document_values[:, column]))
            for column, name in enumerate(stats.coordinate_names)
        }
    return output


def build_result_payload(
    authority_sha256: str, select: execution.RoleSufficientStatistics,
    ood: execution.RoleSufficientStatistics, closures: Mapping[tuple[str, str], tuple[Any, ...]],
    authority: execution.ExecutionAuthority,
) -> dict[str, Any]:
    integrity = execution.ExecutionIntegrity(*(True for _ in range(7)))
    score = execution.score_roles(select, ood, integrity, authority.delimiter_family_names)
    return {
        "schema": RESULT_SCHEMA, "authority_sha256": authority_sha256,
        "raw_statistics": {"select": _stats_payload(select), "ood": _stats_payload(ood)},
        "call_ledger": closure_summary(
            closures, {"select": len(select.document_ids), "ood": len(ood.document_ids)},
        ),
        "top1_secondary_no_gate": {
            "select": _top1_payload(select), "ood": _top1_payload(ood),
        },
        "score": score.to_payload(), "promoted": score.promoted,
    }


def validate_result_payload(
    payload: Mapping[str, Any], *, authority_sha256: str,
    authority: execution.ExecutionAuthority,
    expected_roles: tuple[execution.RoleMaterialization, execution.RoleMaterialization],
    expected_call_ledger: Mapping[str, Any],
) -> None:
    if set(payload) != {
        "schema", "authority_sha256", "raw_statistics", "call_ledger",
        "top1_secondary_no_gate", "score", "promoted",
    } or payload.get("schema") != RESULT_SCHEMA or payload.get(
        "authority_sha256"
    ) != authority_sha256 or payload.get("call_ledger") != expected_call_ledger:
        raise RuntimeError("execution result envelope changed")
    raw = payload["raw_statistics"]
    if not isinstance(raw, Mapping) or tuple(raw) != ("select", "ood"):
        raise RuntimeError("execution result role order changed")
    select, ood = (_stats_from_payload(raw[role]) for role in execution.ROLE_ORDER)
    if tuple(select.document_ids) != expected_roles[0].document_ids or tuple(
        ood.document_ids
    ) != expected_roles[1].document_ids:
        raise RuntimeError("execution result document identities changed")
    score = execution.score_roles(
        select, ood, execution.ExecutionIntegrity(*(True for _ in range(7))),
        authority.delimiter_family_names,
    )
    if payload["score"] != score.to_payload() or payload["promoted"] is not score.promoted or (
        payload["top1_secondary_no_gate"] != {
            "select": _top1_payload(select), "ood": _top1_payload(ood),
        }
    ):
        raise RuntimeError("execution result score/top1 does not replay")


def _guard_inputs(
    authority_path: Path, authority_sha256: str, audit_path: Path, audit_sha256: str,
    payload: Mapping[str, Any], authority: execution.ExecutionAuthority,
) -> None:
    current, current_sha = _stable_json(authority_path)
    audit, current_audit_sha = _stable_json(audit_path)
    if current_sha != authority_sha256 or current != payload or current_audit_sha != audit_sha256:
        raise RuntimeError("execution authority/audit changed")
    validate_independent_audit(
        audit, authority_sha256=authority_sha256, authority_payload=payload,
    )
    lifecycle.verify_source_binding(authority)
    receipt_path = _path(payload["row_receipt"]["path"], "row receipt")
    if lifecycle.file_sha256(receipt_path) != authority.row_receipt_sha256:
        raise RuntimeError("execution row receipt changed")
    cache = _path(payload["row_cache"], "row cache")
    for role, binding in payload["roles"].items():
        if lifecycle.file_sha256(cache / binding["filename"]) != binding["file_sha256"]:
            raise RuntimeError(f"execution {role} role file changed")
    if lifecycle.file_sha256(
        _path(payload["derangement"]["path"], "derangement")
    ) != payload["derangement"]["file_sha256"]:
        raise RuntimeError("execution derangement file changed")
    facade.validate_snapshot(payload["model"]["snapshot"], verify_weights_sha256=True)


def run(authority_path: Path, audit_path: Path) -> dict[str, Any]:
    """Execute one externally authorized transaction; never creates authority/audit."""
    payload, authority_sha256 = _stable_json(authority_path)
    audit, audit_sha256 = _stable_json(audit_path)
    authority = validate_authority_payload(payload, audit_sha256=audit_sha256)
    validate_independent_audit(
        audit, authority_sha256=authority_sha256, authority_payload=payload,
    )
    execution.require_launch_ready(authority)
    lifecycle.verify_source_binding(authority)
    outputs = {key: _path(value, key) for key, value in payload["outputs"].items()}
    if any(path.exists() for path in outputs.values()):
        raise RuntimeError("execution output namespace is spent")
    with lifecycle.RunLock(outputs["lock"]) as lock:
        try:
            guard = lambda: _guard_inputs(
                authority_path, authority_sha256, audit_path, audit_sha256, payload, authority,
            )
            guard()
            roles = load_bound_roles(payload, authority)
            permutation = load_derangement(payload, authority)
            guard()
            model, receipt = facade.load_bilin18(
                device="cuda", dtype=torch.float32, snapshot=payload["model"]["snapshot"],
                verify_weights_sha256=True,
            )
            if (receipt.config_sha256, receipt.weights_sha256) != (
                authority.model_config_sha256, authority.model_weights_sha256,
            ):
                raise RuntimeError("execution loaded model differs from authority")
            select, ood, closures = execution.execute_loaded_roles(
                model, roles, permutation, authority, source_guard=guard,
            )
            del model
            ledger = closure_summary(
                closures, {"select": len(select.document_ids), "ood": len(ood.document_ids)},
            )
            result = build_result_payload(
                authority_sha256, select, ood, closures, authority,
            )
            semantic_guard = lambda: (
                guard(),
                validate_result_payload(
                    result, authority_sha256=authority_sha256, authority=authority,
                    expected_roles=roles, expected_call_ledger=ledger,
                ),
            )
            semantic_guard()
            lifecycle.publish_result_receipt_last(
                result, outputs["result"], outputs["receipt"],
                authority_sha256=authority_sha256, lock=lock, final_guard=semantic_guard,
            )
            return result
        except BaseException as error:
            if not outputs["receipt"].exists() and not outputs["failure"].exists():
                try:
                    lifecycle.publish_json_receipt_last({
                        "schema": "bracket_closure_execution_v1_failure",
                        "status": "terminal_failure_without_success_receipt",
                        "authority_sha256": authority_sha256,
                        "error_type": type(error).__name__, "error": str(error),
                    }, outputs["failure"], lock=lock, final_guard=lock.require_owned)
                except BaseException:
                    pass
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.authority, arguments.audit), indent=2))


__all__ = (
    "build_result_payload", "closure_summary", "load_bound_roles", "load_derangement",
    "run", "validate_authority_payload", "validate_independent_audit",
    "validate_result_payload",
)
