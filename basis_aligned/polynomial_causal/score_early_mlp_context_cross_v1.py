#!/usr/bin/env python3
"""Capability-separated CPU scorer for early-MLP/context cross v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

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


def load_terminal_bundles(
    paths: lifecycle.OutputPaths,
) -> tuple[dict[str, measurement.StagedRoleBundle], dict[str, Any]]:
    """Open outcomes only after verifying the last-write terminal receipt."""

    if not paths.receipt.is_file() or paths.failure.exists():
        raise RuntimeError("measurement lacks a unique successful terminal receipt")
    receipt = _read_json(paths.receipt)
    if receipt.get("status") != "complete_two_role_measurement_receipt_last" or (
        receipt.get("authorized_for_final_role") is not False
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
    if manifest.get("status") != "terminal_payload_verified" or manifest.get(
        "authorized_for_final_role"
    ) is not False or manifest.get("source_closure_sha256") != receipt.get(
        "source_closure_sha256"
    ) or manifest.get("program_bank_sha256") != receipt.get(
        "program_bank_sha256"
    ) or authority.get("two_role_authority_sha256") != receipt.get(
        "two_role_authority_sha256"
    ):
        raise RuntimeError("measurement authority/manifest/receipt disagree")
    payload = torch.load(paths.payload, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "status", "two_role_authority_sha256", "roles",
    } or payload["schema_version"] != SCHEMA_VERSION or payload["status"] != (
        "complete_two_role_staged_sufficient_statistics"
    ) or payload["two_role_authority_sha256"] != receipt[
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
    paths.require_pristine()
    lock = lifecycle.RunLock(paths.lock)
    lock.acquire()
    phase = "verify_terminal_measurement"
    try:
        source = lifecycle.committed_source_closure()
        bundles, measurement_receipt = load_terminal_bundles(measurement_paths)
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
        selected = 3 if rank3_pass and rank4_pass else (4 if rank4_pass else None)
        results = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_capability_separated_two_role_score",
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
            "ce_final_useful_pass": rank4_pass,
            "selected_minimal_rank": selected,
            "top1_broad_behavior_pass": None,
        }
        phase = "publish_results"
        lifecycle.publish_json_create_only(paths.results, results, lock)
        results_sha256 = lifecycle.file_sha256(paths.results)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete_score_receipt_last",
            "source_closure_sha256": source.sha256,
            "measurement_receipt_file_sha256": results[
                "measurement_receipt_file_sha256"
            ],
            "results_file_sha256": results_sha256,
            "selected_minimal_rank": selected,
            "ce_final_useful_pass": rank4_pass,
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
