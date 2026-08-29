"""Semantic binding for licensing the completed E4 fit bank in a future selection authority.

This module performs no row, checkpoint, or model load and creates no authority.  It
turns the exact fit v3 bundle into a replayed parent record that a separately audited
selection authority may bind.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import terminal_copy_fit_mean_lifecycle as life
import terminal_copy_fit_mean_recovery_v3 as v3


V3_AUTHORITY_SHA256 = "566d301680b3ff70b400d34d6dc85ca58eda2bc16b64813f2cff2bdc15045fa8"
V3_BANK_SHA256 = "91111377349515125c634c0075c2066ea5c60f827da6dbef6a4f1108dfc84bba"
V3_RESULT_SHA256 = "17fdd8556c1e4972104397df90cffe9d8a64b8fda077d5c41e4d049d8fe7d37e"
V3_MANIFEST_SHA256 = "b8c539f63e70503f743e99c44775bcb8c7c6abb4bb664a9f0a1688da6c6b0dad"
V3_RECEIPT_SHA256 = "663d1f85dab3fbe16d8bd88cb95c2783b17f306ffe928918602e667c6bf2b72f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    before = file_sha256(path)
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    value = json.loads(raw)
    if (
        before != expected_sha256 or raw_sha256 != expected_sha256
        or after != expected_sha256 or not isinstance(value, dict)
    ):
        raise RuntimeError(f"selection fit-parent JSON changed while loading: {path}")
    return value


def replay_fit_parent() -> dict[str, Any]:
    expected = {
        v3.AUTHORITY: V3_AUTHORITY_SHA256,
        v3.BANK: V3_BANK_SHA256,
        v3.RESULT: V3_RESULT_SHA256,
        v3.MANIFEST: V3_MANIFEST_SHA256,
        v3.RECEIPT: V3_RECEIPT_SHA256,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("terminal-copy fit v3 parent bytes changed")
    names = ("AUTHORITY", "BANK", "RESULT", "MANIFEST", "RECEIPT", "FAILURE", "LOCK")
    original_outputs = {name: getattr(life, name) for name in names}
    original_sources = life.SOURCE_PATHS
    original_protected = life.PROTECTED_PATHS
    original_snapshot = life.protected_snapshot
    try:
        v3.configure()
        authority = stable_json(v3.AUTHORITY, V3_AUTHORITY_SHA256)
        result = stable_json(v3.RESULT, V3_RESULT_SHA256)
        manifest = stable_json(v3.MANIFEST, V3_MANIFEST_SHA256)
        receipt = stable_json(v3.RECEIPT, V3_RECEIPT_SHA256)
        life.validate_execution_authority(authority)
        bank = life.load_bank_semantically(
            v3.BANK, authority["authority_sha256"], require_production=True,
        )
        closure_payload = dict(result.get("owner_closure", {}))
        for key in (
            "native_attention_calls", "adapter_decomposition_calls",
            "native_mlp_calls", "final_state_sha256s",
        ):
            if isinstance(closure_payload.get(key), list):
                closure_payload[key] = tuple(closure_payload[key])
        closure = life.FitMeanOwnerClosure(**closure_payload)
        life.validate_closure(closure)
        authority_sha = authority.get("authority_sha256")
        row_sha = authority.get("row_binding", {}).get("input_file_sha256")
        document_sha = authority.get("row_binding", {}).get("ordered_document_ids_sha256")
        checkpoint_sha = authority.get("checkpoint", {}).get("weights_sha256")
        if (
            result.get("schema") != "terminal_copy_fit_means_v1_result"
            or result.get("status") != "complete_fit_only_no_outcome_access"
            or result.get("authority_sha256") != authority_sha
            or result.get("bank_file_sha256") != V3_BANK_SHA256
            or result.get("row_file_sha256") != row_sha
            or result.get("ordered_document_ids_sha256") != document_sha
            or result.get("checkpoint_weights_sha256_before_load") != checkpoint_sha
            or result.get("checkpoint_weights_sha256_after_load") != checkpoint_sha
            or result.get("authorized_for_E4_evidence") is not False
            or result.get("outcome_access") != {
                "candidate_selection": False,
                "label_or_copy_cell_reads": 0,
                "loss_or_logit_reads": 0,
                "unembedding_calls": 0,
            }
            or manifest.get("authority_sha256") != authority_sha
            or manifest.get("protected_unchanged") is not True
            or manifest.get("protected_before") != manifest.get("protected_after")
            or manifest.get("files") != {
                str(v3.BANK): V3_BANK_SHA256,
                str(v3.RESULT): V3_RESULT_SHA256,
            }
            or receipt.get("status") != "complete_receipt_last_fit_only"
            or receipt.get("authority_file_sha256") != V3_AUTHORITY_SHA256
            or receipt.get("authority_sha256") != authority_sha
            or receipt.get("bank_file_sha256") != V3_BANK_SHA256
            or receipt.get("result_file_sha256") != V3_RESULT_SHA256
            or receipt.get("manifest_file_sha256") != V3_MANIFEST_SHA256
            or receipt.get("row_file_sha256") != row_sha
            or receipt.get("ordered_document_ids_sha256") != document_sha
            or receipt.get("checkpoint_weights_sha256_before_load") != checkpoint_sha
            or receipt.get("checkpoint_weights_sha256_after_load") != checkpoint_sha
            or receipt.get("fit_means_prerequisite_complete") is not True
            or receipt.get("authorized_for_candidate_selection_parent") is not False
            or receipt.get("authorized_for_E4_evidence") is not False
            or receipt.get("selection_or_outcome_access") is not False
            or bank.document_count != receipt.get("document_count")
            or bank.document_count != 192
            or bank.ordered_document_ids_sha256 != document_sha
            or bank.master_means_sha256 != result.get("master_means_sha256")
            or bank.runtime_means_sha256 != result.get("runtime_means_sha256")
            or bank.master_means_sha256 != receipt.get("master_means_sha256")
            or bank.runtime_means_sha256 != receipt.get("runtime_means_sha256")
        ):
            raise RuntimeError("terminal-copy fit v3 parent semantics changed")
        binding = {
        "schema": "terminal_copy_selection_fit_parent_binding_v1",
        "fit_authority_file_sha256": V3_AUTHORITY_SHA256,
        "fit_bank_file_sha256": V3_BANK_SHA256,
        "fit_result_file_sha256": V3_RESULT_SHA256,
        "fit_manifest_file_sha256": V3_MANIFEST_SHA256,
        "fit_receipt_file_sha256": V3_RECEIPT_SHA256,
        "fit_authority_sha256": authority["authority_sha256"],
        "master_means_sha256": bank.master_means_sha256,
        "runtime_means_sha256": bank.runtime_means_sha256,
        "ordered_document_ids_sha256": bank.ordered_document_ids_sha256,
        "document_count": bank.document_count,
        "fit_receipt_self_authorizes_selection": False,
        "requires_separate_selection_authority": True,
        }
        if any(file_sha256(path) != digest for path, digest in expected.items()):
            raise RuntimeError("terminal-copy fit v3 parent changed during aggregate replay")
        return binding
    finally:
        for name, value in original_outputs.items():
            setattr(life, name, value)
        life.SOURCE_PATHS = original_sources
        life.PROTECTED_PATHS = original_protected
        life.protected_snapshot = original_snapshot
