#!/usr/bin/env python3
"""CPU-only receipt-last finalization of the hash-pinned invalid MLP2 v1 run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import mlp2_cmr_v1_validation_runtime as runtime


AMENDMENT = HERE / "MLP2_CMR_V1R_FINALIZATION_AMENDMENT.md"
V1_AUTHORITY = HERE / "mlp2_cmr_v1_validation_authority.json"
V1_LEDGER = HERE / "mlp2_cmr_v1_validation_ledger.pt"
V1_RESULT = HERE / "mlp2_cmr_v1_validation_result.json"
V1_RECEIPT = HERE / "mlp2_cmr_v1_validation_receipt.json"
V1_FAILURE = HERE / "mlp2_cmr_v1_validation_failure.json"
V1_LOCK = HERE / ".mlp2_cmr_v1_validation.lock"

AUTHORITY = HERE / "mlp2_cmr_v1r_finalization_authority.json"
RESULT = HERE / "mlp2_cmr_v1r_finalization_result.json"
RECEIPT = HERE / "mlp2_cmr_v1r_finalization_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1r_finalization_failure.json"
LOCK = HERE / ".mlp2_cmr_v1r_finalization.lock"

EXPECTED_V1 = {
    "authority": "c22e1fe9e075953d95668893c371f73616823592ba035d6a6fb05c3cf9826bab",
    "ledger": "f7146285e2206872184a26e4df39c45faa3f66e8b0b45959590fc2e38cad5a01",
    "result": "743ef3d5d503c170963202413edc216ba9ec6ce781c5bbb5c2371e49da655178",
    "failure": "47473c67cd8d651af67a767620ede4338d9be3a705039697c04ff24b0ea24f46",
    "lock": "5cbc4f1f3ca33b75867f42274864e5a4b049d1b9774393f355b17721f5b452c9",
}
V1_PATHS = {
    "authority": V1_AUTHORITY, "ledger": V1_LEDGER, "result": V1_RESULT,
    "failure": V1_FAILURE, "lock": V1_LOCK,
}
PARENT_PATHS = {
    "role_rows": HERE / "mlp2_cmr_v1_validation_rows.pt",
    "role_manifest": HERE / "mlp2_cmr_v1_validation_rows_manifest.json",
    "role_receipt": HERE / "mlp2_cmr_v1_validation_rows_receipt.json",
    "fit_bundle": HERE / "mlp2_cmr_v1_fit_mean_bundle.pt",
    "fit_result": HERE / "mlp2_cmr_v1_fit_mean_result.json",
    "fit_receipt": HERE / "mlp2_cmr_v1_fit_mean_receipt.json",
    "suffix_bundle": HERE / "mlp2_cmr_v1_suffix_v2_bundle.pt",
    "suffix_result": HERE / "mlp2_cmr_v1_suffix_v2_result.json",
    "suffix_receipt": HERE / "mlp2_cmr_v1_suffix_v2_receipt.json",
    "correction": HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction.json",
    "correction_receipt": HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction_receipt.json",
    "calibration_bundle": HERE / "mlp2_cmr_v1_fit_selector_calibration_bundle.pt",
    "calibration_result": HERE / "mlp2_cmr_v1_fit_selector_calibration_result.json",
    "calibration_receipt": HERE / "mlp2_cmr_v1_fit_selector_calibration_receipt.json",
    "calibration_role_rows": HERE / "mlp2_cmr_v1_fit_selector_rows.pt",
    "calibration_role_manifest": HERE / "mlp2_cmr_v1_fit_selector_rows_manifest.json",
    "calibration_role_receipt": HERE / "mlp2_cmr_v1_fit_selector_rows_receipt.json",
}
EXPECTED_PARENTS = {
    "role_rows": "f0436268b6a17f1c4c47621ff16d542fe7c20a6579a3fee6e10bce241cee90db",
    "role_manifest": "4ff1155386cec27daf7702675797ec8ed0a1a5534f07c4f24a31fea9ef384a40",
    "role_receipt": "98e077918ab132a6d3e1cc6ffc5d03f3295f3e30adfe7670cf088843b43acebe",
    "fit_bundle": "043bb52b9580d9c9c342460e5bb80ff579db01486b3b6c6672bf5fba77e46f8e",
    "fit_result": "65c1ee33f0399d6489cae0227442d479a9d59b9be98f619d92423cfd39fc7833",
    "fit_receipt": "9dc14d909a1b4aafd33c67dc7a3d066db4ccc9cb83c7059fe7aaf499ca9e5efa",
    "suffix_bundle": "cb3f8d3caecab86881eba825785cabd58c1b7ac8e2aa1eb93b459168cff17ce1",
    "suffix_result": "ab08dc0f0a71b5daf21228991b9e78a272aa74d226d97189ac414a546dc16f62",
    "suffix_receipt": "b61c7308409ec64dc05601206bda21e1f4e24097871ba8dff0c92bc84e761e1f",
    "correction": "ffd5a826962f09ffec0af6c842eaf0bf64530423b827f6239556bc43db9d7ff4",
    "correction_receipt": "dd557dc6366503bea2f3f7649d6312abc8a89857bb277ea4e844d8822e4e968a",
    "calibration_bundle": "3f9aa5ff69530a099c7859b454298eca48cd0789413b412f599746793fd6c1fa",
    "calibration_result": "e30ae749d59dedad4c17159d5b29af1c4c0c79e3f620e794e3a590f3b049c08c",
    "calibration_receipt": "08267122572157203ccf87f9d901d9c4efdfb41c9bb3b4f0d34f1f1f4e669b52",
    "calibration_role_rows": "08a508d6e1526800347d94c6637c84a662c220d84ef30bc674bf6b905ab67798",
    "calibration_role_manifest": "6073c2fd38ad3287c6b7349f2d99aae41c0e98655961255fc18ba4c7c4b745a2",
    "calibration_role_receipt": "6a0dad2f7df3dd17d20fc16df15c03b47c3ef0da30fd65c1cc2149d762709a21",
}
SOURCE_CLOSURE = (
    AMENDMENT, Path(__file__).resolve(), HERE / "test_finalize_mlp2_cmr_v1r.py",
    HERE / "validate_mlp2_cmr_v1.py",
    HERE / "test_validate_mlp2_cmr_v1.py",
    HERE / "mlp2_cmr_v1_validation_runtime.py",
    HERE / "mlp2_cmr_v1_validation_statistics.py",
    HERE / "bilin18_observed_model_facade.py",
)
FORBIDDEN_KEYS = {
    "raw_logits", "native_logits", "candidate_logits", "per_token_losses",
    "validation_targets", "rows", "tokens", "targets", "products", "states",
    "responses",
}
ALLOWED_HASH_PATH = ("role_summary", "tensor_hashes", "rows")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def write_create_only_guarded(
    path: Path, data: bytes, *, before_link: Callable[[], None],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        before_link()
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    hashes: dict[str, str] = {}
    if len(SOURCE_CLOSURE) != len(set(SOURCE_CLOSURE)):
        raise RuntimeError("v1R source closure contains duplicates")
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"v1R source differs from pushed commit: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def reject_forbidden_payloads(value: Any, path: tuple[str, ...] = ()) -> None:
    if torch.is_tensor(value) or isinstance(value, (bytes, bytearray, memoryview)):
        raise RuntimeError("v1R semantic value contains a binary/tensor payload")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RuntimeError("v1R semantic object contains a non-string key")
            child_path = path + (key,)
            if key in FORBIDDEN_KEYS:
                allowed = child_path == ALLOWED_HASH_PATH and type(child) is str and (
                    re.fullmatch(r"[0-9a-f]{64}", child) is not None
                )
                if not allowed:
                    raise RuntimeError("v1R semantic value contains a forbidden raw payload")
            reject_forbidden_payloads(child, child_path)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_forbidden_payloads(child, path)


def current_protected() -> tuple[dict[str, str], dict[str, str], bool]:
    v1 = {name: file_sha256(path) for name, path in V1_PATHS.items()}
    parents = {name: file_sha256(path) for name, path in PARENT_PATHS.items()}
    return v1, parents, V1_RECEIPT.exists()


def validate_claim(nonce: str, inode: tuple[int, int]) -> None:
    stat = LOCK.stat(follow_symlinks=False)
    if (stat.st_dev, stat.st_ino) != inode:
        raise RuntimeError("v1R lock identity changed")
    claim = json.loads(LOCK.read_bytes())
    if claim != {"experiment_id": "bilin18_mlp2_cmr_v1r_finalization", "nonce": nonce}:
        raise RuntimeError("v1R lock claim changed")


def guard_all(
    source_hashes: dict[str, str], nonce: str, inode: tuple[int, int],
    authority_hash: str, *, include_result: str | None = None,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError("v1R source changed after authority")
    v1, parents, receipt_exists = current_protected()
    if v1 != EXPECTED_V1 or parents != EXPECTED_PARENTS or receipt_exists:
        raise RuntimeError("v1R protected v1 snapshot changed")
    if file_sha256(AUTHORITY) != authority_hash:
        raise RuntimeError("v1R authority changed")
    if include_result is not None and file_sha256(RESULT) != include_result:
        raise RuntimeError("v1R result changed")
    validate_claim(nonce, inode)


def _json_parent(name: str) -> Any:
    return json.loads(PARENT_PATHS[name].read_bytes())


def replay_v1() -> tuple[dict[str, Any], dict[str, Any]]:
    if not AUTHORITY.exists():
        raise RuntimeError("v1R authority must exist before v1 parsing")
    old_authority = json.loads(V1_AUTHORITY.read_bytes())
    old_failure = json.loads(V1_FAILURE.read_bytes())
    old_result = json.loads(V1_RESULT.read_bytes())
    ledger = torch.load(V1_LEDGER, map_location="cpu", weights_only=True)
    if old_authority.get("parents") != EXPECTED_PARENTS or old_result.get(
        "parents"
    ) != EXPECTED_PARENTS or old_failure.get("parents") != EXPECTED_PARENTS:
        raise RuntimeError("v1 parent joins changed")
    if old_failure != {
        "schema_version": 1,
        "experiment_id": "bilin18_mlp2_cmr_v1_validation",
        "status": "mlp2_cmr_v1_validation_failed_invalid_no_scientific_decision",
        "source_commit": old_authority["source_commit"],
        "source_hashes": old_authority["source_hashes"],
        "parents": EXPECTED_PARENTS,
        "authority_sha256": EXPECTED_V1["authority"],
        "error_type": "RuntimeError",
        "error": "MLP2 validation contains a forbidden raw payload",
        "partial_outputs": {
            V1_AUTHORITY.name: EXPECTED_V1["authority"],
            V1_LEDGER.name: EXPECTED_V1["ledger"],
            V1_RESULT.name: EXPECTED_V1["result"],
        },
        "replication_opened": False,
    }:
        raise RuntimeError("v1 terminal failure semantics changed")
    expected_keys = {
        "schema", "status", "checkpoint", "checkpoint_after_load", "device",
        "device_name", "model_dtype", "role_summary", "support_hashes",
        "program_receipts", "physical_materialization",
        "selector_gauge_and_permutation_audit", "selector_gauge_permutation_replay",
        "physical_gauge_permutation_replay", "call_ledger", "precision_audit",
        "protocol_audits", "score", "fit_reference", "runtime_seconds",
        "validation_opened", "replication_opened", "raw_logits_published",
        "per_token_losses_published", "validation_targets_published",
        "authority_sha256", "ledger_sha256", "source_commit", "source_hashes",
        "parents",
    }
    if set(old_result) != expected_keys or old_result.get("schema") != (
        "mlp2_cmr_v1_validation_result"
    ) or old_result.get("validation_opened") is not True or old_result.get(
        "replication_opened"
    ) is not False or any(old_result.get(key) is not False for key in (
        "raw_logits_published", "per_token_losses_published", "validation_targets_published",
    )):
        raise RuntimeError("v1 result schema changed")
    reject_forbidden_payloads(old_result)
    role_receipt = _json_parent("role_receipt")
    if old_result.get("role_summary") != role_receipt.get("summary"):
        raise RuntimeError("v1 role summary does not replay published role receipt")
    correction = _json_parent("correction")
    suffix_result = _json_parent("suffix_result")
    import validate_mlp2_cmr_v1 as v1
    protocol = v1.derive_protocol_audits(
        old_result, expected_support_hashes=correction["support_hashes"],
        expected_selector_audit=suffix_result["gauge_and_permutation_audit"],
    )
    if protocol != old_result.get("protocol_audits") or not all(protocol.values()):
        raise RuntimeError("v1 protocol evidence does not replay")
    score = runtime.score_validation_bundle(ledger, protocol_audits=protocol)
    if score != old_result.get("score") or old_result.get("status") != (
        "validation_passed_replication_implementation_authorized"
        if score["validation_passed"] else "validation_failed_replication_remains_sealed"
    ) or score["replication_authorized"] != score["validation_passed"]:
        raise RuntimeError("v1 scientific score does not replay")
    if old_result.get("authority_sha256") != EXPECTED_V1["authority"] or old_result.get(
        "ledger_sha256"
    ) != EXPECTED_V1["ledger"] or old_authority.get("source_commit") != old_result.get(
        "source_commit"
    ) or old_authority.get("source_hashes") != old_result.get("source_hashes") or (
        old_authority.get("replication_authorized") is not False
    ):
        raise RuntimeError("v1 authority/result joins changed")
    if old_result.get("checkpoint") != old_result.get("checkpoint_after_load") or old_result.get(
        "checkpoint", {}
    ).get("weights_sha256") != facade.WEIGHTS_SHA256 or old_result.get("device_name") != (
        "NVIDIA GeForce RTX 5090"
    ) or old_result.get("model_dtype") != str(torch.bfloat16) or not runtime.call_ledger_passes(
        old_result.get("call_ledger", {})
    ) or old_result.get("precision_audit", {}).get("passed") is not True:
        raise RuntimeError("v1 runtime evidence does not replay")
    fit_result = _json_parent("fit_result")
    calibration_result = _json_parent("calibration_result")
    if old_result.get("support_hashes") != correction["support_hashes"] or old_result.get(
        "fit_reference"
    ) != {
        "fit_observations": fit_result["fit_observations"],
        "selector_documents": suffix_result["documents"],
        "calibration_documents": calibration_result["documents"],
    } or type(old_result.get("runtime_seconds")) is not float or old_result["runtime_seconds"] <= 0:
        raise RuntimeError("v1 support/fit/runtime joins changed")
    return old_result, score


def main() -> None:
    namespace = (AUTHORITY, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("v1R namespace already exists")
    if current_protected() != (EXPECTED_V1, EXPECTED_PARENTS, False):
        raise RuntimeError("v1 artifacts do not match frozen amendment")
    source_commit, source_hashes = committed_source()
    experiment_id = "bilin18_mlp2_cmr_v1r_finalization"
    nonce = secrets.token_hex(32)
    inode: tuple[int, int] | None = None
    authority_hash: str | None = None
    try:
        write_create_only_guarded(
            LOCK, canonical_json_bytes({"experiment_id": experiment_id, "nonce": nonce}),
            before_link=lambda: None,
        )
        stat = LOCK.stat(follow_symlinks=False)
        inode = (stat.st_dev, stat.st_ino)
        validate_claim(nonce, inode)
        authority = {
            "schema_version": 1, "experiment_id": experiment_id,
            "status": "authority_frozen_before_v1_result_or_ledger_access",
            "source_commit": source_commit, "source_hashes": source_hashes,
            "v1_artifacts": EXPECTED_V1, "v1_parents": EXPECTED_PARENTS,
            "v1_receipt_required_absent": True,
            "authorized_access": "exact-byte CPU semantic replay only",
            "model_access_authorized": False, "row_access_authorized": False,
            "replication_access_authorized": False,
            "authorized_outputs": [RESULT.name, RECEIPT.name],
        }
        write_create_only_guarded(
            AUTHORITY, canonical_json_bytes(authority),
            before_link=lambda: validate_claim(nonce, inode),
        )
        authority_hash = file_sha256(AUTHORITY)
        guard_all(source_hashes, nonce, inode, authority_hash)
        old_result, score = replay_v1()
        decision = {
            "schema_version": 1, "experiment_id": experiment_id,
            "status": "v1r_semantic_replay_complete",
            "v1_artifacts": EXPECTED_V1, "v1_parents": EXPECTED_PARENTS,
            "source_commit": source_commit, "source_hashes": source_hashes,
            "authority_sha256": authority_hash,
            "correction_scope": "role_summary.tensor_hashes.rows SHA-256 metadata leaf only",
            "v1_failure_preserved": True, "v1_receipt_absent": True,
            "model_accessed": False, "rows_deserialized": False,
            "replication_opened": False,
            "validation_passed": score["validation_passed"],
            "authorized_for_replication_implementation": score["replication_authorized"],
            "authorized_for_replication_execution": False,
            "score": old_result["score"],
        }
        reject_forbidden_payloads(decision)
        guard_all(source_hashes, nonce, inode, authority_hash)
        write_create_only_guarded(
            RESULT, canonical_json_bytes(decision),
            before_link=lambda: guard_all(source_hashes, nonce, inode, authority_hash),
        )
        result_hash = file_sha256(RESULT)
        if json.loads(RESULT.read_bytes()) != decision:
            raise RuntimeError("v1R decision JSON replay failed")
        receipt = {
            "schema_version": 1, "experiment_id": experiment_id,
            "status": "mlp2_cmr_v1r_finalization_complete_receipt_last",
            "authority_sha256": authority_hash, "result_sha256": result_hash,
            "source_commit": source_commit, "source_hashes": source_hashes,
            "v1_artifacts": EXPECTED_V1, "v1_parents": EXPECTED_PARENTS,
            "validation_passed": decision["validation_passed"],
            "authorized_for_replication_implementation": decision[
                "authorized_for_replication_implementation"
            ],
            "authorized_for_replication_execution": False,
        }
        def receipt_guard() -> None:
            guard_all(source_hashes, nonce, inode, authority_hash, include_result=result_hash)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("v1R terminal namespace changed")
        write_create_only_guarded(RECEIPT, canonical_json_bytes(receipt), before_link=receipt_guard)
    except BaseException as error:
        if inode is not None and authority_hash is not None and AUTHORITY.exists() and not (
            RECEIPT.exists() or FAILURE.exists()
        ):
            snapshot = {
                path.name: file_sha256(path) if path.exists() else None
                for path in (AUTHORITY, RESULT)
            }
            failure = {
                "schema_version": 1, "experiment_id": experiment_id,
                "status": "mlp2_cmr_v1r_finalization_failed_no_decision",
                "source_commit": source_commit, "source_hashes": source_hashes,
                "v1_artifacts": EXPECTED_V1, "v1_parents": EXPECTED_PARENTS,
                "error_type": type(error).__name__, "error": str(error),
                "partial_outputs": snapshot, "replication_opened": False,
            }
            def failure_guard() -> None:
                validate_claim(nonce, inode)
                if authority_hash is not None:
                    guard_all(source_hashes, nonce, inode, authority_hash)
                current = {
                    path.name: file_sha256(path) if path.exists() else None
                    for path in (AUTHORITY, RESULT)
                }
                if current != snapshot or RECEIPT.exists() or FAILURE.exists():
                    raise RuntimeError("v1R failure snapshot changed")
            try:
                write_create_only_guarded(FAILURE, canonical_json_bytes(failure), before_link=failure_guard)
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    main()
