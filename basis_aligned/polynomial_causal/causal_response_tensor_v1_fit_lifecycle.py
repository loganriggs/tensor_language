"""Authority and source-closure layer for causal-response tensor v1 FIT.

This module intentionally stops before model execution.  It can construct a
nonauthorizing draft and, after an exact source-bound independent GO exists, publish
the FIT authority before any parent tensor or model is loaded.  Bundle/model/receipt
execution is a later source-closed layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from typing import Any, Mapping, NamedTuple

import bilin18_observed_model_facade as facade
import causal_response_tensor_v1_fit_inputs as fit_inputs


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

AUTHORITY = HERE / "causal_response_tensor_v1_fit_authority.json"
BUNDLE = HERE / "causal_response_tensor_v1_fit_bundle.pt"
MANIFEST = HERE / "causal_response_tensor_v1_fit_manifest.json"
RECEIPT = HERE / "causal_response_tensor_v1_fit_receipt.json"
FAILURE = HERE / "causal_response_tensor_v1_fit_failure.json"
TERMINAL = HERE / "causal_response_tensor_v1_fit_terminal_claim.json"
LOCK = Path("/workspace/runs/.causal_response_tensor_v1_fit.lock")
AUDIT = HERE / "causal_response_tensor_v1_fit_lifecycle_independent_audit.json"

SOURCE_PATHS = tuple(ROOT / relative for relative in (
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_TENSOR_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_TENSOR_V1_AMENDMENT_1.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_TENSOR_V1_AMENDMENT_2.md",
    "basis_aligned/polynomial_causal/causal_response_tensor_contract.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_split.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_collection.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_backend.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_fit_bundle.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_fit_inputs.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_fit_lifecycle.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_contract.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_split.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_collection.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_v1_backend.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_v1_fit_bundle.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_v1_fit_inputs.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_v1_fit_lifecycle.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def stable_json(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    if expected_sha256 is not None and before != expected_sha256:
        raise RuntimeError(f"FIT JSON hash changed: {path}")
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"FIT JSON changed during stable read: {path}")
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError(f"FIT JSON is not a plain object: {path}")
    return value, before


def source_closure(commit: str | None = None) -> dict[str, Any]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)):
        raise RuntimeError("FIT source closure contains duplicate paths")
    if commit is None:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FIT source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"live FIT source differs from commit: {relative}")
        hashes[relative] = digest
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(binding: Mapping[str, Any]) -> None:
    if type(binding) is not dict or set(binding) != {"commit", "paths", "sha256"}:
        raise RuntimeError("FIT source closure schema changed")
    body = {"commit": binding["commit"], "paths": binding["paths"]}
    if logical_sha256(body) != binding["sha256"] or source_closure(
        str(binding["commit"])
    ) != dict(binding):
        raise RuntimeError("FIT source closure does not replay")


def parent_snapshot_without_tensor_load() -> dict[str, Any]:
    observed = {
        "census_state_diverse_sha256": file_sha256(fit_inputs.CENSUS),
        "curated_rows_sha256": file_sha256(fit_inputs.CURATED),
        "battery_sha256": file_sha256(fit_inputs.BATTERY),
        "document_split_sha256": file_sha256(fit_inputs.SPLIT),
        "config_sha256": file_sha256(facade.DEFAULT_SNAPSHOT / "config.json"),
        "weights_sha256": file_sha256(facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"),
    }
    expected = {
        "census_state_diverse_sha256": fit_inputs.PARENT_SHA256S["census_state_diverse"],
        "curated_rows_sha256": fit_inputs.PARENT_SHA256S["curated_rows"],
        "battery_sha256": fit_inputs.PARENT_SHA256S["battery"],
        "document_split_sha256": fit_inputs.PARENT_SHA256S["split"],
        "config_sha256":
            "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
        "weights_sha256":
            "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    }
    if observed != expected:
        raise RuntimeError("a frozen FIT parent changed")
    return observed


def protocol() -> dict[str, Any]:
    return {
        "role": "FIT",
        "model_rows_sha256":
            "1786a30bc0d27d26324486e582a539cc292428c2f3f4f1ed7594014390a437ce",
        "fit_role_sha256":
            "6873c2a279bf73fe17c38d72ac25003f4741825efc271ff91b6b783615cdd815",
        "fit_document_ids_sha256":
            "0f514805a7615e5ef3fe862eb8bf37bebfe8c57b8b7e781fbb25907c729b808d",
        "spec_order_sha256":
            "86d0bd7250102fc8dcdee517562fcadda74f2f6bf6d026582bcab71a33f24ca0",
        "support_hashes_sha256":
            "a8e033d981e82b5e39404ed5ee705119897e1d5d5a1cceaf80ea12c0b711a5aa",
        "rows": 496,
        "source_documents": 343,
        "positions": 256,
        "sources": 49,
        "targets": 49,
        "phases": ["full", "residual"],
        "batch_size": 4,
        "batches": 124,
        "outer_forwards": 12_400,
        "projection_event_shape": [2, 49, 124],
        "capture_event_shape": [6, 124],
        "model_dtype": "torch.float32",
        "fit_arithmetic": "CPU torch.float64 then one deploy cast",
        "artifact_order": [
            "authority_before_parent_tensor_or_model_load", "bundle", "manifest",
            "shared_terminal_claim", "receipt_last",
        ],
        "authorized_for_eval": False,
        "authorized_for_factor_selection": False,
    }


def output_paths() -> dict[str, str]:
    return {
        "authority": str(AUTHORITY), "bundle": str(BUNDLE),
        "manifest": str(MANIFEST), "receipt": str(RECEIPT),
        "failure": str(FAILURE), "terminal": str(TERMINAL), "lock": str(LOCK),
    }


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, BUNDLE, MANIFEST, RECEIPT, FAILURE, TERMINAL, LOCK


def require_pristine_namespace() -> None:
    spent = [str(path) for path in output_namespace() if path.exists()]
    if spent:
        raise RuntimeError(f"causal-response FIT namespace is spent: {spent}")


class RunClaim(NamedTuple):
    descriptor: int
    device: int
    inode: int
    nonce: str
    path: Path


def acquire_claim(path: Path | None = None) -> RunClaim:
    if path is None:
        path = LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"causal-response FIT namespace is locked: {path}") from error
    try:
        os.write(descriptor, (nonce + "\n").encode("ascii"))
        os.fsync(descriptor)
        stat = os.fstat(descriptor)
        return RunClaim(descriptor, stat.st_dev, stat.st_ino, nonce, path)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim) -> None:
    original = os.fstat(claim.descriptor)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(claim.path, flags)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("causal-response FIT claim changed") from error
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, 4096)
        overflow = os.read(descriptor, 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_stat = claim.path.stat(follow_symlinks=False)
    identity = (claim.device, claim.inode)
    if (
        (original.st_dev, original.st_ino) != identity
        or (before.st_dev, before.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or overflow or raw != (claim.nonce + "\n").encode("ascii")
    ):
        raise RuntimeError("causal-response FIT claim changed")


def release_claim(claim: RunClaim) -> None:
    try:
        try:
            require_claim(claim)
        except RuntimeError:
            pass
        else:
            claim.path.unlink()
    finally:
        os.close(claim.descriptor)


def _stable_audit(path: Path | None = None) -> tuple[dict[str, Any], str]:
    if path is None:
        path = AUDIT
    audit, digest = stable_json(path)
    required = {
        "schema", "status", "approved", "outcome_access", "reviewer",
        "audited_source_commit", "audited_source_hashes", "tests_passed",
        "remaining_execution_blockers",
    }
    if set(audit) != required or audit.get("schema") != (
        "causal_response_tensor_v1_fit_lifecycle_independent_audit"
    ) or audit.get("status") != "GO" or audit.get("approved") is not True or (
        audit.get("outcome_access") is not False or not audit.get("reviewer")
        or type(audit.get("tests_passed")) is not int or audit["tests_passed"] < 1
        or audit.get("remaining_execution_blockers") != []
    ):
        raise RuntimeError("FIT independent audit is not an exact execution GO")
    closure = source_closure(str(audit["audited_source_commit"]))
    if audit["audited_source_hashes"] != closure["paths"]:
        raise RuntimeError("FIT independent audit source binding changed")
    return audit, digest


def build_authority_draft() -> dict[str, Any]:
    """Construct a checkable, explicitly nonauthorizing authority draft."""
    require_pristine_namespace()
    closure = source_closure()
    parents = parent_snapshot_without_tensor_load()
    body = {
        "schema": "causal_response_tensor_v1_fit_authority_draft",
        "status": "nonauthoritative_pending_exact_source_bound_independent_GO",
        "source_closure": closure,
        "parents": parents,
        "protocol": protocol(),
        "output_paths": output_paths(),
        "outcome_access_before_authority": {
            "parent_tensors_loaded": False, "model_loaded": False,
            "model_forward_calls": 0, "scientific_outcomes_read": False,
        },
        "authorized_for_fit_execution": False,
        "authorized_for_eval": False,
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def _stage_json(value: Mapping[str, Any], target: Path) -> tuple[Path, str]:
    normalized = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as sink:
            descriptor = -1
            json.dump(normalized, sink, sort_keys=True, indent=2, allow_nan=False)
            sink.write("\n")
            sink.flush(); os.fsync(sink.fileno())
        replay, digest = stable_json(temporary)
        if replay != normalized:
            raise RuntimeError("staged FIT authority JSON does not replay")
        return temporary, digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def freeze_fit_authority() -> dict[str, Any]:
    """Publish FIT-only authority; does not load parent tensors or the model."""
    require_pristine_namespace()
    claim = acquire_claim()
    try:
        audit, audit_sha256 = _stable_audit()
        closure = source_closure(str(audit["audited_source_commit"]))
        parents = parent_snapshot_without_tensor_load()
        body = {
            "schema": "causal_response_tensor_v1_fit_authority",
            "status": "frozen_before_any_parent_tensor_or_bilin18_model_load",
            "source_closure": closure,
            "independent_audit": {
                "path": str(AUDIT), "sha256": audit_sha256,
                "reviewer": audit["reviewer"],
            },
            "parents": parents,
            "protocol": protocol(),
            "output_paths": output_paths(),
            "outcome_access_before_authority": {
                "parent_tensors_loaded": False, "model_loaded": False,
                "model_forward_calls": 0, "scientific_outcomes_read": False,
            },
            "authorized_for_fit_execution": True,
            "authorized_for_eval": False,
        }
        authority = {**body, "authority_sha256": logical_sha256(body)}
        temporary, digest = _stage_json(authority, AUTHORITY)
        try:
            require_claim(claim)
            if any(path.exists() for path in (AUTHORITY, BUNDLE, MANIFEST, RECEIPT,
                                               FAILURE, TERMINAL)):
                raise RuntimeError("FIT authority namespace raced publication")
            if source_closure(closure["commit"]) != closure or (
                parent_snapshot_without_tensor_load() != parents
            ):
                raise RuntimeError("FIT authority protected state changed")
            require_claim(claim)
            os.link(temporary, AUTHORITY)
            directory_descriptor = os.open(AUTHORITY.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            replay, replay_sha = stable_json(AUTHORITY, digest)
            if replay != authority or replay_sha != digest:
                raise RuntimeError("published FIT authority does not replay")
        finally:
            temporary.unlink(missing_ok=True)
        return authority
    finally:
        release_claim(claim)
