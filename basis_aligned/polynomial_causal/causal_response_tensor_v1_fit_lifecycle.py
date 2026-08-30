"""Single-owner source-closed lifecycle for causal-response tensor v1 FIT.

The public production entrypoint accepts no arguments.  It freezes authority before
any parent tensor or model load, reconstructs the canonical FIT inputs, owns exactly
one collector, and publishes bundle -> manifest -> one terminal receipt/failure.  It
returns only the terminal digest and cannot mint an EVAL-capable object.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Mapping, NamedTuple

import torch

import bilin18_observed_model_facade as facade
import causal_response_tensor_v1_fit_bundle as fit_bundle
import causal_response_tensor_v1_fit_inputs as fit_inputs
from causal_response_tensor_v1_backend import ObservedResponseCollector


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


def _stable_json_exact_inode(
    path: Path, *, expected_sha256: str, expected_device: int, expected_inode: int,
) -> dict[str, Any]:
    """Read one exact JSON inode through one descriptor, rejecting path replacement."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = (expected_device, expected_inode)
    if (
        (before.st_dev, before.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("FIT JSON inode changed during stable read")
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("FIT JSON hash changed")
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError("FIT JSON is not a plain object")
    return value


def source_closure(commit: str | None = None) -> dict[str, Any]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)):
        raise RuntimeError("FIT source closure contains duplicate paths")
    if commit is None:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    # A short SHA, branch, and full SHA naming the same commit must produce one
    # authority identity.  Resolve before putting the name inside the hashed body.
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("FIT source commit did not resolve to a full SHA-1")
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
    if type(binding["commit"]) is not str or len(binding["commit"]) != 40:
        raise RuntimeError("FIT source closure commit is not canonical")
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


AUTHORITY_KEYS = {
    "schema", "status", "source_closure", "independent_audit", "parents",
    "protocol", "output_paths", "outcome_access_before_authority",
    "authorized_for_fit_execution", "authorized_for_eval", "authority_sha256",
}


def validate_fit_authority() -> tuple[dict[str, Any], str]:
    """Exact-reload the canonical authority and every protected non-outcome binding."""
    authority, artifact_sha256 = stable_json(AUTHORITY)
    if set(authority) != AUTHORITY_KEYS or authority.get("schema") != (
        "causal_response_tensor_v1_fit_authority"
    ) or authority.get("status") != (
        "frozen_before_any_parent_tensor_or_bilin18_model_load"
    ) or authority.get("authorized_for_fit_execution") is not True or (
        authority.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("FIT authority schema or role changed")
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if authority["authority_sha256"] != logical_sha256(body):
        raise RuntimeError("FIT authority logical identity does not replay")
    expected_access = {
        "parent_tensors_loaded": False, "model_loaded": False,
        "model_forward_calls": 0, "scientific_outcomes_read": False,
    }
    if authority["outcome_access_before_authority"] != expected_access or (
        authority["protocol"] != protocol()
        or authority["output_paths"] != output_paths()
    ):
        raise RuntimeError("FIT authority protocol or output namespace changed")
    verify_source_closure(authority["source_closure"])
    if authority["parents"] != parent_snapshot_without_tensor_load():
        raise RuntimeError("FIT authority parent binding changed")
    audit = authority["independent_audit"]
    if type(audit) is not dict or set(audit) != {"path", "sha256", "reviewer"} or (
        audit["path"] != str(AUDIT)
    ):
        raise RuntimeError("FIT authority independent-audit binding changed")
    replay_audit, replay_audit_sha256 = _stable_audit()
    if replay_audit_sha256 != audit["sha256"] or (
        replay_audit["reviewer"] != audit["reviewer"]
        or replay_audit["audited_source_commit"] != authority["source_closure"]["commit"]
    ):
        raise RuntimeError("FIT authority independent audit does not replay")
    return authority, artifact_sha256


def model_state_sha256(
    model: torch.nn.Module, *, require_production: bool = True,
) -> str:
    """Hash the exact named tensor tree, independent of torch serialization bytes."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("FIT model state owner must be a torch module")
    if require_production:
        facade.validate_production_model(model)
    state = model.state_dict()
    if not state:
        raise RuntimeError("FIT model state is empty")
    records: list[list[object]] = []
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or type(value) is not torch.Tensor:
            raise TypeError("FIT model state contains a noncanonical entry")
        owned = value.detach().cpu().contiguous()
        records.append([
            name, str(owned.dtype), list(owned.shape),
            fit_inputs.tensor_sha256(owned),
        ])
    return logical_sha256(records)


def _fsync_parent_best_effort(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        os.fsync(descriptor)
    except OSError:
        # The create-only hard link is already the authoritative transition.
        pass
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _publish_terminal_record(
    record: Mapping[str, Any], *, kind: str, claim: RunClaim,
    final_guard: Callable[[], None],
) -> str:
    """Publish one self-contained terminal record, then its receipt/failure link.

    ``TERMINAL`` and the selected target are hard links to the same fully staged JSON
    inode. Thus even an exceptional failure of the second link leaves a complete,
    hash-bound terminal record rather than a claim that points to absent payload bytes.
    """
    if kind not in ("receipt", "failure") or not callable(final_guard):
        raise ValueError("FIT terminal publication kind or guard is malformed")
    target = RECEIPT if kind == "receipt" else FAILURE
    opposite = FAILURE if kind == "receipt" else RECEIPT
    if type(record) is not dict or set(record) != {
        "schema", "kind", "authority_artifact_sha256",
        "authority_logical_sha256", "aggregate", "payload",
    } or record.get("schema") != "causal_response_tensor_v1_fit_terminal" or (
        record.get("kind") != kind
    ):
        raise RuntimeError("FIT terminal record schema or kind changed")
    for name in ("authority_artifact_sha256", "authority_logical_sha256"):
        value = record[name]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuntimeError("FIT terminal authority binding is malformed")
    if type(record["aggregate"]) is not dict or type(record["payload"]) is not dict:
        raise RuntimeError("FIT terminal aggregate or payload is malformed")
    temporary, digest = _stage_json(record, target)
    try:
        if TERMINAL.exists() or target.exists() or opposite.exists():
            raise RuntimeError("FIT terminal aggregate is already spent")
        # All absence lookups precede the protected-state callback.  The production
        # callback ends in require_claim(); the create-only serialization link is then
        # literally the next operation, so a lookup cannot mutate a protected artifact
        # after it was replayed but before its digest becomes terminal.
        require_claim(claim)
        final_guard()
        # Serialization point shared by success and failure. The record itself is the
        # complete payload, so terminal-only remains semantically recoverable.
        os.link(temporary, TERMINAL)
        _fsync_parent_best_effort(TERMINAL)
        temporary_stat = temporary.stat(follow_symlinks=False)
        if target.exists() or opposite.exists():
            raise RuntimeError("FIT terminal aggregate changed before final link")
        terminal_replay = _stable_json_exact_inode(
            TERMINAL, expected_sha256=digest,
            expected_device=temporary_stat.st_dev,
            expected_inode=temporary_stat.st_ino,
        )
        if terminal_replay != json.loads(json.dumps(
            record, sort_keys=True, allow_nan=False
        )):
            raise RuntimeError("FIT terminal semantics changed before final link")
        require_claim(claim)
        os.link(temporary, target)
        _fsync_parent_best_effort(target)
        return digest
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


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


MANIFEST_KEYS = {
    "schema", "status", "authority_artifact_sha256", "authority_logical_sha256",
    "bundle", "bundle_summary", "protocol", "authorized_for_eval",
    "manifest_sha256",
}


def build_fit_manifest(
    *, authority: Mapping[str, Any], authority_artifact_sha256: str,
    bundle_artifact_sha256: str,
) -> dict[str, Any]:
    """Derive the JSON manifest from the exact published bundle, never from a caller."""
    if type(authority) is not dict or not all(
        isinstance(value, str) and len(value) == 64
        for value in (authority_artifact_sha256, bundle_artifact_sha256)
    ):
        raise TypeError("FIT manifest bindings are malformed")
    summary = fit_bundle.fit_bundle_manifest_summary(
        BUNDLE,
        expected_authority_sha256=authority["authority_sha256"],
        expected_artifact_sha256=bundle_artifact_sha256,
        require_production=True,
    )
    body = {
        "schema": "causal_response_tensor_v1_fit_manifest",
        "status": "complete_fit_bundle_semantically_replayed",
        "authority_artifact_sha256": authority_artifact_sha256,
        "authority_logical_sha256": authority["authority_sha256"],
        "bundle": {
            "path": str(BUNDLE),
            "sha256": bundle_artifact_sha256,
            "bytes": BUNDLE.stat().st_size,
        },
        "bundle_summary": summary,
        "protocol": protocol(),
        "authorized_for_eval": False,
    }
    return {**body, "manifest_sha256": logical_sha256(body)}


def validate_fit_manifest(
    *, expected_authority: Mapping[str, Any],
    expected_authority_artifact_sha256: str,
    expected_bundle_artifact_sha256: str,
    expected_manifest_artifact_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Replay a manifest and rederive its summary from the exact bundle bytes."""
    manifest, artifact_sha256 = stable_json(
        MANIFEST, expected_manifest_artifact_sha256
    )
    if set(manifest) != MANIFEST_KEYS or manifest.get("schema") != (
        "causal_response_tensor_v1_fit_manifest"
    ) or manifest.get("status") != "complete_fit_bundle_semantically_replayed" or (
        manifest.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("FIT manifest schema or status changed")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != logical_sha256(body):
        raise RuntimeError("FIT manifest logical identity does not replay")
    expected_bundle = {
        "path": str(BUNDLE),
        "sha256": expected_bundle_artifact_sha256,
        "bytes": BUNDLE.stat().st_size,
    }
    if (
        manifest["authority_artifact_sha256"]
        != expected_authority_artifact_sha256
        or manifest["authority_logical_sha256"]
        != expected_authority["authority_sha256"]
        or manifest["bundle"] != expected_bundle
        or manifest["protocol"] != protocol()
    ):
        raise RuntimeError("FIT manifest protected binding changed")
    expected_summary = fit_bundle.fit_bundle_manifest_summary(
        BUNDLE,
        expected_authority_sha256=expected_authority["authority_sha256"],
        expected_artifact_sha256=expected_bundle_artifact_sha256,
        require_production=True,
    )
    if manifest["bundle_summary"] != expected_summary:
        raise RuntimeError("FIT manifest bundle summary does not replay")
    return manifest, artifact_sha256


def publish_fit_manifest(
    manifest: Mapping[str, Any], *, claim: RunClaim,
    final_guard: Callable[[], None],
) -> str:
    """Publish a complete manifest create-only after one adjacent owner guard."""
    if type(manifest) is not dict or set(manifest) != MANIFEST_KEYS or not callable(
        final_guard
    ):
        raise TypeError("FIT manifest publication input is malformed")
    temporary, digest = _stage_json(manifest, MANIFEST)
    try:
        if MANIFEST.exists() or TERMINAL.exists() or RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("FIT manifest namespace is already spent")
        require_claim(claim)
        # The production callback performs the exact bundle/model/checkpoint replay and
        # ends in require_claim().  Nothing fallible or path-based may follow it.
        final_guard()
        os.link(temporary, MANIFEST)
        _fsync_parent_best_effort(MANIFEST)
        replay, replay_sha256 = stable_json(MANIFEST, digest)
        if replay != manifest or replay_sha256 != digest:
            raise RuntimeError("published FIT manifest does not replay")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _freeze_fit_authority_under_claim(claim: RunClaim) -> dict[str, Any]:
    """Freeze authority while a sole outer owner retains the run claim."""
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
        # Namespace absence is part of the protected preimage, but every lookup must
        # occur before the final audit/source/parent replay.  A hostile lookup can
        # mutate a parent; the replay below must see and reject that drift.
        if any(path.exists() for path in (
            AUTHORITY, BUNDLE, MANIFEST, RECEIPT, FAILURE, TERMINAL,
        )):
            raise RuntimeError("FIT authority namespace raced publication")
        require_claim(claim)
        replay_audit, replay_audit_sha256 = _stable_audit()
        if (
            replay_audit != audit
            or replay_audit_sha256 != audit_sha256
            or source_closure(closure["commit"]) != closure
            or parent_snapshot_without_tensor_load() != parents
        ):
            raise RuntimeError("FIT authority protected state changed")
        # No path lookup or callback follows this final protected replay.  Claim
        # verification and the create-only authority link are the only operations.
        require_claim(claim)
        os.link(temporary, AUTHORITY)
    finally:
        temporary.unlink(missing_ok=True)
    return authority


def freeze_fit_authority() -> dict[str, Any]:
    """Publish FIT-only authority; does not load parent tensors or the model."""
    require_pristine_namespace()
    claim = acquire_claim()
    try:
        return _freeze_fit_authority_under_claim(claim)
    finally:
        release_claim(claim)


def _require_exact_owner_state(
    *, claim: RunClaim, authority: Mapping[str, Any],
    authority_artifact_sha256: str,
    bundle_artifact_sha256: str | None,
    manifest_artifact_sha256: str | None,
    model: torch.nn.Module | None = None,
    model_state_sha256_expected: str | None = None,
    checkpoint_expected: Mapping[str, Any] | None = None,
) -> None:
    """Replay every state the sole owner is expected to have published so far."""
    require_claim(claim)
    replay_authority, replay_authority_artifact = validate_fit_authority()
    if replay_authority != authority or (
        replay_authority_artifact != authority_artifact_sha256
    ):
        raise RuntimeError("FIT owner authority changed")
    if bundle_artifact_sha256 is None:
        if BUNDLE.exists():
            raise RuntimeError("FIT bundle appeared before its publication boundary")
    elif fit_bundle.semantic_replay_fit_bundle(
        BUNDLE,
        expected_authority_sha256=authority["authority_sha256"],
        expected_artifact_sha256=bundle_artifact_sha256,
        require_production=True,
    ) != bundle_artifact_sha256:
        raise RuntimeError("FIT owner bundle changed")
    if manifest_artifact_sha256 is None:
        if MANIFEST.exists():
            raise RuntimeError("FIT manifest appeared before its publication boundary")
    else:
        validate_fit_manifest(
            expected_authority=authority,
            expected_authority_artifact_sha256=authority_artifact_sha256,
            expected_bundle_artifact_sha256=bundle_artifact_sha256,
            expected_manifest_artifact_sha256=manifest_artifact_sha256,
        )
    if any(path.exists() for path in (TERMINAL, RECEIPT, FAILURE)):
        raise RuntimeError("FIT terminal namespace appeared before owner completion")
    if checkpoint_expected is not None:
        replay_checkpoint = asdict(facade.validate_snapshot())
        if replay_checkpoint != dict(checkpoint_expected):
            raise RuntimeError("FIT checkpoint changed during owner execution")
    if (model is None) != (model_state_sha256_expected is None):
        raise TypeError("FIT owner model guard is incomplete")
    if model is not None and model_state_sha256(model) != model_state_sha256_expected:
        raise RuntimeError("FIT model state changed during owner execution")
    require_claim(claim)


def _artifact_record(path: Path) -> dict[str, Any]:
    """Observe one exact regular-file inode and reject mutation during observation."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        # Confirm absence rather than trusting one raced lookup.
        try:
            os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return {
                "path": str(path), "present": False, "sha256": None,
                "bytes": None, "device": None, "inode": None,
                "mtime_ns": None, "ctime_ns": None,
            }
        raise RuntimeError(f"FIT artifact appeared during absence observation: {path}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"FIT artifact is not a regular file: {path}")

        def digest_descriptor() -> tuple[str, int]:
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            count = 0
            while chunk := os.read(descriptor, 8 << 20):
                digest.update(chunk)
                count += len(chunk)
            return digest.hexdigest(), count

        first_digest, first_count = digest_descriptor()
        middle = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        # A second descriptor read occurs after the path lookup.  Thus an injected
        # same-size write from stat(), or an in-place write invisible to inode checks,
        # changes the digest and is rejected.
        second_digest, second_count = digest_descriptor()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino)
    metadata = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if (
        (middle.st_dev, middle.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or (middle.st_size, middle.st_mtime_ns, middle.st_ctime_ns) != metadata
        or (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != metadata
        or (path_stat.st_size, path_stat.st_mtime_ns, path_stat.st_ctime_ns)
        != metadata
        or first_count != before.st_size
        or second_count != before.st_size
        or first_digest != second_digest
    ):
        raise RuntimeError(f"FIT artifact changed during stable observation: {path}")
    return {
        "path": str(path), "present": True, "sha256": first_digest,
        "bytes": before.st_size, "device": before.st_dev, "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
    }


def _failure_protected_observation(
    model: torch.nn.Module | None,
) -> dict[str, Any]:
    """Observe, rather than require, exact protected state after a failed attempt."""
    source_files = {
        str(path.relative_to(ROOT)): _artifact_record(path) for path in SOURCE_PATHS
    }
    parent_files = {
        "census_state_diverse": _artifact_record(fit_inputs.CENSUS),
        "curated_rows": _artifact_record(fit_inputs.CURATED),
        "battery": _artifact_record(fit_inputs.BATTERY),
        "document_split": _artifact_record(fit_inputs.SPLIT),
        "config": _artifact_record(facade.DEFAULT_SNAPSHOT / "config.json"),
        "weights": _artifact_record(facade.DEFAULT_SNAPSHOT / "pytorch_model.bin"),
    }
    if model is None:
        model_observation: dict[str, Any] = {"present": False, "sha256": None}
    else:
        try:
            model_observation = {
                "present": True,
                "sha256": model_state_sha256(model, require_production=False),
            }
        except Exception as error:
            model_observation = {
                "present": True, "sha256": None,
                "observation_error_type": type(error).__name__,
            }
    return {
        "source_files": source_files,
        "parent_files": parent_files,
        "independent_audit": _artifact_record(AUDIT),
        "model_state": model_observation,
    }


def _failure_guard(
    *, claim: RunClaim, authority: Mapping[str, Any],
    authority_artifact_sha256: str, aggregate: Mapping[str, Any],
    model: torch.nn.Module | None,
) -> None:
    """Bind exact failed state; drift is evidence here, not a success invariant."""
    del authority
    for name, path in (
        ("authority", AUTHORITY), ("bundle", BUNDLE), ("manifest", MANIFEST),
    ):
        if aggregate[name] != _artifact_record(path):
            raise RuntimeError(f"FIT failure {name} bytes changed")
    if aggregate["authority"]["sha256"] != authority_artifact_sha256:
        raise RuntimeError("FIT failure authority digest is inconsistent")
    if aggregate["protected_state"] != _failure_protected_observation(model):
        raise RuntimeError("FIT observed protected failure state changed")
    require_claim(claim)


def execute_causal_response_fit_v1() -> str:
    """Run the canonical production FIT transaction and return only its terminal hash.

    There are intentionally no parameters: device, dtype, checkpoint, inputs, roles,
    masks, batch size, collector, and publication paths are all source-closed.
    """
    require_pristine_namespace()
    claim = acquire_claim()
    authority: dict[str, Any] | None = None
    authority_artifact_sha256: str | None = None
    model: torch.nn.Module | None = None
    model_state_before: str | None = None
    model_state_after: str | None = None
    checkpoint: dict[str, Any] | None = None
    bundle_artifact_sha256: str | None = None
    manifest_artifact_sha256: str | None = None
    try:
        authority = _freeze_fit_authority_under_claim(claim)
        # Transfer the linked authority identity to the outer owner before any
        # fallible post-link operation.  If fsync or semantic validation observes
        # drift, the catch path can still publish a drift-binding failure terminal.
        authority_artifact_sha256 = file_sha256(AUTHORITY)
        _fsync_parent_best_effort(AUTHORITY)
        replay_authority, replay_authority_artifact_sha256 = validate_fit_authority()
        if replay_authority != authority or (
            replay_authority_artifact_sha256 != authority_artifact_sha256
        ):
            raise RuntimeError("published FIT authority does not replay")
        authority = replay_authority

        def inputs_guard() -> None:
            _require_exact_owner_state(
                claim=claim, authority=authority,
                authority_artifact_sha256=authority_artifact_sha256,
                bundle_artifact_sha256=None, manifest_artifact_sha256=None,
            )

        inputs = fit_inputs._reconstruct_production_fit_inputs_after_authority(
            inputs_guard
        )
        inputs_guard()
        model, checkpoint_receipt = facade.load_bilin18(
            device="cuda", dtype=torch.float32,
            snapshot=facade.DEFAULT_SNAPSHOT, verify_weights_sha256=True,
        )
        checkpoint = asdict(checkpoint_receipt)
        if checkpoint["config_sha256"] != authority["parents"]["config_sha256"] or (
            checkpoint["weights_sha256"] != authority["parents"]["weights_sha256"]
        ):
            raise RuntimeError("loaded FIT checkpoint differs from authority")
        model_state_before = model_state_sha256(model)
        collector = ObservedResponseCollector(
            model, inputs.rows, inputs.row_document_ids, inputs.specs,
            batch_size=4, require_production=True,
        )
        preimage = collector.fit_stage(inputs.fit_row_indices)
        model_state_after = model_state_sha256(model)
        if model_state_after != model_state_before:
            raise RuntimeError("model state changed across FIT collection")
        binding = fit_bundle.FitBundleBinding(
            authority_sha256=authority["authority_sha256"],
            source_closure_sha256=authority["source_closure"]["sha256"],
            census_state_diverse_sha256=inputs.parent_sha256s["census_state_diverse"],
            curated_rows_sha256=inputs.parent_sha256s["curated_rows"],
            battery_sha256=inputs.parent_sha256s["battery"],
            document_split_sha256=inputs.parent_sha256s["split"],
            config_sha256=checkpoint["config_sha256"],
            weights_sha256=checkpoint["weights_sha256"],
            model_state_sha256_before=model_state_before,
            model_state_sha256_after=model_state_after,
            model_rows_sha256=inputs.model_rows_sha256,
            fit_role_sha256=inputs.fit_role_sha256,
            fit_document_ids_sha256=inputs.fit_document_ids_sha256,
            support_hashes_sha256=fit_inputs.logical_sha256(inputs.support_hashes),
        )
        payload = fit_bundle.build_fit_bundle_payload(
            preimage, binding, require_production=True
        )

        def before_bundle_link() -> None:
            _require_exact_owner_state(
                claim=claim, authority=authority,
                authority_artifact_sha256=authority_artifact_sha256,
                bundle_artifact_sha256=None, manifest_artifact_sha256=None,
                model=model, model_state_sha256_expected=model_state_after,
                checkpoint_expected=checkpoint,
            )

        bundle_artifact_sha256 = fit_bundle.publish_fit_bundle(
            BUNDLE, payload,
            expected_authority_sha256=authority["authority_sha256"],
            require_production=True, before_link=before_bundle_link,
        )
        manifest = build_fit_manifest(
            authority=authority,
            authority_artifact_sha256=authority_artifact_sha256,
            bundle_artifact_sha256=bundle_artifact_sha256,
        )

        def before_manifest_link() -> None:
            _require_exact_owner_state(
                claim=claim, authority=authority,
                authority_artifact_sha256=authority_artifact_sha256,
                bundle_artifact_sha256=bundle_artifact_sha256,
                manifest_artifact_sha256=None,
                model=model, model_state_sha256_expected=model_state_after,
                checkpoint_expected=checkpoint,
            )

        manifest_artifact_sha256 = publish_fit_manifest(
            manifest, claim=claim, final_guard=before_manifest_link
        )
        aggregate = {
            "authority": _artifact_record(AUTHORITY),
            "bundle": _artifact_record(BUNDLE),
            "manifest": _artifact_record(MANIFEST),
        }
        receipt_payload = {
            "status": "complete",
            "authorized_for_eval": False,
            "checkpoint": checkpoint,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "outer_forwards": payload["call_ledger"]["outer_forwards"],
            "projection_event_shape": protocol()["projection_event_shape"],
            "capture_event_shape": protocol()["capture_event_shape"],
        }
        receipt = {
            "schema": "causal_response_tensor_v1_fit_terminal",
            "kind": "receipt",
            "authority_artifact_sha256": authority_artifact_sha256,
            "authority_logical_sha256": authority["authority_sha256"],
            "aggregate": aggregate,
            "payload": receipt_payload,
        }

        def before_receipt_link() -> None:
            if aggregate != {
                "authority": _artifact_record(AUTHORITY),
                "bundle": _artifact_record(BUNDLE),
                "manifest": _artifact_record(MANIFEST),
            }:
                raise RuntimeError("FIT receipt aggregate changed")
            # End on the complete semantic replay and claim check; the shared terminal
            # helper performs no lookup between this return and its hard link.
            _require_exact_owner_state(
                claim=claim, authority=authority,
                authority_artifact_sha256=authority_artifact_sha256,
                bundle_artifact_sha256=bundle_artifact_sha256,
                manifest_artifact_sha256=manifest_artifact_sha256,
                model=model, model_state_sha256_expected=model_state_after,
                checkpoint_expected=checkpoint,
            )

        return _publish_terminal_record(
            receipt, kind="receipt", claim=claim, final_guard=before_receipt_link
        )
    except Exception as error:
        # Authority publication is the point after which every attempted execution must
        # become externally observable.  Preserve exact partial bytes; never relabel a
        # malformed partial bundle or manifest as a valid scientific artifact.
        if authority is not None and authority_artifact_sha256 is not None and not (
            TERMINAL.exists() or RECEIPT.exists() or FAILURE.exists()
        ):
            try:
                if model is not None:
                    try:
                        model_state_after = model_state_sha256(
                            model, require_production=False
                        )
                    except Exception:
                        model_state_after = None
                # Bind the live authority bytes even when semantic replay failed.
                live_authority = _artifact_record(AUTHORITY)
                if not live_authority["present"]:
                    raise RuntimeError("published FIT authority disappeared")
                authority_artifact_sha256 = live_authority["sha256"]
                aggregate = {
                    "authority": live_authority,
                    "bundle": _artifact_record(BUNDLE),
                    "manifest": _artifact_record(MANIFEST),
                    "protected_state": _failure_protected_observation(model),
                }
                failure = {
                    "schema": "causal_response_tensor_v1_fit_terminal",
                    "kind": "failure",
                    "authority_artifact_sha256": authority_artifact_sha256,
                    "authority_logical_sha256": authority["authority_sha256"],
                    "aggregate": aggregate,
                    "payload": {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "checkpoint": checkpoint,
                        "model_state_sha256_before": model_state_before,
                        "model_state_sha256_after": model_state_after,
                        "authorized_for_eval": False,
                    },
                }
                _publish_terminal_record(
                    failure, kind="failure", claim=claim,
                    final_guard=lambda: _failure_guard(
                        claim=claim, authority=authority,
                        authority_artifact_sha256=authority_artifact_sha256,
                        aggregate=aggregate, model=model,
                    ),
                )
            except Exception as publication_error:
                raise RuntimeError(
                    "FIT execution and failure publication both failed"
                ) from ExceptionGroup(
                    "causal-response FIT double failure", [error, publication_error]
                )
        raise
    finally:
        release_claim(claim)


def main() -> None:
    """CLI surface: execute the fixed transaction and print only its terminal digest."""
    print(execute_causal_response_fit_v1())


if __name__ == "__main__":
    main()
