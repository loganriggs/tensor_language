"""Source-closed transaction publishing the 229-document FIT training input.

The no-argument production entrypoint binds the completed FIT receipt as opaque
bytes, freezes independently audited analysis authority, spends one exact-byte loader
capability, and publishes a sanitized training-only artifact -> manifest -> one shared
terminal receipt/failure. It cannot open validation or EVAL.
"""

from __future__ import annotations

import hashlib
import json
import os
import ctypes
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, NamedTuple

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_training_input as training_input
from causal_response_factorization_v1_training_loader import OneUseFitTrainingLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "causal_response_factorization_v1_training_authority.json"
INPUT = HERE / "causal_response_factorization_v1_training_input.pt"
MANIFEST = HERE / "causal_response_factorization_v1_training_manifest.json"
TERMINAL_DIR = HERE / "causal_response_factorization_v1_training_terminal"
RECEIPT = TERMINAL_DIR / "receipt.json"
FAILURE = TERMINAL_DIR / "failure.json"
TERMINAL = TERMINAL_DIR / "terminal.json"
AUDIT = HERE / "causal_response_factorization_v1_training_lifecycle_independent_audit.json"
LOCK = Path("/workspace/runs/.causal_response_factorization_v1_training.lock")

SOURCE_PATHS = tuple(ROOT / path for path in (
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_1.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_2.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_3.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_4.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_5.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_6.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_7.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_8.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_9.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_10.md",
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_11.md",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_accelerated.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_fit_adapter.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_parent_binding.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_loader.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_snapshot.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_input.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_lifecycle.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_collection.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_backend.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_fit_bundle.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_accelerated.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_fit_adapter.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_parent_binding.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_training_loader.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_training_snapshot.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_training_input.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_training_lifecycle.py",
    "basis_aligned/polynomial_causal/test_causal_response_tensor_v1_fit_bundle.py",
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


def _normalized_json_bytes(value: Mapping[str, Any]) -> bytes:
    normalized = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    return (
        json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"factor training JSON changed during read: {path}")
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError(f"factor training JSON is not a plain object: {path}")
    return value, before


def protocol() -> dict[str, Any]:
    return {
        "role": "FIT_TRAINING",
        "training_documents": 229,
        "validation_documents_exposed": 0,
        "eval_documents_exposed": 0,
        "response_shape": [2, 49, 49, 229],
        "response_dtype": "torch.float64",
        "valid_dtype": "torch.bool",
        "artifact_order": ["authority", "training_input", "manifest", "terminal_receipt"],
        "authorized_for_candidate_fitting_parent": True,
        "authorized_for_validation": False,
        "authorized_for_eval": False,
    }


def output_paths() -> dict[str, str]:
    return {
        "authority": str(AUTHORITY), "input": str(INPUT),
        "manifest": str(MANIFEST), "receipt": str(RECEIPT),
        "failure": str(FAILURE), "terminal": str(TERMINAL),
        "terminal_directory": str(TERMINAL_DIR), "lock": str(LOCK),
    }


def source_closure(commit: str) -> dict[str, Any]:
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    if resolved != commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("factor training source commit is not published ancestry")
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if completed.returncode != 0 or not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"factor training source does not replay: {relative}")
        hashes[relative] = digest
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def stable_audit() -> tuple[dict[str, Any], str]:
    audit, digest = stable_json(AUDIT)
    if set(audit) != {
        "schema", "status", "approved", "outcome_access", "reviewer",
        "audited_source_commit", "audited_source_hashes", "tests_passed",
        "remaining_execution_blockers",
    } or audit.get("schema") != (
        "causal_response_factorization_v1_training_lifecycle_independent_audit"
    ) or audit.get("status") != "GO" or audit.get("approved") is not True or (
        audit.get("outcome_access") is not False or not audit.get("reviewer")
        or not isinstance(audit.get("tests_passed"), int) or audit["tests_passed"] < 1
        or audit.get("remaining_execution_blockers") != []
    ):
        raise RuntimeError("factor training lifecycle lacks an exact independent GO")
    closure = source_closure(audit["audited_source_commit"])
    if audit["audited_source_hashes"] != closure["paths"]:
        raise RuntimeError("factor training audit source binding changed")
    return audit, digest


class Claim(NamedTuple):
    descriptor: int
    device: int
    inode: int
    nonce: str


def acquire_claim() -> Claim:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(LOCK, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("factor training lifecycle is already locked") from error
    os.write(descriptor, (nonce + "\n").encode())
    os.fsync(descriptor)
    value = os.fstat(descriptor)
    return Claim(descriptor, value.st_dev, value.st_ino, nonce)


def require_claim(claim: Claim) -> None:
    current = os.fstat(claim.descriptor)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(LOCK, flags)
    try:
        observed = os.fstat(descriptor)
        raw = os.read(descriptor, 4096)
        overflow = os.read(descriptor, 1)
    finally:
        os.close(descriptor)
    path_stat = LOCK.stat(follow_symlinks=False)
    identity = (claim.device, claim.inode)
    if any((value.st_dev, value.st_ino) != identity for value in (
        current, observed, path_stat
    )) or overflow or raw != (claim.nonce + "\n").encode():
        raise RuntimeError("factor training owner claim changed")


def release_claim(claim: Claim) -> None:
    try:
        try:
            require_claim(claim)
        except Exception:
            pass
        else:
            try:
                LOCK.unlink()
            except OSError:
                # A terminal snapshot, once installed, is authoritative. A stale
                # advisory lock is recoverable and cannot reverse that transaction.
                pass
    finally:
        try:
            os.close(claim.descriptor)
        except OSError:
            # Once a terminal directory is visible, owner cleanup is bookkeeping and
            # may not turn a committed success/failure into a caller-visible error.
            pass


def _stage_json(value: Mapping[str, Any], target: Path) -> tuple[Path, str]:
    normalized = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    raw = _normalized_json_bytes(normalized)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = -1
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        replay, digest = stable_json(temporary)
        if replay != normalized:
            raise RuntimeError("staged factor training JSON does not replay")
        return temporary, digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_json(value: Mapping[str, Any], target: Path, guard) -> str:
    temporary, digest = _stage_json(value, target)
    try:
        if target.exists():
            raise RuntimeError(f"factor training target is already spent: {target}")
        guard()
        os.link(temporary, target)
        replay, observed = stable_json(target)
        if replay != value or observed != digest:
            raise RuntimeError("published factor training JSON does not replay")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_observation(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": None}
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return {"path": str(path), "exists": True, "sha256": None, "bytes": None,
                    "regular_file": False}
        first_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            first_chunks.append(chunk)
        middle = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        second_chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            second_chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    first, second = b"".join(first_chunks), b"".join(second_chunks)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if any(
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != identity
        for value in (middle, after, path_stat)
    ) or first != second:
        raise RuntimeError(f"factor training artifact changed during observation: {path}")
    return {
        "path": str(path), "exists": True, "regular_file": True,
        "sha256": hashlib.sha256(first).hexdigest(), "bytes": after.st_size,
        "device": after.st_dev, "inode": after.st_ino,
    }


def _stable_artifact_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Capture one exact regular-file version and its physical identity."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"factor training snapshot source is not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if any(
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != identity
        for value in (after, path_stat)
    ) or len(raw) != before.st_size:
        raise RuntimeError(f"factor training snapshot source changed during read: {path}")
    return raw, {
        "source_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _stage_terminal_snapshot(
    staging: Path,
    sources: Mapping[str, tuple[Path, str | None]],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Copy exact historical inputs into the private terminal transaction."""

    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(sources):
        if not name or Path(name).name != name or name in ("receipt.json", "failure.json", "terminal.json"):
            raise RuntimeError("factor training terminal snapshot name is malformed")
        source, expected_sha256 = sources[name]
        raw, record = _stable_artifact_bytes(source)
        if expected_sha256 is not None and record["sha256"] != expected_sha256:
            raise RuntimeError(f"factor training terminal snapshot source changed: {name}")
        target = staging / name
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        if file_sha256(target) != record["sha256"] or target.stat().st_size != record["bytes"]:
            raise RuntimeError(f"factor training staged snapshot does not replay: {name}")
        record = {
            "path_within_terminal_directory": name,
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        paths[name] = target
        records[name] = record
    return paths, records


def _fsync_directory_best_effort(path: Path) -> None:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = _LIBC.renameat2
_RENAMEAT2.argtypes = (
    ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
)
_RENAMEAT2.restype = ctypes.c_int
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically install one directory and refuse every existing destination."""

    result = _RENAMEAT2(
        _AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def _validate_staged_terminal(
    staging: Path,
    *,
    kind: str,
    snapshot_records: Mapping[str, Mapping[str, Any]],
    terminal_sha256: str,
) -> None:
    """Exact final census and byte replay immediately before serialization."""

    expected_names = set(snapshot_records) | {f"{kind}.json", "terminal.json"}
    observed_names = {path.name for path in staging.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError("factor training staged terminal file census changed")
    payload = staging / f"{kind}.json"
    terminal = staging / "terminal.json"
    payload_stat = payload.stat(follow_symlinks=False)
    terminal_stat = terminal.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(payload_stat.st_mode)
        or not stat.S_ISREG(terminal_stat.st_mode)
        or (payload_stat.st_dev, payload_stat.st_ino)
        != (terminal_stat.st_dev, terminal_stat.st_ino)
        or file_sha256(payload) != terminal_sha256
        or file_sha256(terminal) != terminal_sha256
    ):
        raise RuntimeError("factor training staged terminal pair changed")
    for name, record in snapshot_records.items():
        path = staging / name
        observed = path.stat(follow_symlinks=False)
        if (
            set(record) != {"path_within_terminal_directory", "sha256", "bytes"}
            or record["path_within_terminal_directory"] != name
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"factor training staged snapshot record changed: {name}")


def _publish_terminal_pair(
    value: Mapping[str, Any], *, kind: str, claim: Claim, final_guard,
    snapshot_sources: Mapping[str, tuple[Path, str | None]] | None = None,
) -> str:
    """Atomically publish terminal plus receipt/failure as one directory rename.

    Both names are hard links prepared inside a private staging directory.  The sole
    externally visible serialization point is the directory rename, so no state can
    expose only one of the two names.
    """
    if kind not in ("receipt", "failure"):
        raise ValueError("factor training terminal kind is malformed")
    if "terminal_snapshot" in value:
        raise RuntimeError("terminal snapshot is owned by the publisher")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TERMINAL_DIR.name}.", dir=TERMINAL_DIR.parent
    ))
    try:
        snapshot_paths, snapshot_records = _stage_terminal_snapshot(
            staging, {} if snapshot_sources is None else snapshot_sources
        )
        normalized = json.loads(json.dumps(
            {**value, "terminal_snapshot": snapshot_records},
            sort_keys=True, allow_nan=False,
        ))
        raw = _normalized_json_bytes(normalized)
        digest = hashlib.sha256(raw).hexdigest()
        payload = staging / f"{kind}.json"
        terminal = staging / "terminal.json"
        descriptor = os.open(payload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        os.link(payload, terminal)
        if file_sha256(payload) != digest or file_sha256(terminal) != digest or (
            os.stat(payload).st_ino != os.stat(terminal).st_ino
        ):
            raise RuntimeError("staged factor training terminal pair does not replay")
        _fsync_directory_best_effort(staging)
        require_claim(claim)
        final_guard(snapshot_paths, snapshot_records)
        _validate_staged_terminal(
            staging,
            kind=kind,
            snapshot_records=snapshot_records,
            terminal_sha256=digest,
        )
        # Atomic create-only publication is the final filesystem operation.  There is
        # no lookup, callback, sync, or cleanup on the success path after this call.
        _rename_directory_noreplace(staging, TERMINAL_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return digest


def _freeze_authority(
    claim: Claim, attempt: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    audit, audit_digest = stable_audit()
    closure = source_closure(audit["audited_source_commit"])
    fit_parent = parent.fit_parent_binding_without_tensor_load()
    body = {
        "schema": "causal_response_factorization_v1_training_authority",
        "status": "frozen_before_fit_bundle_tensor_deserialization",
        "source_closure": closure,
        "independent_audit": {
            "path": str(AUDIT), "sha256": audit_digest, "reviewer": audit["reviewer"],
        },
        "parent_binding_sha256": fit_parent["binding_sha256"],
        "protocol": protocol(),
        "output_paths": output_paths(),
        "outcome_access_before_authority": {
            "fit_bundle_deserialized": False,
            "fit_response_values_read": False,
            "validation_values_read": False,
            "eval_values_read": False,
        },
        "authorized_for_training_input": True,
        "authorized_for_validation": False,
        "authorized_for_eval": False,
    }
    authority = {**body, "authority_sha256": logical_sha256(body)}
    attempt.update({
        "authority": authority,
        "authority_artifact_sha256": hashlib.sha256(
            _normalized_json_bytes(authority)
        ).hexdigest(),
        "fit_parent": fit_parent,
        "audit": audit,
        "audit_artifact_sha256": audit_digest,
        "source_closure": closure,
    })

    def guard() -> None:
        if stable_audit() != (audit, audit_digest) or source_closure(
            closure["commit"]
        ) != closure or parent.fit_parent_binding_without_tensor_load() != fit_parent:
            raise RuntimeError("factor training authority protected state changed")
        require_claim(claim)

    digest = _publish_json(authority, AUTHORITY, guard)
    return authority, digest, fit_parent


def execute_training_input_v1() -> str:
    """Publish the fixed training-only artifact and return terminal digest only."""

    if any(path.exists() for path in (
        AUTHORITY, INPUT, MANIFEST, TERMINAL_DIR, LOCK,
    )):
        raise RuntimeError("factor training output namespace is spent")
    claim = acquire_claim()
    authority = None
    authority_digest = None
    attempt: dict[str, Any] = {}
    try:
        authority, authority_digest, fit_parent = _freeze_authority(claim, attempt)
        capability = OneUseFitTrainingLoader()
        value = capability.load_once(
            parent_binding=fit_parent,
            expected_analysis_authority_artifact_sha256=authority_digest,
        )
        payload = training_input.build_training_input_payload(
            value, analysis_authority_sha256=authority["authority_sha256"]
        )

        def input_guard() -> None:
            replay, replay_digest = stable_json(AUTHORITY)
            if replay != authority or replay_digest != authority_digest or (
                parent.fit_parent_binding_without_tensor_load() != fit_parent
                or source_closure(authority["source_closure"]["commit"])
                != authority["source_closure"]
            ):
                raise RuntimeError("factor training input protected state changed")
            require_claim(claim)

        input_digest = training_input.publish_training_input(
            INPUT, payload,
            expected_analysis_authority_sha256=authority["authority_sha256"],
            require_production=True, before_link=input_guard,
        )
        manifest_body = {
            "schema": "causal_response_factorization_v1_training_manifest",
            "status": "complete_training_only_input_semantically_replayed",
            "authority_artifact_sha256": authority_digest,
            "authority_logical_sha256": authority["authority_sha256"],
            "fit_parent_binding_sha256": fit_parent["binding_sha256"],
            "input": {
                "path": str(INPUT), "sha256": input_digest,
                "bytes": INPUT.stat().st_size,
            },
            "tensor_hashes": dict(payload["tensor_hashes"]),
            "protocol": protocol(),
            "authorized_for_validation": False,
            "authorized_for_eval": False,
        }
        manifest = {
            **manifest_body, "manifest_sha256": logical_sha256(manifest_body)
        }

        def manifest_guard() -> None:
            authority_replay, authority_observed = stable_json(AUTHORITY)
            audit_replay, audit_observed = stable_audit()
            replay, observed = training_input.replay_training_input(
                INPUT,
                expected_analysis_authority_sha256=authority["authority_sha256"],
                expected_artifact_sha256=input_digest, require_production=True,
            )
            del replay
            if (
                authority_replay != authority or authority_observed != authority_digest
                or audit_replay.get("audited_source_commit")
                != authority["source_closure"]["commit"]
                or audit_observed != authority["independent_audit"]["sha256"]
                or source_closure(authority["source_closure"]["commit"])
                != authority["source_closure"]
                or observed != input_digest
                or parent.fit_parent_binding_without_tensor_load() != fit_parent
            ):
                raise RuntimeError("factor training manifest protected state changed")
            require_claim(claim)

        manifest_digest = _publish_json(manifest, MANIFEST, manifest_guard)
        receipt = {
            "schema": "causal_response_factorization_v1_training_terminal",
            "kind": "receipt",
            "authority_artifact_sha256": authority_digest,
            "authority_logical_sha256": authority["authority_sha256"],
            "payload": {
                "status": "complete_training_only_receipt_last",
                "fit_parent_binding_sha256": fit_parent["binding_sha256"],
                "input_sha256": input_digest,
                "manifest_sha256": manifest_digest,
                "training_documents": 229,
                "validation_values_read": False,
                "eval_values_read": False,
                "authorized_for_candidate_fitting_parent": True,
                "authorized_for_validation": False,
                "authorized_for_eval": False,
            },
        }
        def receipt_guard(snapshot_paths, snapshot_records) -> None:
            authority_replay, authority_observed = stable_json(snapshot_paths["authority.json"])
            audit_replay, audit_observed = stable_json(snapshot_paths["audit.json"])
            input_replay, observed_input = training_input.replay_training_input(
                snapshot_paths["training_input.pt"],
                expected_analysis_authority_sha256=authority["authority_sha256"],
                expected_artifact_sha256=input_digest, require_production=True,
            )
            del input_replay
            manifest_replay, observed_manifest = stable_json(snapshot_paths["manifest.json"])
            if (
                authority_replay != authority or authority_observed != authority_digest
                or audit_replay != attempt["audit"]
                or audit_observed != attempt["audit_artifact_sha256"]
                or observed_input != input_digest
                or manifest_replay != manifest
                or observed_manifest != manifest_digest
                or set(snapshot_records) != {
                    "authority.json", "audit.json", "training_input.pt", "manifest.json"
                }
            ):
                raise RuntimeError("factor training terminal snapshot does not replay")

        return _publish_terminal_pair(
            receipt, kind="receipt", claim=claim, final_guard=receipt_guard,
            snapshot_sources={
                "authority.json": (AUTHORITY, authority_digest),
                "audit.json": (AUDIT, attempt["audit_artifact_sha256"]),
                "training_input.pt": (INPUT, input_digest),
                "manifest.json": (MANIFEST, manifest_digest),
            },
        )
    except Exception as error:
        attempt_authority = attempt.get("authority")
        attempt_digest = attempt.get("authority_artifact_sha256")
        if (
            type(attempt_authority) is dict and isinstance(attempt_digest, str)
            and not TERMINAL_DIR.exists()
        ):
            failure = {
                "schema": "causal_response_factorization_v1_training_terminal",
                "kind": "failure",
                "attempt_authority_artifact_sha256": attempt_digest,
                "attempt_authority_logical_sha256": attempt_authority["authority_sha256"],
                "payload": {
                    "status": "failed_no_training_receipt",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "authorized_for_validation": False,
                    "authorized_for_eval": False,
                },
            }

            failure_sources: dict[str, tuple[Path, str | None]] = {}
            for name, path in (
                ("authority.json", AUTHORITY),
                ("audit.json", AUDIT),
                ("training_input.pt", INPUT),
                ("manifest.json", MANIFEST),
            ):
                if path.is_file():
                    failure_sources[name] = (path, None)

            def failure_guard(snapshot_paths, snapshot_records) -> None:
                if set(snapshot_paths) != set(failure_sources) or set(snapshot_records) != set(
                    failure_sources
                ):
                    raise RuntimeError("factor training failure snapshot does not replay")
                for name, path in snapshot_paths.items():
                    if file_sha256(path) != snapshot_records[name]["sha256"]:
                        raise RuntimeError("factor training failure snapshot changed")

            _publish_terminal_pair(
                failure, kind="failure", claim=claim, final_guard=failure_guard,
                snapshot_sources=failure_sources,
            )
        raise
    finally:
        release_claim(claim)


def main() -> None:
    print(execute_training_input_v1())


if __name__ == "__main__":
    main()
