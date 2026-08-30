"""Source replay and receipt-last lifecycle for bracket execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Mapping

import bracket_closure_execution_v1 as execution


ROOT = Path(__file__).resolve().parents[2]

SOURCE_CLOSURE = (
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_CANARY_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_ROWS_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_NO_GO.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/bracket_closure_canary_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_execution_lifecycle_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_tensor_v1.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/freeze_bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/run_bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_canary_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_execution_lifecycle_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_tensor_v1.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/test_run_bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/test_tensor_preserving_attention.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)


class RunLock:
    def __init__(self, path: Path):
        self.path = path; self.claim = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_hex(16)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(descriptor, nonce.encode()); os.fsync(descriptor); os.close(descriptor)
        stat = self.path.stat(); self.claim = (stat.st_dev, stat.st_ino, nonce)
        return self

    def require_owned(self):
        if self.claim is None:
            raise RuntimeError("execution lock is not owned")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        try:
            before = os.fstat(descriptor)
            raw = os.read(descriptor, 4096)
            if os.read(descriptor, 1):
                raise RuntimeError("execution lock contents changed")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_stat = self.path.stat(follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or (
            before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
        ) or (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino) or (
            before.st_dev, before.st_ino, raw.decode()
        ) != self.claim:
            raise RuntimeError("execution lock ownership changed")

    def __exit__(self, *_):
        try: self.require_owned(); self.path.unlink()
        except (FileNotFoundError, RuntimeError): pass


def publish_json_receipt_last(
    payload: Mapping[str, Any], path: Path, *, lock: RunLock, final_guard,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as sink:
            sink.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
            sink.flush(); os.fsync(sink.fileno())
        if json.loads(temporary.read_bytes()) != payload:
            raise RuntimeError("execution receipt semantic replay failed")
        final_guard(); lock.require_owned()
        if path.exists():
            raise RuntimeError("execution receipt namespace is spent")
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_binding(authority: execution.ExecutionAuthority) -> None:
    """Replay exact live and committed bytes from the authority's pushed commit."""
    if type(authority) is not execution.ExecutionAuthority or tuple(
        path for path, _digest in authority.source_hashes
    ) != SOURCE_CLOSURE:
        raise RuntimeError("execution source closure path set changed")
    for relative, digest in authority.source_hashes:
        completed = subprocess.run(
            ["git", "show", f"{authority.source_commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        live = ROOT / relative
        if completed.returncode != 0 or hashlib.sha256(completed.stdout).hexdigest() != digest \
                or not live.is_file() or file_sha256(live) != digest:
            raise RuntimeError(f"execution source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", authority.source_commit, "origin/main"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )


def publish_result_receipt_last(
    result: Mapping[str, Any], result_path: Path, receipt_path: Path, *,
    failure_path: Path, terminal_path: Path, authority_sha256: str,
    lock: RunLock, final_guard,
) -> None:
    """Publish an immutable result, then its success receipt as the final artifact."""
    paths = (result_path, receipt_path, failure_path, terminal_path)
    if len(set(paths)) != 4 or any(path.exists() for path in paths) or not (
        isinstance(authority_sha256, str) and len(authority_sha256) == 64
    ):
        raise RuntimeError("execution result namespace/binding is malformed or spent")
    result_bytes = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    result_tmp = result_path.with_name(f".{result_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with result_tmp.open("xb") as sink:
            sink.write(result_bytes); sink.flush(); os.fsync(sink.fileno())
        if hashlib.sha256(result_tmp.read_bytes()).hexdigest() != result_sha256 or (
            json.loads(result_tmp.read_bytes()) != result
        ):
            raise RuntimeError("execution result semantic replay failed")
        final_guard(); lock.require_owned()
        if any(path.exists() for path in paths):
            raise RuntimeError("execution terminal aggregate changed before result")
        os.link(result_tmp, result_path)
        directory = os.open(result_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        result_tmp.unlink(missing_ok=True)
    receipt = {
        "schema": "bracket_closure_execution_v1_receipt",
        "authority_sha256": authority_sha256,
        "result_sha256": result_sha256,
        "status": "complete_promoted" if result.get("promoted") is True else "complete_nonpromotion",
    }
    terminal = {
        "schema": "bracket_closure_execution_v1_terminal_claim",
        "status": "success_claimed_before_receipt",
        "authority_sha256": authority_sha256,
        "result_sha256": result_sha256,
    }
    terminal_bytes = (json.dumps(terminal, sort_keys=True, indent=2) + "\n").encode()
    terminal_tmp = terminal_path.with_name(
        f".{terminal_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        with terminal_tmp.open("xb") as sink:
            sink.write(terminal_bytes); sink.flush(); os.fsync(sink.fileno())
        final_guard(); lock.require_owned()
        if not result_path.is_file() or file_sha256(result_path) != result_sha256 or any(
            path.exists() for path in (receipt_path, failure_path, terminal_path)
        ):
            raise RuntimeError("execution terminal aggregate changed before success claim")
        os.link(terminal_tmp, terminal_path)
    finally:
        terminal_tmp.unlink(missing_ok=True)
    terminal_sha256 = hashlib.sha256(terminal_bytes).hexdigest()
    publish_json_receipt_last(
        receipt, receipt_path, lock=lock,
        final_guard=lambda: (
            final_guard(),
            (_ for _ in ()).throw(RuntimeError("execution result changed before receipt"))
            if file_sha256(result_path) != result_sha256 else None,
            (_ for _ in ()).throw(RuntimeError("execution failure appeared before receipt"))
            if failure_path.exists() else None,
            (_ for _ in ()).throw(RuntimeError("execution terminal claim changed"))
            if file_sha256(terminal_path) != terminal_sha256 else None,
        ),
    )


def publish_failure_receipt_last(
    payload: Mapping[str, Any], failure_path: Path, *, result_path: Path,
    receipt_path: Path, terminal_path: Path, lock: RunLock, final_guard,
) -> None:
    """Atomically win the common terminal claim, then publish one bound failure."""
    if any(path.exists() for path in (receipt_path, failure_path, terminal_path)):
        raise RuntimeError("execution terminal aggregate is already spent")
    result_present = result_path.is_file()
    result_sha256 = file_sha256(result_path) if result_present else None
    aggregate = {"result_present": result_present, "result_sha256": result_sha256,
                 "receipt_present": False, "failure_present": False}
    failure = dict(payload)
    if "aggregate" in failure:
        raise RuntimeError("failure caller cannot supply terminal aggregate")
    failure["aggregate"] = aggregate
    terminal = {
        "schema": "bracket_closure_execution_v1_terminal_claim",
        "status": "failure_claimed_before_failure_receipt",
        "authority_sha256": failure.get("authority_sha256"),
        "aggregate": aggregate,
    }
    terminal_bytes = (json.dumps(terminal, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = terminal_path.with_name(
        f".{terminal_path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("xb") as sink:
            sink.write(terminal_bytes); sink.flush(); os.fsync(sink.fileno())
        final_guard(); lock.require_owned()
        current_result = result_path.is_file()
        if current_result != result_present or (
            current_result and file_sha256(result_path) != result_sha256
        ) or any(path.exists() for path in (receipt_path, failure_path, terminal_path)):
            raise RuntimeError("execution terminal aggregate changed before failure claim")
        os.link(temporary, terminal_path)
    finally:
        temporary.unlink(missing_ok=True)
    terminal_sha256 = hashlib.sha256(terminal_bytes).hexdigest()
    publish_json_receipt_last(
        failure, failure_path, lock=lock,
        final_guard=lambda: (
            final_guard(),
            (_ for _ in ()).throw(RuntimeError("success receipt appeared before failure"))
            if receipt_path.exists() else None,
            (_ for _ in ()).throw(RuntimeError("failure terminal claim changed"))
            if file_sha256(terminal_path) != terminal_sha256 else None,
            (_ for _ in ()).throw(RuntimeError("partial result aggregate changed"))
            if result_path.is_file() != result_present or (
                result_present and file_sha256(result_path) != result_sha256
            ) else None,
        ),
    )


def run_transaction(authority: execution.ExecutionAuthority, backend) -> None:
    """Preflight boundary: a concrete source-closed backend must own later I/O."""
    execution.require_launch_ready(authority)
    verify_source_binding(authority)
    if backend is not None:
        raise RuntimeError(
            "generic backend injection is prohibited; use execute_loaded_roles from a "
            "concrete source-closed adapter"
        )
    raise RuntimeError("bracket execution concrete loader/publisher adapter is not yet source-closed")


__all__ = (
    "RunLock", "SOURCE_CLOSURE", "file_sha256", "publish_json_receipt_last",
    "publish_failure_receipt_last", "publish_result_receipt_last", "run_transaction",
    "verify_source_binding",
)
