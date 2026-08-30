"""Receipt-last lifecycle shell for bracket execution; launch is prospectively NO-GO."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
from typing import Any, Mapping

import bracket_closure_execution_v1 as execution


SOURCE_CLOSURE = (
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_CANARY_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_ROWS_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_NO_GO.md",
    "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bracket_closure_canary_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_execution_lifecycle_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_masks_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_rows_v1.py",
    "basis_aligned/polynomial_causal/bracket_closure_tensor_v1.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_execution_v1.py",
    "basis_aligned/polynomial_causal/test_bracket_closure_execution_lifecycle_v1.py",
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
        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino, self.path.read_text()) != self.claim:
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


def run_transaction(authority: execution.ExecutionAuthority, backend) -> None:
    """Fail before touching a backend while inference is unresolved."""
    execution.require_launch_ready(authority)
    raise AssertionError("unreachable until a prospective inference ruling exists")


__all__ = ("RunLock", "SOURCE_CLOSURE", "publish_json_receipt_last", "run_transaction")
