#!/usr/bin/env python3
"""Implementation-only retry of the frozen induction FINAL/OOD v2 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

import induction_equality_tensor_final_ood_v2 as v2


HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_V2_RETRY1_AMENDMENT.md"
AUDIT = HERE / "induction_equality_tensor_final_ood_v2_retry1_independent_audit.json"
AUTHORITY = HERE / "induction_equality_tensor_final_ood_v2_retry1_authority.json"
LEDGER = HERE / "induction_equality_tensor_final_ood_v2_retry1_ledger.json"
RESULT = HERE / "induction_equality_tensor_final_ood_v2_retry1_result.json"
MANIFEST = HERE / "induction_equality_tensor_final_ood_v2_retry1_manifest.json"
RECEIPT = HERE / "induction_equality_tensor_final_ood_v2_retry1_receipt.json"
FAILURE = HERE / "induction_equality_tensor_final_ood_v2_retry1_failure.json"
LOCK = Path("/workspace/runs/.induction_equality_tensor_final_ood_v2_retry1.lock")
ORIGINAL_AUDIT = HERE / "induction_equality_tensor_final_ood_v2_independent_audit.json"
ORIGINAL_AUTHORITY = HERE / "induction_equality_tensor_final_ood_v2_authority.json"
ORIGINAL_FAILURE = HERE / "induction_equality_tensor_final_ood_v2_failure.json"
ORIGINAL_REQUIRED = (ORIGINAL_AUDIT, ORIGINAL_AUTHORITY, ORIGINAL_FAILURE)
ORIGINAL_ABSENT = tuple(
    HERE / f"induction_equality_tensor_final_ood_v2_{suffix}.json"
    for suffix in ("ledger", "result", "manifest", "receipt")
)
_base_protected_snapshot = v2.protected_snapshot


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash exact state bytes, including zero-dimensional bf16 tensors."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _lineage_snapshot() -> dict[str, str | None]:
    for path in ORIGINAL_REQUIRED:
        if not path.is_file():
            raise RuntimeError(f"required original v2 lineage artifact missing: {path.name}")
    for path in ORIGINAL_ABSENT:
        if path.exists():
            raise RuntimeError(f"spent original v2 namespace was reused: {path.name}")
    return {
        str(path.resolve()): v2.file_sha256(path) if path.is_file() else None
        for path in (*ORIGINAL_REQUIRED, *ORIGINAL_ABSENT)
    }


def protected_snapshot() -> dict[str, str | None]:
    snapshot = dict(_base_protected_snapshot())
    lineage = _lineage_snapshot()
    overlap = set(snapshot) & set(lineage)
    if overlap and any(snapshot[path] != lineage[path] for path in overlap):
        raise RuntimeError("original v2 lineage snapshot disagrees with source closure")
    snapshot.update(lineage)
    return snapshot


def _configure() -> None:
    """Give the unchanged v2 transaction a fresh, source-closed namespace."""

    v2.AUDIT = AUDIT
    v2.AUTHORITY = AUTHORITY
    v2.LEDGER = LEDGER
    v2.RESULT = RESULT
    v2.MANIFEST = MANIFEST
    v2.RECEIPT = RECEIPT
    v2.FAILURE = FAILURE
    v2.LOCK = LOCK
    v2.OUTPUTS = (AUTHORITY, LEDGER, RESULT, MANIFEST, RECEIPT, FAILURE)
    v2.model_state_sha256 = model_state_sha256
    v2.SOURCE_PATHS = tuple(dict.fromkeys((
        *v2.SOURCE_PATHS,
        *ORIGINAL_REQUIRED,
        Path(__file__).resolve(),
        HERE / "test_induction_equality_tensor_final_ood_v2_retry1.py",
        AMENDMENT,
    )))
    v2.protected_snapshot = protected_snapshot


_configure()

freeze_authority = v2.freeze_authority
validate_authority = v2.validate_authority
execute = v2.execute


if __name__ == "__main__":
    import json

    print(json.dumps(execute(), indent=2))
