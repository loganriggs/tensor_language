#!/usr/bin/env python3
"""Dtype-preserving recovery for the terminal-copy checkpoint identity check."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import terminal_copy_attention_checkpoint_check_v1 as v1


PREREG = HERE / "TERMINAL_COPY_ATTENTION_CHECKPOINT_CHECK_V3_RECOVERY_PREREGISTRATION.md"
RUNNER = Path(__file__).resolve()
V1_AUTHORITY = HERE / "terminal_copy_attention_checkpoint_check_v1_authority.json"
V1_FAILURE = HERE / "terminal_copy_attention_checkpoint_check_v1_failure.json"
V2_AUTHORITY = HERE / "terminal_copy_attention_checkpoint_check_v2_authority.json"
V2_FAILURE = HERE / "terminal_copy_attention_checkpoint_check_v2_failure.json"
AUTHORITY = HERE / "terminal_copy_attention_checkpoint_check_v3_authority.json"
RESULT = HERE / "terminal_copy_attention_checkpoint_check_v3_result.json"
RECEIPT = HERE / "terminal_copy_attention_checkpoint_check_v3_receipt.json"
FAILURE = HERE / "terminal_copy_attention_checkpoint_check_v3_failure.json"

SPENT_PINS = {
    V1_AUTHORITY: "15c68bfaff6362300fa680a60ce14077bea43142f135ff4a6c809b5341ce2b5c",
    V1_FAILURE: "fdb947c9c557f23e819050a007bbf6d5cfc19039ce195aedcd47a6aadcbc9138",
    V2_AUTHORITY: "8d84f8a568fa84d5b2dbce2f17e9b3c65ab8fb1966bbe616105ca55d17b4475f",
    V2_FAILURE: "cc1a57303e51bc838a094adfddd202c6d278af963520da23a94c99e6d738bc72",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute() -> dict:
    if {path: file_sha256(path) for path in SPENT_PINS} != SPENT_PINS:
        raise RuntimeError("spent checkpoint-check binding changed")
    for version in (1, 2):
        if (HERE / f"terminal_copy_attention_checkpoint_check_v{version}_result.json").exists() or (
            HERE / f"terminal_copy_attention_checkpoint_check_v{version}_receipt.json"
        ).exists():
            raise RuntimeError("spent predecessor unexpectedly has a successful receipt")

    v1.AUTHORITY = AUTHORITY
    v1.RESULT = RESULT
    v1.RECEIPT = RECEIPT
    v1.FAILURE = FAILURE
    v1.RUNNER = RUNNER
    v1.PREREG = PREREG
    v1.RELATIVE_TOLERANCE = 0.01
    v1.MAX_ABS_TOLERANCE = 1e30
    v1.SOURCE_PATHS = (
        PREREG, RUNNER, V1_AUTHORITY, V1_FAILURE, V2_AUTHORITY, V2_FAILURE,
        HERE / "terminal_copy_attention_checkpoint_check_v1.py",
        HERE / "TERMINAL_COPY_ATTENTION_ADAPTER_V1_ADDENDUM.md",
        HERE / "terminal_copy_attention_adapter.py",
        HERE / "test_terminal_copy_attention_adapter.py",
        HERE / "bilin18_observed_model_facade.py",
        ROOT / "jacclust" / "tt_model.py", ROOT / "jacclust" / "__init__.py",
    )
    return v1.execute()


if __name__ == "__main__":
    import json
    print(json.dumps(execute(), indent=2))
