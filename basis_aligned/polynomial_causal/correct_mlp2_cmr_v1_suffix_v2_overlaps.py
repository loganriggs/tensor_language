#!/usr/bin/env python3
"""Receipt-last CPU correction for the MLP2 SUFFIX v2 support-overlap summary."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import mlp2_cmr_v1_suffix_math as suffix_math


ADDENDUM = HERE / "MLP2_CMR_V1_SUFFIX_V2_OVERLAP_CORRECTION.md"
ORIGINAL_BUNDLE = HERE / "mlp2_cmr_v1_suffix_v2_bundle.pt"
ORIGINAL_RESULT = HERE / "mlp2_cmr_v1_suffix_v2_result.json"
ORIGINAL_RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_receipt.json"
OUT = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction.json"
RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction_receipt.json"

ORIGINAL_BUNDLE_SHA256 = "cb3f8d3caecab86881eba825785cabd58c1b7ac8e2aa1eb93b459168cff17ce1"
ORIGINAL_RESULT_SHA256 = "ab08dc0f0a71b5daf21228991b9e78a272aa74d226d97189ac414a546dc16f62"
ORIGINAL_RECEIPT_SHA256 = "b61c7308409ec64dc05601206bda21e1f4e24097871ba8dff0c92bc84e761e1f"
SUPPORT_NAMES = ("SUFFIX", "DERANGED", "LOCAL", "RMS", "MASS", "HASH_RANDOM")
SOURCE_CLOSURE = (
    ADDENDUM, Path(__file__).resolve(),
    HERE / "test_correct_mlp2_cmr_v1_suffix_v2_overlaps.py",
    HERE / "mlp2_cmr_v1_suffix_math.py",
    HERE / "test_mlp2_cmr_v1_suffix_math.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write_create_only(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"overlap-correction source differs from commit: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def corrected_overlaps(supports: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    if tuple(supports) != SUPPORT_NAMES or any(
        not torch.is_tensor(value) or value.shape != (512,)
        or len(set(map(int, value.tolist()))) != 512 for value in supports.values()
    ):
        raise ValueError("selector support bundle is malformed")
    result = {}
    for left, right in itertools.combinations(SUPPORT_NAMES, 2):
        first = set(map(int, supports[left].tolist()))
        second = set(map(int, supports[right].tolist()))
        intersection = len(first & second)
        result[f"{left}_{right}"] = {
            "intersection": intersection,
            "union": len(first | second),
            "jaccard": suffix_math.support_jaccard(supports[left], supports[right]),
        }
    return result


def main() -> None:
    if OUT.exists() or RECEIPT.exists():
        raise RuntimeError("overlap-correction namespace is create-only")
    parents = {
        "bundle": file_sha256(ORIGINAL_BUNDLE),
        "result": file_sha256(ORIGINAL_RESULT),
        "receipt": file_sha256(ORIGINAL_RECEIPT),
    }
    if parents != {
        "bundle": ORIGINAL_BUNDLE_SHA256,
        "result": ORIGINAL_RESULT_SHA256,
        "receipt": ORIGINAL_RECEIPT_SHA256,
    }:
        raise RuntimeError("original MLP2 suffix artifacts changed")
    commit, source_hashes = committed_source()
    original_result = json.loads(ORIGINAL_RESULT.read_text())
    original_receipt = json.loads(ORIGINAL_RECEIPT.read_text())
    if set(original_result["support_overlaps"].values()) != {0.0} or (
        original_receipt.get("authorized_for_validation") is not True
        or original_receipt.get("authorized_for_replication") is not False
    ):
        raise RuntimeError("original overlap discrepancy or authority changed")
    bundle = torch.load(ORIGINAL_BUNDLE, map_location="cpu", weights_only=True)
    supports = bundle["supports"]
    expected_hashes = original_result["tensor_hashes"]["supports"]
    actual_hashes = {name: tensor_sha256(supports[name]) for name in SUPPORT_NAMES}
    if actual_hashes != expected_hashes:
        raise RuntimeError("support tensors do not replay original result")
    overlaps = corrected_overlaps(supports)
    result = {
        "schema_version": 1,
        "experiment_id": "bilin18_mlp2_cmr_v1_suffix_v2_overlap_correction",
        "status": "corrected_support_overlap_summary_no_model_access",
        "parents": parents,
        "source_commit": commit,
        "source_hashes": source_hashes,
        "support_hashes": actual_hashes,
        "corrected_support_overlaps": overlaps,
        "original_all_zero_summary_invalid": True,
        "selector_scores_or_supports_changed": False,
        "validation_opened": False,
        "replication_opened": False,
    }
    write_create_only(OUT, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    receipt = {
        "schema_version": 1,
        "experiment_id": result["experiment_id"],
        "status": "overlap_correction_complete_receipt_last",
        "result_sha256": file_sha256(OUT),
        "parents": parents,
        "source_commit": commit,
        "source_hashes": source_hashes,
        "authorized_for_validation_with_original_selector_receipt": True,
        "authorized_for_replication": False,
        "supersedes_only": "mlp2_cmr_v1_suffix_v2_result.json:support_overlaps",
    }
    write_create_only(RECEIPT, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

