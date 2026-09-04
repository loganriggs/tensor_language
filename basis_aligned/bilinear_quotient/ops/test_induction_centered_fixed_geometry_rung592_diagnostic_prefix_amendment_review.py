#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free attacks on the exact R592 diagnostic-prefix amendment."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
PARENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_AMENDMENT.md"
BLOCK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_AMENDMENT_INDEPENDENT_REVIEW.md"

REVIEWED_COMMIT = "3be7c21c3886502ea989efdaeba5c137aef45d8e"
AMENDMENT_REPO_PATH = (
    "basis_aligned/polynomial_causal/"
    "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
)
EXPECTED_HASHES = {
    AMENDMENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    PARENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    BLOCK_REVIEW: "21bdc310b4798d3ae6d47fc2ed7dfee969afd871bc90db381db634e2c4cae2f5",
}

ARM_ORDER = ("native", "replay", "score", "payload", "joint")
PREDICATE_ORDER = (
    "nonfinite_observation",
    "fixed_width_token_manifest_failed",
    "native_full_write_reconstruction_failed",
    "native_equality_remainder_reconstruction_failed",
    "factor_transport_failed",
    "centered_hook_delta_failed",
    "directed_native_zero_replay_failed",
    "structural_output_identity_failed",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def call(call_id: str, kind: str, chunk: int, batch: int) -> dict[str, object]:
    return {
        "call_id": call_id,
        "kind": kind,
        "chunk": chunk,
        "batch": batch,
        "width": 30,
    }


def toy_manifest(*, tail_batch: int = 16) -> list[dict[str, object]]:
    records = [call("SELECT:endpoint:0000", "endpoint", 0, 32)]
    for chunk, batch in ((0, 32), (1, tail_batch)):
        records.extend(
            call(f"SELECT:directed:{chunk:04d}:{arm}", arm, chunk, batch)
            for arm in ARM_ORDER
        )
    return records


def directory_name(index: int, record: dict[str, object]) -> str:
    return f"{index:04d}_{record['call_id']}"


def validate_prefix(
    manifest: list[dict[str, object]],
    records: list[dict[str, object]],
    directories: list[str],
) -> None:
    if records != manifest[: len(records)]:
        raise ValueError("call records are not an exact frozen manifest prefix")
    expected_directories = [directory_name(i, row) for i, row in enumerate(records)]
    if directories != expected_directories:
        raise ValueError("call directories do not exactly equal the completed prefix")
    for row in records:
        if row["batch"] not in (16, 32) or row["width"] != 30:
            raise ValueError("wrong physical tensor shape")


def mandatory_shapes(record: dict[str, object]) -> dict[str, tuple[int, ...]]:
    batch = int(record["batch"])
    kind = str(record["kind"])
    common = {"tokens.npy": (batch, 30), "logits.npy": (batch, 50_257)}
    if kind == "endpoint":
        return common | {
            "factor_e.npy": (batch, 4, 2),
            "factor_u.npy": (batch, 4, 2, 1_152),
            "support.npy": (batch, 4, 2),
        }
    if kind == "native":
        return common | {
            "live_e.npy": (batch, 4, 2),
            "live_u.npy": (batch, 4, 2, 1_152),
        }
    return common | {
        "hook_deltas.npy": (batch, 4, 1_152),
        "planned_hook_deltas.npy": (batch, 4, 1_152),
    }


def first_failure(failures: set[str]) -> str:
    return next(predicate for predicate in PREDICATE_ORDER if predicate in failures)


def validate_single_nonfinite_mask(raw: np.ndarray, mask: np.ndarray) -> None:
    if mask.dtype != np.bool_ or mask.shape != raw.shape:
        raise ValueError("nonfinite mask dtype/shape mismatch")
    if not np.array_equal(mask, ~np.isfinite(raw)):
        raise ValueError("nonfinite mask content mismatch")


def recognized_namespace(public_paths: set[str]) -> str | None:
    normal = {
        "normal/evidence",
        "normal/result.json",
        "normal/receipt.json",
    }.issubset(public_paths)
    invalid = {
        "invalid/evidence",
        "invalid/diagnostic.json",
        "invalid/receipt.json",
    }.issubset(public_paths)
    if normal and invalid:
        raise ValueError("normal and invalid namespaces are mutually exclusive")
    if normal:
        return "normal"
    if invalid:
        return "invalid"
    return None


def test_exact_commit_blob_and_pinned_parent_review_hashes() -> None:
    assert {path: sha256(path.read_bytes()) for path in EXPECTED_HASHES} == EXPECTED_HASHES
    blob = subprocess.check_output(
        ["git", "show", f"{REVIEWED_COMMIT}:{AMENDMENT_REPO_PATH}"],
        cwd=ROOT.parents[1],
    )
    assert blob == AMENDMENT.read_bytes()
    text = AMENDMENT.read_text()
    assert EXPECTED_HASHES[PARENT] in text
    assert EXPECTED_HASHES[BLOCK_REVIEW] in text


@pytest.mark.parametrize("failing_arm", ARM_ORDER)
def test_failure_after_each_directed_arm_is_an_exact_unpadded_prefix(failing_arm: str) -> None:
    manifest = toy_manifest()
    terminal_index = next(
        i
        for i, row in enumerate(manifest)
        if row["chunk"] == 0 and row["kind"] == failing_arm
    )
    records = manifest[: terminal_index + 1]
    directories = [directory_name(i, row) for i, row in enumerate(records)]
    validate_prefix(manifest, records, directories)
    assert records[-1]["kind"] == failing_arm
    assert all(row["chunk"] != 1 for row in records if row["kind"] != "endpoint")
    later_arms = ARM_ORDER[ARM_ORDER.index(failing_arm) + 1 :]
    assert not any(
        row["chunk"] == 0 and row["kind"] in later_arms for row in records
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "nonprefix"))
def test_missing_extra_and_nonprefix_directories_are_rejected(mutation: str) -> None:
    manifest = toy_manifest()
    records = manifest[:4]
    directories = [directory_name(i, row) for i, row in enumerate(records)]
    if mutation == "missing":
        directories.pop()
    elif mutation == "extra":
        directories.append("0004_SELECT:directed:0000:payload")
    else:
        records[-1] = manifest[4]
        directories[-1] = directory_name(3, records[-1])
    with pytest.raises(ValueError, match="prefix|directories"):
        validate_prefix(manifest, records, directories)


def test_select_tail_is_literal_batch_16_with_no_padding() -> None:
    manifest = toy_manifest()
    tail = manifest[-1]
    shapes = mandatory_shapes(tail)
    assert tail["batch"] == 16
    assert shapes == {
        "tokens.npy": (16, 30),
        "logits.npy": (16, 50_257),
        "hook_deltas.npy": (16, 4, 1_152),
        "planned_hook_deltas.npy": (16, 4, 1_152),
    }
    planted = dict(tail, batch=32)
    with pytest.raises(ValueError, match="prefix"):
        validate_prefix(manifest, manifest[:-1] + [planted], [
            directory_name(i, row) for i, row in enumerate(manifest[:-1] + [planted])
        ])


def test_one_nonfinite_array_has_exact_mask_semantics_and_bad_mask_fails() -> None:
    raw = np.array([[0.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    correct = ~np.isfinite(raw)
    validate_single_nonfinite_mask(raw, correct)
    wrong = correct.copy()
    wrong[0, 0] = True
    with pytest.raises(ValueError, match="content"):
        validate_single_nonfinite_mask(raw, wrong)
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_single_nonfinite_mask(raw, correct.astype(np.uint8))


def test_predicate_precedence_is_total_and_independent_of_report_order() -> None:
    planted = {
        "structural_output_identity_failed",
        "centered_hook_delta_failed",
        "nonfinite_observation",
    }
    assert first_failure(planted) == "nonfinite_observation"
    planted.remove("nonfinite_observation")
    assert first_failure(planted) == "centered_hook_delta_failed"
    assert first_failure(set(reversed(PREDICATE_ORDER))) == "nonfinite_observation"


def test_hard_abort_leaves_no_recognized_public_namespace() -> None:
    staged = {".stage/evidence", ".stage/diagnostic.json", ".stage/receipt.json"}
    public: set[str] = set()
    forward_completed = False
    if not forward_completed:
        # A killed/raised call may leave temporary bytes, but none may be renamed.
        assert staged
        assert recognized_namespace(public) is None


def test_receipt_last_and_diagnostic_normal_separation() -> None:
    partial_invalid = {"invalid/evidence", "invalid/diagnostic.json"}
    assert recognized_namespace(partial_invalid) is None
    complete_invalid = partial_invalid | {"invalid/receipt.json"}
    assert recognized_namespace(complete_invalid) == "invalid"
    complete_normal = {"normal/evidence", "normal/result.json", "normal/receipt.json"}
    assert recognized_namespace(complete_normal) == "normal"
    with pytest.raises(ValueError, match="mutually exclusive"):
        recognized_namespace(complete_invalid | complete_normal)


def test_invalid_diagnostic_cannot_be_interpreted_as_scientific_result() -> None:
    forbidden = {
        "split_scores",
        "scientific_terminal",
        "bootstrap_interval",
        "held",
        "null",
    }
    diagnostic = {
        "status": "invalid_diagnostic",
        "failure_predicate": "centered_hook_delta_failed",
        "executed_call_prefix": ["FIT:endpoint:0000"],
    }
    assert forbidden.isdisjoint(diagnostic)
    planted = diagnostic | {"split_scores": {"FIT": {}}}
    with pytest.raises(ValueError, match="scientific field"):
        if not forbidden.isdisjoint(planted):
            raise ValueError("invalid diagnostic contains a scientific field")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the exact amendment assigns the same flat filename nonfinite_mask.npy "
        "to every affected raw array"
    ),
)
def test_multiple_nonfinite_raw_arrays_have_unique_mask_paths() -> None:
    # More than one affected array is realistic: a nonfinite planned delta can
    # propagate to the observed hook delta and logits in the same completed arm.
    affected = ("hook_deltas.npy", "planned_hook_deltas.npy", "logits.npy")
    amendment_text = AMENDMENT.read_text()
    assert "for each affected raw array" in amendment_text
    literal_mask_path = "nonfinite_mask.npy"
    mask_paths = {raw_name: literal_mask_path for raw_name in affected}
    assert len(set(mask_paths.values())) == len(affected)

