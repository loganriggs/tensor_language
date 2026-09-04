#!/usr/bin/env python3
# BQLANE: cpu
"""Filesystem-level attacks on the exact R592 nonfinite-mask amendment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT.md"
PARENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
BLOCK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT_INDEPENDENT_REVIEW.md"

REVIEWED_COMMIT = "6d779ae45b68ffa4c3e7bdf58963cb7f7c2ed2d2"
AMENDMENT_REPO_PATH = (
    "basis_aligned/polynomial_causal/"
    "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT.md"
)
EXPECTED_HASHES = {
    AMENDMENT: "f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc",
    PARENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    BLOCK_REVIEW: "e7373c2249e0456327d386559d4f3fa68e0661ed076a35fb120ad9d8effaa675",
}
INDEX_FIELDS = {
    "raw_filename",
    "mask_filename",
    "raw_dtype",
    "mask_dtype",
    "shape",
    "mask_byte_length",
    "mask_sha256",
    "nonfinite_count",
    "first_lexicographic_coordinate",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_mask_name(raw_filename: str) -> str:
    raw_path = PurePosixPath(raw_filename)
    if raw_path.is_absolute() or len(raw_path.parts) != 1 or raw_path.suffix != ".npy":
        raise ValueError("raw filename is not one safe mandatory array name")
    if raw_path.name in (".", "..") or ".." in raw_path.parts:
        raise ValueError("raw filename traversal")
    return f"nonfinite_masks/{raw_path.stem}.mask.npy"


def first_true_coordinate(mask: np.ndarray) -> list[int]:
    flat_index = int(np.flatnonzero(mask.ravel(order="C"))[0])
    return [int(x) for x in np.unravel_index(flat_index, mask.shape, order="C")]


def write_raw_arrays(call_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    call_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in arrays.items():
        np.save(call_dir / filename, array, allow_pickle=False)


def write_canonical_masks(call_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    mask_dir = call_dir / "nonfinite_masks"
    for raw_path in sorted(call_dir.glob("*.npy"), key=lambda path: path.name):
        raw = np.load(raw_path, allow_pickle=False)
        if raw.dtype.kind != "f" or np.isfinite(raw).all():
            continue
        mask = np.asarray(~np.isfinite(raw), dtype=np.bool_, order="C")
        mask_filename = canonical_mask_name(raw_path.name)
        path = call_dir / mask_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask, allow_pickle=False)
        entries.append(
            {
                "raw_filename": raw_path.name,
                "mask_filename": mask_filename,
                "raw_dtype": str(raw.dtype),
                "mask_dtype": "bool",
                "shape": list(raw.shape),
                "mask_byte_length": int(mask.nbytes),
                "mask_sha256": sha256_file(path),
                "nonfinite_count": int(mask.sum()),
                "first_lexicographic_coordinate": first_true_coordinate(mask),
            }
        )
    assert entries and mask_dir.is_dir()
    (call_dir / "nonfinite_mask_index.json").write_text(
        json.dumps(entries, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return entries


def validate_masks(call_dir: Path, terminal: str) -> None:
    index_path = call_dir / "nonfinite_mask_index.json"
    mask_dir = call_dir / "nonfinite_masks"
    flat_mask = call_dir / "nonfinite_mask.npy"
    if flat_mask.exists():
        raise ValueError("flat nonfinite mask path is forbidden")
    if terminal != "nonfinite_observation":
        if index_path.exists() or mask_dir.exists():
            raise ValueError("mask artifacts under non-nonfinite terminal")
        return
    if not index_path.is_file() or not mask_dir.is_dir():
        raise ValueError("nonfinite terminal requires index and mask directory")

    entries = json.loads(index_path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("mask index must be a nonempty array")
    if entries != sorted(entries, key=lambda entry: entry["raw_filename"]):
        raise ValueError("mask index is not lexicographically sorted")
    if any(set(entry) != INDEX_FIELDS for entry in entries):
        raise ValueError("wrong exact index fields")
    raw_names = [entry["raw_filename"] for entry in entries]
    mask_names = [entry["mask_filename"] for entry in entries]
    if len(set(raw_names)) != len(raw_names) or len(set(mask_names)) != len(mask_names):
        raise ValueError("duplicate raw or mask path")

    expected_nonfinite: set[str] = set()
    for raw_path in call_dir.glob("*.npy"):
        raw = np.load(raw_path, allow_pickle=False)
        if raw.dtype.kind == "f" and not np.isfinite(raw).all():
            expected_nonfinite.add(raw_path.name)
    if set(raw_names) != expected_nonfinite:
        raise ValueError("mask index does not exactly equal nonfinite float arrays")

    expected_mask_files: set[str] = set()
    for entry in entries:
        raw_name = entry["raw_filename"]
        expected_mask_name = canonical_mask_name(raw_name)
        if entry["mask_filename"] != expected_mask_name:
            raise ValueError("mask path is not deterministic")
        pure_mask_path = PurePosixPath(entry["mask_filename"])
        if pure_mask_path.is_absolute() or ".." in pure_mask_path.parts:
            raise ValueError("mask path traversal")
        expected_mask_files.add(pure_mask_path.name)

        raw = np.load(call_dir / raw_name, allow_pickle=False)
        mask_path = call_dir / entry["mask_filename"]
        if not mask_path.is_file():
            raise ValueError("missing mask")
        mask = np.load(mask_path, allow_pickle=False)
        if not mask.flags.c_contiguous:
            raise ValueError("mask is not C-contiguous")
        if entry["raw_dtype"] != str(raw.dtype) or entry["mask_dtype"] != "bool":
            raise ValueError("wrong dtype metadata")
        if mask.dtype != np.bool_:
            raise ValueError("mask file is not NumPy bool")
        if list(mask.shape) != list(raw.shape) or entry["shape"] != list(raw.shape):
            raise ValueError("wrong mask shape")
        if (
            entry["mask_byte_length"] != mask.nbytes
            or mask.nbytes != int(np.prod(mask.shape))
        ):
            raise ValueError("wrong mask byte length")
        if entry["mask_sha256"] != sha256_file(mask_path):
            raise ValueError("wrong mask content hash")
        expected_mask = ~np.isfinite(raw)
        if not np.array_equal(mask, expected_mask):
            raise ValueError("wrong mask content")
        if entry["nonfinite_count"] != int(mask.sum()) or int(mask.sum()) <= 0:
            raise ValueError("wrong nonfinite count")
        if entry["first_lexicographic_coordinate"] != first_true_coordinate(mask):
            raise ValueError("wrong first coordinate")

    observed_mask_files = {path.name for path in mask_dir.iterdir() if path.is_file()}
    if observed_mask_files != expected_mask_files:
        raise ValueError("missing or extra mask file")


def fixture_arrays(*, three_nonfinite: bool = False) -> dict[str, np.ndarray]:
    arrays = {
        "tokens.npy": np.arange(60, dtype=np.int64).reshape(2, 30),
        "support.npy": np.ones((2, 4, 2), dtype=np.bool_),
        "logits.npy": np.array([[0.0, np.nan, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "hook_deltas.npy": np.array(
            [[[0.0, 1.0], [np.inf, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
            dtype=np.float32,
        ),
        "planned_hook_deltas.npy": np.zeros((2, 2, 2), dtype=np.float32),
    }
    if three_nonfinite:
        arrays["planned_hook_deltas.npy"][1, 1, 0] = -np.inf
    return arrays


def test_exact_commit_blob_and_parent_review_pins() -> None:
    assert {path: sha256_file(path) for path in EXPECTED_HASHES} == EXPECTED_HASHES
    blob = subprocess.check_output(
        ["git", "show", f"{REVIEWED_COMMIT}:{AMENDMENT_REPO_PATH}"],
        cwd=ROOT.parents[1],
    )
    assert blob == AMENDMENT.read_bytes()
    text = AMENDMENT.read_text()
    assert REVIEWED_COMMIT in subprocess.check_output(
        ["git", "rev-parse", f"{REVIEWED_COMMIT}^{{}}"], cwd=ROOT.parents[1], text=True
    )
    assert "3be7c21c3886502ea989efdaeba5c137aef45d8e" in text
    assert EXPECTED_HASHES[BLOCK_REVIEW] in text


@pytest.mark.parametrize("three_nonfinite", (False, True))
def test_two_and_three_nonfinite_arrays_get_distinct_exact_masks(
    tmp_path: Path, three_nonfinite: bool
) -> None:
    write_raw_arrays(tmp_path, fixture_arrays(three_nonfinite=three_nonfinite))
    entries = write_canonical_masks(tmp_path)
    validate_masks(tmp_path, "nonfinite_observation")
    expected_count = 3 if three_nonfinite else 2
    assert len(entries) == expected_count
    assert len({entry["mask_filename"] for entry in entries}) == expected_count
    assert [entry["raw_filename"] for entry in entries] == sorted(
        entry["raw_filename"] for entry in entries
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("flat", "flat"),
        ("duplicate_path", "duplicate"),
        ("missing", "missing"),
        ("extra_finite", "exactly equal"),
        ("wrong_shape", "shape"),
        ("wrong_length", "byte length"),
        ("wrong_count", "count"),
        ("wrong_first", "first coordinate"),
        ("wrong_hash", "content hash"),
        ("wrong_content", "content"),
    ),
)
def test_required_mask_mutations_are_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    write_raw_arrays(tmp_path, fixture_arrays(three_nonfinite=True))
    entries = write_canonical_masks(tmp_path)
    if mutation == "flat":
        np.save(tmp_path / "nonfinite_mask.npy", np.ones((1,), dtype=np.bool_))
    elif mutation == "duplicate_path":
        entries[1]["mask_filename"] = entries[0]["mask_filename"]
    elif mutation == "missing":
        (tmp_path / entries[0]["mask_filename"]).unlink()
    elif mutation == "extra_finite":
        extra = np.zeros((2, 4, 2), dtype=np.bool_)
        extra_path = tmp_path / "nonfinite_masks/support.mask.npy"
        np.save(extra_path, extra)
        entries.append(
            {
                "raw_filename": "support.npy",
                "mask_filename": "nonfinite_masks/support.mask.npy",
                "raw_dtype": "bool",
                "mask_dtype": "bool",
                "shape": [2, 4, 2],
                "mask_byte_length": extra.nbytes,
                "mask_sha256": sha256_file(extra_path),
                "nonfinite_count": 1,
                "first_lexicographic_coordinate": [0, 0, 0],
            }
        )
        entries.sort(key=lambda entry: entry["raw_filename"])
    elif mutation == "wrong_shape":
        entries[0]["shape"] = [999]
    elif mutation == "wrong_length":
        entries[0]["mask_byte_length"] = int(entries[0]["mask_byte_length"]) + 1
    elif mutation == "wrong_count":
        entries[0]["nonfinite_count"] = int(entries[0]["nonfinite_count"]) + 1
    elif mutation == "wrong_first":
        entries[0]["first_lexicographic_coordinate"] = [0] * len(entries[0]["shape"])
    elif mutation == "wrong_hash":
        entries[0]["mask_sha256"] = "0" * 64
    elif mutation == "wrong_content":
        mask_path = tmp_path / entries[0]["mask_filename"]
        mask = np.load(mask_path, allow_pickle=False)
        mask.flat[0] = ~mask.flat[0]
        np.save(mask_path, mask, allow_pickle=False)
        entries[0]["mask_sha256"] = sha256_file(mask_path)
        entries[0]["nonfinite_count"] = int(mask.sum())
        entries[0]["first_lexicographic_coordinate"] = first_true_coordinate(mask)
    (tmp_path / "nonfinite_mask_index.json").write_text(json.dumps(entries) + "\n")
    with pytest.raises(ValueError, match=message):
        validate_masks(tmp_path, "nonfinite_observation")


@pytest.mark.parametrize("bad_path", ("/tmp/escape.mask.npy", "nonfinite_masks/../escape.mask.npy"))
def test_absolute_and_parent_traversing_mask_paths_are_rejected(
    tmp_path: Path, bad_path: str
) -> None:
    write_raw_arrays(tmp_path, fixture_arrays())
    entries = write_canonical_masks(tmp_path)
    entries[0]["mask_filename"] = bad_path
    (tmp_path / "nonfinite_mask_index.json").write_text(json.dumps(entries) + "\n")
    with pytest.raises(ValueError, match="deterministic|traversal"):
        validate_masks(tmp_path, "nonfinite_observation")


@pytest.mark.parametrize(
    "terminal",
    (
        "centered_hook_delta_failed",
        "factor_transport_failed",
        "structural_output_identity_failed",
        "normal_scientific_result",
    ),
)
def test_mask_artifacts_are_forbidden_under_every_non_nonfinite_terminal(
    tmp_path: Path, terminal: str
) -> None:
    write_raw_arrays(tmp_path, fixture_arrays())
    write_canonical_masks(tmp_path)
    with pytest.raises(ValueError, match="non-nonfinite"):
        validate_masks(tmp_path, terminal)


def test_finite_call_has_no_mask_index_or_directory(tmp_path: Path) -> None:
    arrays = fixture_arrays()
    for name, array in arrays.items():
        if array.dtype.kind == "f":
            arrays[name] = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    write_raw_arrays(tmp_path, arrays)
    validate_masks(tmp_path, "centered_hook_delta_failed")
    with pytest.raises(ValueError, match="requires index"):
        validate_masks(tmp_path, "nonfinite_observation")


def test_index_requires_exact_fields_and_lexicographic_order(tmp_path: Path) -> None:
    write_raw_arrays(tmp_path, fixture_arrays())
    entries = write_canonical_masks(tmp_path)
    entries[0]["unfrozen_field"] = 1
    (tmp_path / "nonfinite_mask_index.json").write_text(json.dumps(entries) + "\n")
    with pytest.raises(ValueError, match="exact index fields"):
        validate_masks(tmp_path, "nonfinite_observation")

    entries[0].pop("unfrozen_field")
    entries.reverse()
    (tmp_path / "nonfinite_mask_index.json").write_text(json.dumps(entries) + "\n")
    with pytest.raises(ValueError, match="lexicographically sorted"):
        validate_masks(tmp_path, "nonfinite_observation")
