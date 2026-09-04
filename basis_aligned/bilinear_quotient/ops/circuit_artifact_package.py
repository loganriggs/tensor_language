"""Generic evidence validation and atomic publication for circuit experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from circuit_experiment_spec import canonical_json_bytes


class PackageError(ValueError):
    """Evidence or a staged package violates its compiled contract."""


def _strict_json_loads(payload: bytes, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise PackageError(f"{label} contains nonfinite JSON number: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise PackageError(f"{label} contains duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload, parse_constant=reject_constant, object_pairs_hook=reject_duplicates
        )
        canonical_json_bytes(value)
        return value
    except PackageError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise PackageError(f"{label} is not strict standard JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def call_directory_name(index: int, call_id: str) -> str:
    return f"{index:04d}_{call_id}"


def validate_call_prefix(
    manifest: Sequence[Mapping[str, object]],
    prefix: Sequence[Mapping[str, object]],
    directory_names: Sequence[str],
) -> None:
    """Require literal manifest-prefix records and exactly one directory per call."""
    expected = [dict(record) for record in manifest[: len(prefix)]]
    observed = [dict(record) for record in prefix]
    if observed != expected:
        raise PackageError("observed calls are not an exact compiled manifest prefix")
    expected_directories = [
        call_directory_name(index, str(record["call_id"]))
        for index, record in enumerate(observed)
    ]
    if list(directory_names) != expected_directories:
        raise PackageError("call-directory census differs from the completed prefix")


def first_true_coordinate(mask: np.ndarray) -> list[int]:
    positions = np.flatnonzero(mask.ravel(order="C"))
    if not len(positions):
        raise PackageError("nonfinite mask is empty")
    return [
        int(value)
        for value in np.unravel_index(int(positions[0]), mask.shape, order="C")
    ]


def canonical_mask_filename(raw_filename: str) -> str:
    path = PurePosixPath(raw_filename)
    if path.is_absolute() or len(path.parts) != 1 or path.suffix != ".npy" \
            or path.name in (".", ".."):
        raise PackageError("raw evidence filename is unsafe")
    return f"nonfinite_masks/{path.stem}.mask.npy"


def _raw_arrays(call_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for path in sorted(call_dir.glob("*.npy"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise PackageError("raw evidence array is not a regular file")
        arrays[path.name] = np.load(path, allow_pickle=False)
    return arrays


def _call_record(call_dir: Path) -> dict[str, object]:
    path = call_dir / "call.json"
    if path.is_symlink() or not path.is_file():
        raise PackageError("call evidence lacks its exact saved request")
    value = _strict_json_loads(path.read_bytes(), "saved call request")
    if not isinstance(value, dict):
        raise PackageError("saved call request is not an object")
    return value


def _validate_call_arrays(call_dir: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Bind physical evidence width and typed-arm activity to saved requests."""
    if not (call_dir / "call.json").exists():
        return  # standalone array checks have no request-shape authority
    call = _call_record(call_dir)
    width = call.get("logical_batch_size")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise PackageError("saved request has invalid physical width")
    if not arrays or any(array.ndim < 1 or array.shape[0] != width for array in arrays.values()):
        raise PackageError("saved array physical shape differs from exact request width")
    if call.get("arm_role") != "counterfactual":
        return
    peers: list[tuple[dict[str, object], dict[str, np.ndarray]]] = []
    for sibling in call_dir.parent.iterdir():
        if sibling == call_dir or sibling.is_symlink() or not sibling.is_dir():
            continue
        peer = _call_record(sibling)
        if peer.get("arm_role") == "native" and all(
            peer.get(field) == call.get(field)
            for field in ("split", "row_ids", "call_kind", "arm_direction")
        ):
            peers.append((peer, _raw_arrays(sibling)))
    for _, peer_arrays in peers:
        if set(peer_arrays) == set(arrays) and all(
            np.array_equal(arrays[name], peer_arrays[name], equal_nan=True) for name in arrays
        ):
            raise PackageError("counterfactual arm is dead: evidence is identical to native")


def write_nonfinite_masks(call_dir: Path) -> list[dict[str, object]]:
    """Write the approved R592 one-to-one masks for one failing call."""
    index_path = call_dir / "nonfinite_mask_index.json"
    mask_dir = call_dir / "nonfinite_masks"
    if index_path.exists() or mask_dir.exists() or (call_dir / "nonfinite_mask.npy").exists():
        raise PackageError("nonfinite-mask namespace is already occupied")
    entries: list[dict[str, object]] = []
    for raw_filename, raw in _raw_arrays(call_dir).items():
        if raw.dtype.kind != "f" or bool(np.isfinite(raw).all()):
            continue
        mask = np.asarray(~np.isfinite(raw), dtype=np.bool_, order="C")
        relative = canonical_mask_filename(raw_filename)
        destination = call_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, mask, allow_pickle=False)
        entries.append({
            "raw_filename": raw_filename,
            "mask_filename": relative,
            "raw_dtype": str(raw.dtype),
            "mask_dtype": "bool",
            "shape": list(raw.shape),
            "mask_byte_length": int(mask.nbytes),
            "mask_sha256": sha256_file(destination),
            "nonfinite_count": int(mask.sum()),
            "first_lexicographic_coordinate": first_true_coordinate(mask),
        })
    if not entries:
        raise PackageError("nonfinite terminal has no nonfinite raw float array")
    entries.sort(key=lambda item: str(item["raw_filename"]))
    index_path.write_bytes(canonical_json_bytes(entries) + b"\n")
    return entries


_MASK_FIELDS = {
    "raw_filename", "mask_filename", "raw_dtype", "mask_dtype", "shape",
    "mask_byte_length", "mask_sha256", "nonfinite_count",
    "first_lexicographic_coordinate",
}


def validate_nonfinite_masks(call_dir: Path, predicate_id: str) -> None:
    """Reconstruct exact mask membership and metadata from saved raw arrays."""
    index_path = call_dir / "nonfinite_mask_index.json"
    mask_dir = call_dir / "nonfinite_masks"
    if (call_dir / "nonfinite_mask.npy").exists():
        raise PackageError("flat nonfinite mask is forbidden")
    raw_arrays = _raw_arrays(call_dir)
    _validate_call_arrays(call_dir, raw_arrays)
    if predicate_id != "nonfinite_observation":
        if index_path.exists() or mask_dir.exists():
            raise PackageError("mask artifacts exist under a finite predicate")
        if any(array.dtype.kind == "f" and not bool(np.isfinite(array).all())
               for array in raw_arrays.values()):
            raise PackageError("nonfinite array before final diagnostic call")
        return
    if not index_path.is_file() or index_path.is_symlink() \
            or not mask_dir.is_dir() or mask_dir.is_symlink():
        raise PackageError("nonfinite terminal lacks a safe mask index/directory")
    try:
        entries = _strict_json_loads(index_path.read_bytes(), "nonfinite mask index")
    except (OSError, ValueError, TypeError) as error:
        raise PackageError("nonfinite mask index is invalid JSON") from error
    if not isinstance(entries, list) or not entries:
        raise PackageError("nonfinite mask index must be nonempty")
    if entries != sorted(entries, key=lambda item: item.get("raw_filename", "")):
        raise PackageError("nonfinite mask index is not sorted")
    if any(not isinstance(item, dict) or set(item) != _MASK_FIELDS for item in entries):
        raise PackageError("nonfinite mask index fields changed")
    expected_raw = {
        name for name, array in raw_arrays.items()
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all())
    }
    indexed_raw = [str(item["raw_filename"]) for item in entries]
    indexed_masks = [str(item["mask_filename"]) for item in entries]
    if len(indexed_raw) != len(set(indexed_raw)) or len(indexed_masks) != len(set(indexed_masks)):
        raise PackageError("duplicate raw or mask name")
    if set(indexed_raw) != expected_raw:
        raise PackageError("mask index is not the exact nonfinite-array set")
    expected_files: set[str] = set()
    for item in entries:
        raw_filename = str(item["raw_filename"])
        relative = canonical_mask_filename(raw_filename)
        if item["mask_filename"] != relative:
            raise PackageError("nonfinite mask path is noncanonical or traversing")
        destination = call_dir / relative
        if not destination.is_file() or destination.is_symlink():
            raise PackageError("nonfinite mask is missing or unsafe")
        mask = np.load(destination, allow_pickle=False)
        raw = raw_arrays[raw_filename]
        expected = ~np.isfinite(raw)
        if mask.dtype != np.bool_ or not mask.flags.c_contiguous:
            raise PackageError("nonfinite mask dtype/layout changed")
        if list(mask.shape) != list(raw.shape) or item["shape"] != list(raw.shape):
            raise PackageError("nonfinite mask shape changed")
        if item["raw_dtype"] != str(raw.dtype) or item["mask_dtype"] != "bool":
            raise PackageError("nonfinite mask dtype metadata changed")
        if item["mask_byte_length"] != mask.nbytes \
                or mask.nbytes != int(np.prod(mask.shape, dtype=np.int64)):
            raise PackageError("nonfinite mask byte length changed")
        if item["mask_sha256"] != sha256_file(destination):
            raise PackageError("nonfinite mask SHA-256 changed")
        if not np.array_equal(mask, expected):
            raise PackageError("nonfinite mask content changed")
        if item["nonfinite_count"] != int(mask.sum()) or int(mask.sum()) <= 0:
            raise PackageError("nonfinite mask count changed")
        if item["first_lexicographic_coordinate"] != first_true_coordinate(mask):
            raise PackageError("nonfinite first coordinate changed")
        expected_files.add(PurePosixPath(relative).name)
    children = list(mask_dir.iterdir())
    observed_files = {
        path.name for path in children if path.is_file() and not path.is_symlink()
    }
    if observed_files != expected_files or any(
        path.is_symlink() or not path.is_file() for path in children
    ):
        raise PackageError("nonfinite mask directory has missing or extra entries")


def validate_science_projection(
    evidence: Mapping[str, object],
    saved_projection: Mapping[str, object],
    projector: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> dict[str, object]:
    """Require an order-independent, environment-pure evidence projection."""
    observed = dict(projector(evidence))
    if observed != dict(saved_projection):
        raise PackageError("saved scientific projection differs from primitive evidence")
    records = evidence.get("records")
    if isinstance(records, list) and len(records) > 1:
        permuted = dict(evidence); permuted["records"] = list(reversed(records))
        if not _equivalent_json(observed, dict(projector(permuted))):
            raise PackageError("scientific projector is not pure under evidence order")
    environment = dict(os.environ)
    try:
        os.environ.clear()
        try:
            isolated = dict(projector(evidence))
        except Exception as error:
            raise PackageError("scientific projector is not environment-pure") from error
    finally:
        os.environ.clear(); os.environ.update(environment)
    if not _equivalent_json(observed, isolated):
        raise PackageError("scientific projector is not pure with respect to environment")
    canonical_json_bytes(observed)
    return observed


def _equivalent_json(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return bool(np.isclose(left, right, rtol=1e-12, atol=1e-15))
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_equivalent_json(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_equivalent_json(a, b) for a, b in zip(left, right))
    return left == right


def decide_experiment(*, spec, compiled: Mapping[str, object], primitives: list,
                      evaluators, projector: Callable) -> dict[str, object]:
    """Evaluate registered instruments before permitting scientific projection."""
    ordered = sorted(spec.predicates, key=lambda item: item.priority)
    if compiled.get("predicate_order") != [item.predicate_id for item in ordered]:
        raise PackageError("compiled predicate order changed")
    results: dict[str, bool] = {}
    calls = {call["call_id"]: call for call in compiled.get("call_manifest", [])}
    for predicate in ordered:
        phase_primitives = [item for item in primitives if (
            item.get("call_id") in calls
            and calls[item["call_id"]].get("split") == predicate.phase
        )]
        try:
            outcome = evaluators[predicate.evaluator_role](phase_primitives)
        except (KeyError, TypeError) as error:
            raise PackageError("registered predicate evaluator is unavailable") from error
        if not isinstance(outcome, bool):
            raise PackageError("predicate evaluator must return a boolean")
        results[predicate.predicate_id] = outcome
        if not outcome and predicate.disposition == "hard_abort":
            projection = {name: None for name in spec.science.output_types}
            return {"terminal": "hard_abort", "projection": projection,
                    "predicates_evaluated": True, "predicate_results": results}
    projection = dict(projector(primitives))
    validate_science_projection(
        {"records": primitives}, projection, lambda value: projector(value["records"])
    )
    return {"terminal": "ok", "projection": projection,
            "predicates_evaluated": True, "predicate_results": results}


@dataclass(frozen=True)
class PackagePaths:
    root: Path
    result: Path
    receipt: Path
    evidence: Path
    namespace: str


def _safe_relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise PackageError(f"unsafe evidence path: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _marker_bytes(namespace: str) -> bytes:
    return canonical_json_bytes({"namespace": namespace, "schema": "circuit-stage-v1"})


def _validate_stage_tree(stage: Path, paths: PackagePaths) -> None:
    """Recognize only this package's unpublished, regular-file staging tree."""
    marker = stage / "marker.json"
    if stage.is_symlink() or not stage.is_dir() or not marker.is_file() \
            or marker.is_symlink() or marker.read_bytes() != _marker_bytes(paths.namespace):
        raise PackageError("unrecognized stage")
    allowed_top = {"marker.json", "evidence", "result.json", "receipt.json"}
    if any(child.name not in allowed_top or child.is_symlink() for child in stage.iterdir()):
        raise PackageError("stage contains arbitrary top-level bytes")
    evidence = stage / "evidence"
    if evidence.exists():
        if evidence.is_symlink() or not evidence.is_dir():
            raise PackageError("staged evidence root is unsafe")
        for child in evidence.rglob("*"):
            if child.is_symlink() or (not child.is_dir() and not child.is_file()):
                raise PackageError("staged evidence tree is unsafe")
    for name in ("result.json", "receipt.json"):
        child = stage / name
        if child.exists() and (child.is_symlink() or not child.is_file()):
            raise PackageError("staged package file is unsafe")


def stage_package(
    paths: PackagePaths,
    *,
    evidence_files: Mapping[str, bytes],
    result: Mapping[str, object],
    crash: Callable[[str], None] | None = None,
) -> Path:
    """Write a mutually hash-bound package under one same-filesystem stage."""
    if any(path.exists() for path in (paths.result, paths.receipt, paths.evidence)):
        raise PackageError("final package namespace is occupied")
    stage = Path(tempfile.mkdtemp(prefix=f".{paths.namespace}-stage-", dir=paths.root))
    if stage.stat().st_dev != paths.root.stat().st_dev:
        raise PackageError("stage and final paths are on different filesystems")
    _write_fsynced(stage / "marker.json", _marker_bytes(paths.namespace))
    stage_evidence = stage / "evidence"
    stage_evidence.mkdir()
    descriptors = []
    for relative_name, payload in sorted(evidence_files.items()):
        relative = _safe_relative(relative_name)
        destination = stage_evidence.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_fsynced(destination, payload)
        descriptors.append({
            "path": relative.as_posix(), "bytes": len(payload), "sha256": sha256_bytes(payload)
        })
        if crash:
            crash(f"evidence:{relative.as_posix()}")
    result_payload = dict(result)
    result_payload["evidence_files"] = descriptors
    result_bytes = canonical_json_bytes(result_payload) + b"\n"
    _write_fsynced(stage / "result.json", result_bytes)
    if crash:
        crash("result")
    receipt = {
        "schema": "circuit-package-receipt-v1",
        "namespace": paths.namespace,
        "result_sha256": sha256_bytes(result_bytes),
        "evidence_files": descriptors,
    }
    _write_fsynced(stage / "receipt.json", canonical_json_bytes(receipt) + b"\n")
    if crash:
        crash("receipt")
    _fsync_directory(stage_evidence); _fsync_directory(stage)
    return stage


def publish_staged_package(
    stage: Path, paths: PackagePaths, *, crash: Callable[[str], None] | None = None
) -> None:
    """Publish evidence, result, and receipt last, rolling back failed renames."""
    _validate_stage_tree(stage, paths)
    moves = (
        (stage / "evidence", paths.evidence, "evidence"),
        (stage / "result.json", paths.result, "result"),
        (stage / "receipt.json", paths.receipt, "receipt"),
    )
    if any(destination.exists() for _, destination, _ in moves):
        raise PackageError("final namespace became occupied")
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination, label in moves:
            os.replace(source, destination)
            published.append((source, destination))
            _fsync_directory(destination.parent)
            if crash:
                crash(f"published:{label}")
    except BaseException:
        for source, destination in reversed(published):
            if destination.exists() and not source.exists():
                os.replace(destination, source)
        _fsync_directory(stage)
        _fsync_directory(paths.root)
        raise
    (stage / "marker.json").unlink()
    stage.rmdir()
    _fsync_directory(paths.root)


def validate_complete_package(paths: PackagePaths) -> dict[str, object]:
    if any(path.is_symlink() or not path.is_file() for path in (paths.receipt, paths.result)) \
            or paths.evidence.is_symlink() or not paths.evidence.is_dir():
        raise PackageError("package has unsafe or missing final paths")
    result_bytes = paths.result.read_bytes()
    result = _strict_json_loads(result_bytes, "result")
    receipt = _strict_json_loads(paths.receipt.read_bytes(), "receipt")
    if not isinstance(result, dict) or not isinstance(receipt, dict):
        raise PackageError("result and receipt must be JSON objects")
    if receipt.get("namespace") != paths.namespace \
            or receipt.get("result_sha256") != sha256_bytes(result_bytes):
        raise PackageError("result/receipt binding changed")
    expected = receipt.get("evidence_files")
    if result.get("evidence_files") != expected or not isinstance(expected, list):
        raise PackageError("result/evidence descriptor binding changed")
    observed = []
    expected_paths: set[str] = set()
    for descriptor in expected:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "bytes", "sha256"}:
            raise PackageError("evidence descriptor fields changed")
        if not isinstance(descriptor["path"], str) \
                or not isinstance(descriptor["bytes"], int) \
                or isinstance(descriptor["bytes"], bool) \
                or descriptor["bytes"] < 0 \
                or not isinstance(descriptor["sha256"], str) \
                or len(descriptor["sha256"]) != 64 \
                or any(ch not in "0123456789abcdef" for ch in descriptor["sha256"]):
            raise PackageError("evidence descriptor types changed")
        relative = _safe_relative(descriptor["path"])
        if relative.as_posix() in expected_paths:
            raise PackageError("evidence descriptor is duplicated")
        expected_paths.add(relative.as_posix())
        path = paths.evidence.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise PackageError("evidence file is missing or unsafe")
        payload = path.read_bytes()
        observed.append({"path": relative.as_posix(), "bytes": len(payload), "sha256": sha256_bytes(payload)})
    if observed != expected:
        raise PackageError("evidence bytes differ from receipt")
    actual_paths: set[str] = set()
    for child in paths.evidence.rglob("*"):
        if child.is_symlink() or (not child.is_dir() and not child.is_file()):
            raise PackageError("evidence tree contains an unsafe entry")
        if child.is_file():
            actual_paths.add(child.relative_to(paths.evidence).as_posix())
    if actual_paths != expected_paths:
        raise PackageError("evidence tree has missing or extra files")
    return result


def discard_stage(stage: Path, paths: PackagePaths) -> None:
    """Remove only a recognizable unpublished stage created for this namespace."""
    _validate_stage_tree(stage, paths)
    shutil.rmtree(stage)
    _fsync_directory(paths.root)


def recover_stale_publication(stage: Path, paths: PackagePaths) -> None:
    """Recover only a recognized incomplete package; never replace a receipt."""
    _validate_stage_tree(stage, paths)
    finals = (paths.evidence.exists(), paths.result.exists(), paths.receipt.exists())
    if finals[2] or all(finals):
        raise PackageError("refusing recovery over a complete package")
    moves = (
        (paths.evidence, stage / "evidence"),
        (paths.result, stage / "result.json"),
    )
    restored: list[tuple[Path, Path]] = []
    try:
        for final, staged in moves:
            if final.exists():
                if staged.exists() or final.is_symlink():
                    raise PackageError("stale package has ambiguous or unsafe duplicate bytes")
                os.replace(final, staged)
                restored.append((final, staged))
        _fsync_directory(stage)
        _fsync_directory(paths.root)
        validate_complete_package(PackagePaths(
            root=stage, result=stage / "result.json", receipt=stage / "receipt.json",
            evidence=stage / "evidence", namespace=paths.namespace,
        ))
    except BaseException:
        for final, staged in reversed(restored):
            if staged.exists() and not final.exists():
                os.replace(staged, final)
        _fsync_directory(stage)
        _fsync_directory(paths.root)
        raise
    discard_stage(stage, paths)
