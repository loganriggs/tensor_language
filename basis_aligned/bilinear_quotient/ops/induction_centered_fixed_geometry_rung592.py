#!/usr/bin/env python3
"""Prospective, CPU-testable contract and producer core for Rung 592.

This module contains no eager model import.  The managed adapter validates the
immutable dependency closure before invoking a future model-backed executor.
The functions here own the frozen row/call schedule, centered-factor algebra,
per-call evidence contracts, invalid-prefix rules, and model-free dry run.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import types
from typing import Any, Mapping, Sequence

import numpy as np


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"

R591 = OPS / "induction_replay_native_numerics_rung591.py"
R585 = OPS / "induction_selector_payload_frozen_factor_rung585.py"
MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
PREREG = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION.md"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_AMENDMENT.md"
DIAGNOSTIC_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
MASK_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT.md"
RUNTIME = OPS / "induction_centered_fixed_geometry_rung592_runtime.py"

SOURCE_HASHES = {
    R591: "fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc",
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    PREREG: "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a",
    AMENDMENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    DIAGNOSTIC_AMENDMENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    MASK_AMENDMENT: "f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc",
    RUNTIME: "df2d59245dc5bd407c96af0a8a6d1c98a70ae25f1925c4540dbd47bb956254a1",
}

SCHEMA = "induction_centered_fixed_geometry_rung592_v1"
DRYRUN_SCHEMA = "induction_centered_fixed_geometry_rung592_dryrun_v1"
WIDTH = 30
BATCH = 32
VOCAB = 50_257
RESIDUAL = 1_152
PAD_TOKEN = 50_256
TOLERANCE = 1e-5
SITES = ("L5H5", "L7H3", "L8H3", "L8H4")
ROLES = ("A", "C")
MACHINE_ARMS = ("replay", "score", "payload", "joint")
DIRECTED_KINDS = ("native", *MACHINE_ARMS)
DIFFERENCE_ORDER = (
    "native_minus_replay", "score_minus_replay",
    "payload_minus_replay", "joint_minus_replay",
)
OPERATIONAL_ARM_LABELS = {
    "replay": "literal_zero_centered_replay",
    "score": "registered_equality_factor_coefficient_swap",
    "payload": "registered_projected_content_swap",
    "joint": "registered_joint_output_factor_swap",
}
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
FORBIDDEN_PREDICATES = ("canonical_term_failure", "factor_mismatch", "padding_failure")
PHASE_COUNTS = {
    "FIT": {"rows": 1_872, "endpoints": 1_728, "directions": 3_744,
            "endpoint_calls": 54, "directed_chunks": 117, "calls": 639},
    "SELECT": {"rows": 936, "endpoints": 864, "directions": 1_872,
               "endpoint_calls": 27, "directed_chunks": 59, "calls": 322},
}

NORMAL_RESULT = ROOT / "induction_centered_fixed_geometry_rung592_results.json"
NORMAL_RECEIPT = ROOT / "induction_centered_fixed_geometry_rung592_receipt.json"
NORMAL_EVIDENCE = ROOT / "induction_centered_fixed_geometry_rung592_evidence"
INVALID_RESULT = ROOT / "induction_centered_fixed_geometry_rung592_invalid_diagnostic.json"
INVALID_RECEIPT = ROOT / "induction_centered_fixed_geometry_rung592_invalid_receipt.json"
INVALID_EVIDENCE = ROOT / "induction_centered_fixed_geometry_rung592_invalid_evidence"
PUBLIC_NAMESPACES = (
    NORMAL_RESULT, NORMAL_RECEIPT, NORMAL_EVIDENCE,
    INVALID_RESULT, INVALID_RECEIPT, INVALID_EVIDENCE,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def content_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode())


def verify_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in SOURCE_HASHES.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen R592 authority changed: {path}")
        observed[str(path)] = expected
    return observed


def _immutable_module(path: Path, expected: str, name: str):
    source = path.read_bytes()
    if sha256_bytes(source) != expected:
        raise RuntimeError(f"immutable source changed before import: {path}")
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path))
    if spec is None:
        raise RuntimeError(f"cannot construct immutable module: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_authority() -> tuple[object, dict[str, object]]:
    """Load only the outcome-blind R585 row authority through pinned R591 code."""
    verify_sources()
    r591 = _immutable_module(R591, SOURCE_HASHES[R591], "r592_pinned_r591_authority")
    r585, execution = r591.load_authority()
    # The callable is not invoked by authority construction or dry runs.  It is
    # the immutable R591 loader for the complete torch/facade dependency closure
    # and becomes reachable only after run_science crosses the model boundary.
    r585.__r592_runtime_loader__ = r591._load_runtime
    return r585, execution


def _little_c(array: np.ndarray, dtype: str | np.dtype) -> np.ndarray:
    target = np.dtype(dtype).newbyteorder("<")
    return np.ascontiguousarray(np.asarray(array, dtype=target))


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def fixed_tokens(records: Sequence[Mapping[str, object]], token_field: str) -> np.ndarray:
    tokens = np.full((len(records), WIDTH), PAD_TOKEN, dtype="<i8")
    for index, row in enumerate(records):
        ids = np.asarray(row[token_field], dtype="<i8")
        if ids.ndim != 1 or not 0 < len(ids) <= WIDTH:
            raise ValueError("invalid token sequence")
        tokens[index, : len(ids)] = ids
    return tokens


def _chunks(rows: Sequence[Mapping[str, object]]) -> list[list[Mapping[str, object]]]:
    return [list(rows[start:start + BATCH]) for start in range(0, len(rows), BATCH)]


def _token_record(
    phase: str, kind: str, chunk: int, rows: Sequence[Mapping[str, object]],
    endpoint_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], np.ndarray]:
    if kind == "endpoint":
        specs = list(rows)
        ids = [str(row["endpoint_id"]) for row in specs]
        directions: list[str] = []
    else:
        specs = [endpoint_by_id[str(row["recipient_endpoint_id"])] for row in rows]
        ids = [str(row["row_id"]) for row in rows]
        directions = [str(row["directed_id"]) for row in rows]
    tokens = fixed_tokens(specs, "token_ids")
    if tokens.shape[1] != WIDTH or tokens.dtype != np.dtype("<i8"):
        raise AssertionError("fixed token geometry changed")
    token_id = f"{phase}:{kind}:{chunk:04d}:tokens"
    record = {
        "token_record_id": token_id,
        "token_sha256": sha256_bytes(tokens.tobytes(order="C")),
        "token_npy_sha256": sha256_bytes(npy_bytes(tokens)),
        "dtype": "int64",
        "shape": list(tokens.shape),
        "byte_length": int(tokens.nbytes),
        "authority_row_ids": ids,
        "direction_ids": directions,
        "query_positions": [int(spec["final_position"]) for spec in specs],
    }
    return record, tokens


def build_phase_manifest(execution: Mapping[str, object], phase: str) -> dict[str, object]:
    if phase not in PHASE_COUNTS:
        raise ValueError("phase must be FIT or SELECT")
    endpoints = [row for row in execution["endpoints"] if row["split"] == phase]
    directions = [row for row in execution["directions"] if row["split"] == phase]
    endpoint_by_id = {str(row["endpoint_id"]): row for row in endpoints}
    endpoint_chunks = _chunks(endpoints)
    direction_chunks = _chunks(directions)
    calls: list[dict[str, object]] = []
    tensors: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    index = 0
    for chunk, batch in enumerate(endpoint_chunks):
        tensor, values = _token_record(phase, "endpoint", chunk, batch, endpoint_by_id)
        tensors[tensor["token_record_id"]] = tensor
        arrays[tensor["token_record_id"]] = values
        calls.append({
            "manifest_index": index, "call_id": f"{phase}:endpoint:{chunk:04d}",
            "phase": phase, "call_kind": "endpoint", "chunk_index": chunk,
            "machine_arm": None, "token_record_id": tensor["token_record_id"],
            "token_sha256": tensor["token_sha256"], "batch_size": len(batch),
            "physical_width": WIDTH, "authority_row_ids": tensor["authority_row_ids"],
            "direction_ids": [], "query_positions": tensor["query_positions"],
        })
        index += 1
    for chunk, batch in enumerate(direction_chunks):
        tensor, values = _token_record(phase, "directed", chunk, batch, endpoint_by_id)
        tensors[tensor["token_record_id"]] = tensor
        arrays[tensor["token_record_id"]] = values
        for kind in DIRECTED_KINDS:
            calls.append({
                "manifest_index": index,
                "call_id": f"{phase}:directed:{chunk:04d}:{kind}",
                "phase": phase, "call_kind": kind, "chunk_index": chunk,
                "machine_arm": None if kind == "native" else kind,
                "token_record_id": tensor["token_record_id"],
                "token_sha256": tensor["token_sha256"], "batch_size": len(batch),
                "physical_width": WIDTH, "authority_row_ids": tensor["authority_row_ids"],
                "direction_ids": tensor["direction_ids"],
                "query_positions": tensor["query_positions"],
            })
            index += 1
    expected = PHASE_COUNTS[phase]
    if (len(endpoints), len(directions), len(endpoint_chunks), len(direction_chunks), len(calls)) != (
        expected["endpoints"], expected["directions"], expected["endpoint_calls"],
        expected["directed_chunks"], expected["calls"],
    ):
        raise RuntimeError(f"{phase} fixed-price census changed")
    if phase == "FIT" and any(call["batch_size"] != 32 for call in calls):
        raise RuntimeError("FIT contains a partial batch")
    if phase == "SELECT":
        tail = [call for call in calls if call["call_kind"] != "endpoint"][-5:]
        if [call["call_kind"] for call in tail] != list(DIRECTED_KINDS) or any(
            call["batch_size"] != 16 for call in tail
        ) or any(call["batch_size"] != 32 for call in calls[:-5]):
            raise RuntimeError("SELECT final batch is not exact 16-row five-call tail")
    for chunk in range(len(direction_chunks)):
        group = [call for call in calls if call["call_id"].startswith(f"{phase}:directed:{chunk:04d}:")]
        if len({call["token_sha256"] for call in group}) != 1 or len({call["token_record_id"] for call in group}) != 1:
            raise RuntimeError("paired directed calls do not share exact token bytes")
    return {"phase": phase, "calls": calls, "token_records": tensors,
            "token_arrays": arrays, "call_manifest_sha256": content_sha256(calls),
            "token_manifest_sha256": content_sha256(tensors)}


def bilinear(e: np.ndarray, u: np.ndarray) -> np.ndarray:
    e = _little_c(e, "<f4")
    u = _little_c(u, "<f4")
    if e.shape[-1] != 2 or u.shape[-2:] != (2, RESIDUAL) or e.shape != u.shape[:-1]:
        raise ValueError("factor shapes must end [2] and [2,1152]")
    return np.einsum("...r,...rj->...j", e, u, dtype=np.float32, optimize=False)


def centered_deltas(
    recipient_e: np.ndarray, recipient_u: np.ndarray,
    donor_e: np.ndarray, donor_u: np.ndarray,
) -> np.ndarray:
    """Return machine-arm axis replay, score, payload, joint in float32."""
    xx = bilinear(recipient_e, recipient_u)
    yx = bilinear(donor_e, recipient_u)
    xy = bilinear(recipient_e, donor_u)
    yy = bilinear(donor_e, donor_u)
    replay = np.zeros_like(xx)
    return _little_c(np.stack((replay, yx - xx, xy - xx, yy - xx), axis=-3), "<f4")


def mixed_identity_error(
    deltas: np.ndarray, recipient_e: np.ndarray, recipient_u: np.ndarray,
    donor_e: np.ndarray, donor_u: np.ndarray,
) -> float:
    observed = deltas[..., 3, :, :] - deltas[..., 1, :, :] - deltas[..., 2, :, :]
    expected = bilinear(donor_e - recipient_e, donor_u - recipient_u)
    return float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64))))


def transport_maxima(
    cached_x_e: np.ndarray, cached_x_u: np.ndarray, cached_y_e: np.ndarray,
    cached_y_u: np.ndarray, live_x_e: np.ndarray, live_x_u: np.ndarray,
) -> dict[str, float]:
    maxima = {
        "e": float(np.max(np.abs(live_x_e.astype(np.float64) - cached_x_e.astype(np.float64)))),
        "u": float(np.max(np.abs(live_x_u.astype(np.float64) - cached_x_u.astype(np.float64)))),
    }
    cached = (
        bilinear(cached_x_e, cached_x_u), bilinear(cached_y_e, cached_x_u),
        bilinear(cached_x_e, cached_y_u), bilinear(cached_y_e, cached_y_u),
    )
    live = (
        bilinear(live_x_e, live_x_u), bilinear(cached_y_e, live_x_u),
        bilinear(live_x_e, cached_y_u), cached[3],
    )
    for name, left, right in zip(("xx", "yx", "xy", "yy"), live, cached):
        maxima[name] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
    return maxima


def activity(actual_deltas: np.ndarray) -> np.ndarray:
    """Median of four site L2 norms, separately for each direction and arm."""
    values = np.asarray(actual_deltas, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (4, 4, RESIDUAL):
        raise ValueError("hook delta shape must be [direction,arm,site,1152]")
    return np.median(np.linalg.norm(values, axis=-1), axis=-1)


def mandatory_call_shapes(call: Mapping[str, object]) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    b = int(call["batch_size"])
    common = {"tokens.npy": (np.dtype("<i8"), (b, WIDTH)),
              "logits.npy": (np.dtype("<f4"), (b, VOCAB))}
    parent = {
        "native_equality_term.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
        "factorized_equality_term.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
        "native_non_equality_remainder.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
        "native_head_write.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
        "independent_full_native_write.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
    }
    kind = str(call["call_kind"])
    if kind == "endpoint":
        return common | {
            "factor_e.npy": (np.dtype("<f4"), (b, 4, 2)),
            "factor_u.npy": (np.dtype("<f4"), (b, 4, 2, RESIDUAL)),
            "support.npy": (np.dtype("bool"), (b, 4, 2)),
        } | parent
    if kind == "native":
        return common | {
            "live_e.npy": (np.dtype("<f4"), (b, 4, 2)),
            "live_u.npy": (np.dtype("<f4"), (b, 4, 2, RESIDUAL)),
        } | parent
    if kind not in MACHINE_ARMS:
        raise ValueError("unknown call kind")
    return common | {
        "hook_deltas.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
        "planned_hook_deltas.npy": (np.dtype("<f4"), (b, 4, RESIDUAL)),
    }


def validate_call_arrays(call: Mapping[str, object], arrays: Mapping[str, np.ndarray], *, allow_nonfinite: bool = False) -> None:
    contract = mandatory_call_shapes(call)
    if set(arrays) != set(contract):
        raise ValueError("call evidence has missing or extra arrays")
    for filename, (dtype, shape) in contract.items():
        value = np.asarray(arrays[filename])
        if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise ValueError(f"wrong dtype/shape/order for {filename}")
        if value.dtype.kind == "f" and not allow_nonfinite and not np.isfinite(value).all():
            raise ValueError(f"nonfinite earlier call array: {filename}")


def validate_prefix(manifest: Sequence[Mapping[str, object]], records: Sequence[Mapping[str, object]]) -> None:
    if list(records) != list(manifest[:len(records)]):
        raise ValueError("call records are not an exact frozen manifest prefix")


def first_failure(failures: Sequence[str]) -> str | None:
    observed = set(failures)
    unknown = observed - set(PREDICATE_ORDER)
    if unknown or observed.intersection(FORBIDDEN_PREDICATES):
        raise ValueError(f"unknown/forbidden R592 predicates: {sorted(unknown)}")
    return next((name for name in PREDICATE_ORDER if name in observed), None)


def canonical_mask_name(raw_filename: str) -> str:
    path = PurePosixPath(raw_filename)
    if path.is_absolute() or len(path.parts) != 1 or path.suffix != ".npy" or ".." in path.parts:
        raise ValueError("unsafe raw filename")
    return f"nonfinite_masks/{path.stem}.mask.npy"


def nonfinite_mask_records(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    masks: dict[str, np.ndarray] = {}
    index: list[dict[str, object]] = []
    for filename in sorted(arrays):
        raw = np.asarray(arrays[filename])
        if raw.dtype.kind != "f" or np.isfinite(raw).all():
            continue
        mask = np.ascontiguousarray(~np.isfinite(raw), dtype=np.bool_)
        mask_name = canonical_mask_name(filename)
        coords = np.argwhere(mask)
        data = npy_bytes(mask)
        masks[mask_name] = mask
        index.append({
            "raw_filename": filename, "mask_filename": mask_name,
            "raw_dtype": str(raw.dtype), "mask_dtype": "bool", "shape": list(raw.shape),
            "mask_byte_length": int(mask.nbytes), "mask_sha256": sha256_bytes(data),
            "nonfinite_count": int(mask.sum()),
            "first_lexicographic_coordinate": [int(x) for x in coords[0]],
        })
    return masks, index


def phase_evidence_schema(phase: str) -> dict[str, object]:
    counts = PHASE_COUNTS[phase]
    ne, nd = counts["endpoints"], counts["directions"]
    return {
        "call_manifest.json": {"records": counts["calls"]},
        "token_manifest.json": {
            "records": counts["endpoint_calls"] + counts["directed_chunks"]
        },
        "authority.jsonl": {"records": counts["rows"]},
        "endpoint_records.jsonl": {"records": ne},
        "endpoint_tokens.npy": {"dtype": "int64", "shape": [ne, WIDTH]},
        "factor_e.npy": {"dtype": "float32", "shape": [ne, 4, 2]},
        "factor_u.npy": {"dtype": "float32", "shape": [ne, 4, 2, RESIDUAL]},
        "support.npy": {"dtype": "bool", "shape": [ne, 4, 2]},
        **{name: {"dtype": "float32", "shape": [ne, 4, RESIDUAL]} for name in (
            "native_equality_term.npy", "factorized_equality_term.npy",
            "native_non_equality_remainder.npy", "native_head_write.npy",
            "independent_full_native_write.npy",
        )},
        "directed_records.jsonl": {"records": nd},
        "directed_tokens.npy": {"dtype": "int64", "shape": [nd, WIDTH]},
        "directed_live_e.npy": {"dtype": "float32", "shape": [nd, 4, 2]},
        "directed_live_u.npy": {"dtype": "float32", "shape": [nd, 4, 2, RESIDUAL]},
        "hook_deltas.npy": {"dtype": "float32", "shape": [nd, 4, 4, RESIDUAL]},
        "logit_differences.npy": {"dtype": "float32", "shape": [nd, 4, VOCAB]},
        "bootstrap_cells": {"records": 124, "replicates": 2_000},
    }


def terminal_contract(terminal: str, fit_calls: int, select_calls: int = 0) -> dict[str, object]:
    if terminal == "fit_runtime_invalid":
        valid = 1 <= fit_calls <= 639 and select_calls == 0
        namespace = "invalid"
    elif terminal == "fit_scientific_null":
        valid = fit_calls == 639 and select_calls == 0
        namespace = "normal"
    elif terminal == "select_runtime_invalid":
        valid = fit_calls == 639 and 1 <= select_calls <= 322
        namespace = "invalid"
    elif terminal in ("select_scientific_null", "held"):
        valid = fit_calls == 639 and select_calls == 322
        namespace = "normal"
    elif terminal == "dependency_or_preflight_failure":
        valid = fit_calls == select_calls == 0
        namespace = "none"
    else:
        raise ValueError("unknown terminal")
    if not valid:
        raise ValueError("terminal call envelope violated")
    return {"terminal": terminal, "namespace": namespace, "fit_calls": fit_calls,
            "select_calls": select_calls, "model_backwards": 0,
            "model_weights_updated": False, "final_opened": False, "ood_opened": False}


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "dtype": str(array.dtype), "shape": list(array.shape),
        "byte_length": path.stat().st_size, "sha256": sha256_file(path),
    }


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_completed_call(
    calls_root: Path, call: Mapping[str, object], arrays: Mapping[str, np.ndarray],
    *, nonfinite_terminal: bool = False,
) -> dict[str, object]:
    """Write one exact completed-call directory for a possible invalid prefix."""
    directory = calls_root / f"{int(call['manifest_index']):04d}_{call['call_id']}"
    directory.mkdir(parents=True, exist_ok=False)
    descriptors = {
        name: _write_npy(directory / name, np.asarray(arrays[name]))
        for name in sorted(arrays)
    }
    if nonfinite_terminal:
        masks, index = nonfinite_mask_records(arrays)
        if not index:
            raise ValueError("nonfinite terminal has no nonfinite arrays")
        for name, mask in masks.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_npy(path, mask)
        _write_bytes(directory / "nonfinite_mask_index.json", _json_bytes(index))
    return {**dict(call), "evidence_files": descriptors}


def load_call_arrays(directory: Path) -> dict[str, np.ndarray]:
    return {
        path.name: np.load(path, allow_pickle=False)
        for path in directory.glob("*.npy")
        if path.name != "nonfinite_mask.npy"
    }


def publish_invalid_prefix(
    stage: Path, manifest: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]], predicate: str,
    details: Mapping[str, object], *, public_root: Path = ROOT,
) -> dict[str, object]:
    validate_prefix(manifest, [{key: row[key] for key in manifest[0]} for row in records])
    if predicate not in PREDICATE_ORDER:
        raise ValueError("invalid diagnostic predicate")
    forbidden = {"split_scores", "scientific_terminal", "held", "null"}
    if forbidden.intersection(details):
        raise ValueError("scientific fields forbidden in invalid diagnostic")
    evidence = stage / "evidence"
    prefix_path = evidence / "call_prefix.jsonl"
    _write_bytes(prefix_path, b"".join(_json_bytes(dict(row)) + b"\n" for row in records))
    diagnostic = {
        "schema": "induction_centered_fixed_geometry_rung592_invalid_diagnostic_v1",
        "status": "invalid_diagnostic", "failure_predicate": predicate,
        "executed_call_ids": [row["call_id"] for row in records],
        "call_prefix_sha256": sha256_file(prefix_path), "details": dict(details),
        "model_backwards": 0, "model_weights_updated": False,
        "final_opened": False, "ood_opened": False,
    }
    diagnostic_path = stage / "diagnostic.json"
    _write_bytes(diagnostic_path, _json_bytes(diagnostic))
    receipt = {
        "schema": "induction_centered_fixed_geometry_rung592_invalid_receipt_v1",
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "call_prefix_sha256": sha256_file(prefix_path),
        "executed_call_ids": diagnostic["executed_call_ids"],
        "evidence_tree_files": sorted(str(path.relative_to(evidence)) for path in evidence.rglob("*") if path.is_file()),
    }
    receipt_path = stage / "receipt.json"
    _write_bytes(receipt_path, _json_bytes(receipt, pretty=True))
    targets = (
        public_root / INVALID_EVIDENCE.name,
        public_root / INVALID_RESULT.name,
        public_root / INVALID_RECEIPT.name,
    )
    if any(path.exists() for path in targets):
        raise RuntimeError("invalid namespace already occupied")
    _fsync_directory(evidence); _fsync_directory(stage)
    os.replace(evidence, targets[0]); _fsync_directory(public_root)
    os.replace(diagnostic_path, targets[1]); _fsync_directory(public_root)
    os.replace(receipt_path, targets[2]); _fsync_directory(public_root)  # receipt is recognition marker
    return diagnostic


def evaluate_completed_call(
    call: Mapping[str, object], arrays: Mapping[str, np.ndarray],
    expected_tokens: np.ndarray, *, metadata: Mapping[str, object] | None = None,
    cached: Mapping[str, object] | None = None,
) -> tuple[str | None, dict[str, object]]:
    """Evaluate every predicate that is decidable after one completed call."""
    metadata = {} if metadata is None else metadata
    failures: list[str] = []
    details: dict[str, object] = {}
    nonfinite = {
        name: np.argwhere(~np.isfinite(value))
        for name, value in arrays.items()
        if np.asarray(value).dtype.kind == "f" and not np.isfinite(value).all()
    }
    if nonfinite:
        failures.append("nonfinite_observation")
        details["nonfinite"] = {
            name: {"count": len(coords), "first_coordinate": [int(x) for x in coords[0]]}
            for name, coords in nonfinite.items()
        }
    try:
        validate_call_arrays(call, arrays, allow_nonfinite=bool(nonfinite))
        if not np.array_equal(arrays["tokens.npy"], expected_tokens) or (
            sha256_bytes(np.asarray(arrays["tokens.npy"]).tobytes(order="C")) != call["token_sha256"]
        ):
            raise ValueError("token bytes differ")
    except ValueError as error:
        failures.append("fixed_width_token_manifest_failed")
        details["token_or_shape_error"] = str(error)
    kind = str(call["call_kind"])
    if kind in ("endpoint", "native") and not nonfinite:
        full = float(metadata.get("native_full_write_reconstruction_max_abs", math.inf))
        details["native_full_write_reconstruction_max_abs"] = full
        if full > TOLERANCE:
            failures.append("native_full_write_reconstruction_failed")
        reconstruction = float(np.max(np.abs(
            arrays["native_equality_term.npy"].astype(np.float64)
            + arrays["native_non_equality_remainder.npy"].astype(np.float64)
            - arrays["native_head_write.npy"].astype(np.float64)
        )))
        details["native_equality_remainder_reconstruction_max_abs"] = reconstruction
        if reconstruction > TOLERANCE:
            failures.append("native_equality_remainder_reconstruction_failed")
        if kind == "endpoint" and not bool(np.asarray(arrays["support.npy"]).all()):
            details["support_false_count"] = int((~np.asarray(arrays["support.npy"])).sum())
            failures.append("factor_transport_failed")
    if kind == "native" and cached is not None and not nonfinite:
        maxima = transport_maxima(
            cached["recipient_e"], cached["recipient_u"], cached["donor_e"], cached["donor_u"],
            arrays["live_e.npy"], arrays["live_u.npy"],
        )
        details["factor_transport_maxima"] = maxima
        if max(maxima.values()) > TOLERANCE:
            failures.append("factor_transport_failed")
    if kind in MACHINE_ARMS and not nonfinite:
        error = float(np.max(np.abs(
            arrays["hook_deltas.npy"].astype(np.float64)
            - arrays["planned_hook_deltas.npy"].astype(np.float64)
        )))
        details["actual_centered_hook_delta_max_abs"] = error
        if error > TOLERANCE or (kind == "replay" and np.any(arrays["planned_hook_deltas.npy"])):
            failures.append("centered_hook_delta_failed")
        if kind == "replay" and cached is not None and "native_logits" in cached:
            replay_error = float(np.max(np.abs(
                cached["native_logits"].astype(np.float64) - arrays["logits.npy"].astype(np.float64)
            )))
            details["directed_native_zero_replay_max_abs"] = replay_error
            if replay_error > TOLERANCE:
                failures.append("directed_native_zero_replay_failed")
        if cached is not None and cached.get("structural_references"):
            structural_error = max(
                float(np.max(np.abs(
                    arrays["logits.npy"][local].astype(np.float64)
                    - np.asarray(reference).astype(np.float64)
                )))
                for local, reference in cached["structural_references"]
            )
            details["structural_output_identity_max_abs"] = structural_error
            if structural_error > TOLERANCE:
                failures.append("structural_output_identity_failed")
    return first_failure(failures), details


def run_manifest_calls(
    executor: object, bundle: Mapping[str, object], contexts: object,
    *, stage: Path, public_root: Path = ROOT,
) -> dict[str, object]:
    """Run one phase sequentially and stop/publish at the first completed-call failure.

    A raised executor call is a hard abort: the temporary tree is retained only
    for the caller to inspect and nothing is renamed into a public namespace.
    """
    calls = bundle["calls"]
    calls_root = stage / "evidence" / "calls"
    calls_root.mkdir(parents=True)
    prefix: list[dict[str, object]] = []
    outputs: dict[str, Path] = {}
    for call in calls:
        token = bundle["token_arrays"][call["token_record_id"]]
        if token.shape != (int(call["batch_size"]), WIDTH) or (
            sha256_bytes(token.tobytes(order="C")) != call["token_sha256"]
        ):
            raise RuntimeError("frozen token tensor changed before model call")
        context = (
            contexts(call, outputs) if callable(contexts)
            else contexts[call["call_id"]]
        )
        try:
            response = executor.execute(call, token, context["specs"], context.get("planned"))
        except Exception:
            # Incomplete call: do not turn temporary bytes into a diagnostic.
            raise
        arrays = {name: np.ascontiguousarray(value) for name, value in response["arrays"].items()}
        predicate, details = evaluate_completed_call(
            call, arrays, token, metadata=response, cached=context.get("cached")
        )
        record = write_completed_call(
            calls_root, call, arrays, nonfinite_terminal=(predicate == "nonfinite_observation")
        )
        prefix.append(record)
        outputs[str(call["call_id"])] = (
            calls_root / f"{int(call['manifest_index']):04d}_{call['call_id']}"
        )
        if predicate is not None:
            diagnostic = publish_invalid_prefix(
                stage, calls, prefix, predicate, details, public_root=public_root
            )
            return {"status": "invalid", "diagnostic": diagnostic, "outputs": outputs}
    return {"status": "complete", "records": prefix, "outputs": outputs}


def make_context_factory(execution: Mapping[str, object], bundle: Mapping[str, object]):
    endpoints = {
        str(row["endpoint_id"]): row for row in execution["endpoints"]
        if row["split"] == bundle["phase"]
    }
    directions = {
        str(row["directed_id"]): row for row in execution["directions"]
        if row["split"] == bundle["phase"]
    }
    endpoint_location: dict[str, tuple[str, int]] = {}
    direction_location: dict[str, tuple[int, int]] = {}
    for call in bundle["calls"]:
        if call["call_kind"] == "endpoint":
            for local, endpoint_id in enumerate(call["authority_row_ids"]):
                endpoint_location[str(endpoint_id)] = (str(call["call_id"]), local)
        elif call["call_kind"] == "native":
            for local, directed_id in enumerate(call["direction_ids"]):
                direction_location[str(directed_id)] = (int(call["chunk_index"]), local)
    structural_pairs: dict[str, list[tuple[str, str]]] = {}
    cells = {
        cell["cell_id"]: cell
        for cell in (*execution["manifests"]["target_cells"], *execution["manifests"]["control_cells"])
    }
    for identity in execution["manifests"]["structural_identities"]:
        if not str(identity["cell_id"]).startswith(bundle["phase"] + "|"):
            continue
        for directed_id in cells[identity["cell_id"]]["directed_ids"]:
            structural_pairs.setdefault(str(directed_id), []).append(
                (str(identity["left_arm"]), str(identity["right_arm"]))
            )

    def endpoint_array(outputs: Mapping[str, Path], endpoint_id: str, filename: str) -> np.ndarray:
        call_id, local = endpoint_location[endpoint_id]
        return np.asarray(load_call_arrays(outputs[call_id])[filename][local])

    def factory(call: Mapping[str, object], outputs: Mapping[str, Path]) -> dict[str, object]:
        kind = str(call["call_kind"])
        if kind == "endpoint":
            return {"specs": [endpoints[str(value)] for value in call["authority_row_ids"]]}
        rows = [directions[str(value)] for value in call["direction_ids"]]
        specs = [endpoints[str(row["recipient_endpoint_id"])] for row in rows]
        recipient_e = np.stack([endpoint_array(outputs, str(row["recipient_endpoint_id"]), "factor_e.npy") for row in rows])
        recipient_u = np.stack([endpoint_array(outputs, str(row["recipient_endpoint_id"]), "factor_u.npy") for row in rows])
        donor_e = np.stack([endpoint_array(outputs, str(row["donor_endpoint_id"]), "factor_e.npy") for row in rows])
        donor_u = np.stack([endpoint_array(outputs, str(row["donor_endpoint_id"]), "factor_u.npy") for row in rows])
        cached: dict[str, object] = {
            "recipient_e": recipient_e, "recipient_u": recipient_u,
            "donor_e": donor_e, "donor_u": donor_u,
        }
        if kind == "native":
            return {"specs": specs, "cached": cached}
        chunk = int(call["chunk_index"])
        native_id = f"{bundle['phase']}:directed:{chunk:04d}:native"
        cached["native_logits"] = load_call_arrays(outputs[native_id])["logits.npy"]
        all_deltas = centered_deltas(recipient_e, recipient_u, donor_e, donor_u)
        planned = all_deltas[:, MACHINE_ARMS.index(kind)]
        comparisons = []
        for local, row in enumerate(rows):
            for left, right in structural_pairs.get(str(row["directed_id"]), []):
                if kind == left:
                    reference_id = f"{bundle['phase']}:directed:{chunk:04d}:{right}"
                    comparisons.append((local, load_call_arrays(outputs[reference_id])["logits.npy"][local]))
        cached["structural_references"] = comparisons
        return {"specs": specs, "planned": planned, "cached": cached}

    factory.endpoint_location = endpoint_location
    factory.direction_location = direction_location
    return factory


def _measurement(logits: np.ndarray, answer: int, other: int) -> dict[str, float]:
    vector = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(vector))
    lse = maximum + math.log(float(np.exp(vector - maximum).sum()))
    answer_logit, other_logit = float(vector[answer]), float(vector[other])
    return {
        "answer_logit": answer_logit, "other_logit": other_logit,
        "correct_margin": answer_logit - other_logit,
        "log_normalizer": lse, "correct_ce": lse - answer_logit,
    }


def derive_scientific_records(
    execution: Mapping[str, object], bundle: Mapping[str, object], outputs: Mapping[str, Path],
) -> list[dict[str, object]]:
    phase = str(bundle["phase"])
    factory = make_context_factory(execution, bundle)
    endpoint_logits = {}
    for endpoint_id, (call_id, local) in factory.endpoint_location.items():
        endpoint_logits[endpoint_id] = load_call_arrays(outputs[call_id])["logits.npy"][local]
    directions = {
        str(row["directed_id"]): row for row in execution["directions"] if row["split"] == phase
    }
    records = []
    for directed_id, (chunk, local) in factory.direction_location.items():
        row = directions[directed_id]
        replay_id = f"{phase}:directed:{chunk:04d}:replay"
        replay_arrays = load_call_arrays(outputs[replay_id])
        replay_logits = replay_arrays["logits.npy"][local]
        recipient_logits = endpoint_logits[str(row["recipient_endpoint_id"])]
        donor_logits = endpoint_logits[str(row["donor_endpoint_id"])]
        recipient_answer = int(row["recipient_answer_id"])
        donor_answer = int(row["donor_answer_id"])
        other = int(row["recipient_other_answer_id"])
        replay_measure = _measurement(replay_logits, recipient_answer, other)
        for arm in ("score", "payload", "joint"):
            arm_arrays = load_call_arrays(outputs[f"{phase}:directed:{chunk:04d}:{arm}"])
            logits = arm_arrays["logits.npy"][local]
            measure = _measurement(logits, recipient_answer, other)
            if bool(row["answer_changes"]):
                m_i = float(logits[donor_answer] - logits[recipient_answer])
                m_r = float(replay_logits[donor_answer] - replay_logits[recipient_answer])
                m_d = float(donor_logits[donor_answer] - donor_logits[recipient_answer])
                m_x = float(recipient_logits[donor_answer] - recipient_logits[recipient_answer])
                n_value, d_value = m_i - m_r, m_d - m_x
                q_value = (
                    _measurement(replay_logits, donor_answer, recipient_answer)["correct_ce"]
                    - _measurement(logits, donor_answer, recipient_answer)["correct_ce"]
                )
            else:
                sign = int(row["donor_coherence_sign"] or 1)
                n_value = sign * (measure["correct_margin"] - replay_measure["correct_margin"])
                donor_measure = _measurement(donor_logits, recipient_answer, other)
                recipient_measure = _measurement(recipient_logits, recipient_answer, other)
                d_value = sign * (donor_measure["correct_margin"] - recipient_measure["correct_margin"])
                q_value = sign * (replay_measure["correct_ce"] - measure["correct_ce"])
            difference = logits.astype(np.float64) - replay_logits.astype(np.float64)
            actual = arm_arrays["hook_deltas.npy"][local]
            records.append({
                **{key: row[key] for key in (
                    "split", "directed_id", "row_id", "group_id", "family", "variant",
                    "recipient_condition", "direction", "control_kind", "answer_changes",
                )},
                "arm": arm, "recipient_endpoint_id": row["recipient_endpoint_id"],
                "donor_endpoint_id": row["donor_endpoint_id"],
                "recipient_answer_id": recipient_answer, "donor_answer_id": donor_answer,
                "other_answer_id": other,
                "replay_correct_margin": replay_measure["correct_margin"],
                "correct_margin": measure["correct_margin"],
                "replay_correct_ce": replay_measure["correct_ce"],
                "correct_ce": measure["correct_ce"],
                "answer_logit": measure["answer_logit"], "other_logit": measure["other_logit"],
                "log_normalizer": measure["log_normalizer"],
                "n": float(n_value), "d": float(d_value), "q": float(q_value),
                "insertion_activity": float(np.median(np.linalg.norm(actual.astype(np.float64), axis=-1))),
                "per_site_delta_norms": [float(value) for value in np.linalg.norm(actual.astype(np.float64), axis=-1)],
                "vocab_squared_difference_sum": float(np.square(difference).sum()),
                "vocab_size": VOCAB,
                "vocab_rms": float(math.sqrt(float(np.square(difference).mean()))),
            })
    return records


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(_json_bytes(dict(row)) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    return {"records": len(rows), "byte_length": path.stat().st_size, "sha256": sha256_file(path)}


def write_complete_phase_evidence(
    evidence_root: Path, phase: str, execution: Mapping[str, object],
    bundle: Mapping[str, object], outputs: Mapping[str, Path], records: Sequence[Mapping[str, object]],
    score_report: Mapping[str, object], fit_scales: Mapping[str, object], r585: object,
) -> dict[str, object]:
    """Materialize the complete rectangular phase evidence from per-call raw bytes."""
    phase_root = evidence_root / phase
    phase_root.mkdir(parents=True, exist_ok=False)
    factory = make_context_factory(execution, bundle)
    endpoint_calls = [call for call in bundle["calls"] if call["call_kind"] == "endpoint"]
    native_calls = [call for call in bundle["calls"] if call["call_kind"] == "native"]
    endpoint_arrays = [load_call_arrays(outputs[str(call["call_id"])]) for call in endpoint_calls]
    native_arrays = [load_call_arrays(outputs[str(call["call_id"])]) for call in native_calls]
    descriptors: dict[str, object] = {}

    for filename, payload in (
        ("call_manifest.json", bundle["calls"]),
        ("token_manifest.json", bundle["token_records"]),
    ):
        path = phase_root / filename
        _write_bytes(path, _json_bytes(payload))
        descriptors[filename] = {
            "records": len(payload), "byte_length": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def save(name: str, value: np.ndarray) -> None:
        descriptors[name] = _write_npy(phase_root / name, value)

    save("endpoint_tokens.npy", np.concatenate([row["tokens.npy"] for row in endpoint_arrays]))
    for name in (
        "factor_e.npy", "factor_u.npy", "support.npy", "native_equality_term.npy",
        "factorized_equality_term.npy", "native_non_equality_remainder.npy",
        "native_head_write.npy", "independent_full_native_write.npy",
    ):
        save(name, np.concatenate([row[name] for row in endpoint_arrays]))
    save("directed_tokens.npy", np.concatenate([row["tokens.npy"] for row in native_arrays]))
    save("directed_live_e.npy", np.concatenate([row["live_e.npy"] for row in native_arrays]))
    save("directed_live_u.npy", np.concatenate([row["live_u.npy"] for row in native_arrays]))

    hook_path = phase_root / "hook_deltas.npy"
    logit_path = phase_root / "logit_differences.npy"
    nd = PHASE_COUNTS[phase]["directions"]
    hooks = np.lib.format.open_memmap(hook_path, mode="w+", dtype="<f4", shape=(nd, 4, 4, RESIDUAL))
    differences = np.lib.format.open_memmap(logit_path, mode="w+", dtype="<f4", shape=(nd, 4, VOCAB))
    offset = 0
    for call in native_calls:
        chunk = int(call["chunk_index"]); b = int(call["batch_size"])
        by_kind = {
            kind: load_call_arrays(outputs[f"{phase}:directed:{chunk:04d}:{kind}"])
            for kind in DIRECTED_KINDS
        }
        replay_logits = by_kind["replay"]["logits.npy"]
        for arm_index, arm in enumerate(MACHINE_ARMS):
            hooks[offset:offset + b, arm_index] = by_kind[arm]["hook_deltas.npy"]
        differences[offset:offset + b, 0] = by_kind["native"]["logits.npy"] - replay_logits
        for index, arm in enumerate(("score", "payload", "joint"), start=1):
            differences[offset:offset + b, index] = by_kind[arm]["logits.npy"] - replay_logits
        offset += b
    hooks.flush(); differences.flush(); del hooks, differences
    descriptors["hook_deltas.npy"] = {
        "dtype": "float32", "shape": [nd, 4, 4, RESIDUAL],
        "byte_length": hook_path.stat().st_size, "sha256": sha256_file(hook_path),
    }
    descriptors["logit_differences.npy"] = {
        "dtype": "float32", "shape": [nd, 4, VOCAB],
        "byte_length": logit_path.stat().st_size, "sha256": sha256_file(logit_path),
    }
    endpoint_rows = []
    for row in (row for row in execution["endpoints"] if row["split"] == phase):
        call_id, local = factory.endpoint_location[str(row["endpoint_id"])]
        logits = load_call_arrays(outputs[call_id])["logits.npy"][local]
        endpoint_rows.append({
            **row,
            "native_measurement": _measurement(
                logits, int(row["answer_id"]), int(row["other_answer_id"])
            ),
            "array_index": len(endpoint_rows),
        })
    descriptors["endpoint_records.jsonl"] = _write_jsonl(
        phase_root / "endpoint_records.jsonl", endpoint_rows
    )
    evidence_records = []
    for directed_id in sorted({str(row["directed_id"]) for row in records}):
        members = [row for row in records if row["directed_id"] == directed_id]
        base = {key: value for key, value in members[0].items() if key not in {
            "arm", "correct_margin", "correct_ce", "answer_logit", "other_logit",
            "log_normalizer", "n", "d", "q", "insertion_activity",
            "per_site_delta_norms", "vocab_squared_difference_sum", "vocab_size", "vocab_rms",
        }}
        base["arms"] = {
            str(row["arm"]): {key: row[key] for key in (
                "correct_margin", "correct_ce", "answer_logit", "other_logit",
                "log_normalizer", "n", "d", "q", "insertion_activity",
                "per_site_delta_norms", "vocab_squared_difference_sum", "vocab_size", "vocab_rms",
            )} for row in members
        }
        for arm in base["arms"].values():
            arm["c"] = arm["correct_margin"]
        chunk, local = factory.direction_location[directed_id]
        authority_row = next(
            row for row in execution["directions"] if row["directed_id"] == directed_id
        )
        answer = int(authority_row["recipient_answer_id"])
        other = int(authority_row["recipient_other_answer_id"])
        for condition in ("native", "replay"):
            logits = load_call_arrays(
                outputs[f"{phase}:directed:{chunk:04d}:{condition}"]
            )["logits.npy"][local]
            base[condition] = _measurement(logits, answer, other)
        evidence_records.append(base)
    descriptors["directed_records.jsonl"] = _write_jsonl(
        phase_root / "directed_records.jsonl", evidence_records
    )
    rows_document = r585.strict_load_json(r585.ROWS)
    authority_rows = [
        row for row in rows_document["rows"]
        if row["split"] == phase and row["family_id"] in r585.load_manifest().INCLUDED_FAMILIES
    ]
    descriptors["authority.jsonl"] = _write_jsonl(phase_root / "authority.jsonl", authority_rows)
    bootstrap = {
        "bootstrap_cells": [cell for cell in execution["bootstrap_cells"] if str(cell["cell_id"]).startswith(phase + "|")],
        "score_report": score_report, "fit_scales": fit_scales,
    }
    bootstrap_path = phase_root / "bootstrap_cells.json"
    _write_bytes(bootstrap_path, _json_bytes(bootstrap))
    descriptors["bootstrap_cells.json"] = {
        "byte_length": bootstrap_path.stat().st_size, "sha256": sha256_file(bootstrap_path),
    }
    expected = phase_evidence_schema(phase)
    if descriptors["authority.jsonl"]["records"] != expected["authority.jsonl"]["records"] or (
        descriptors["endpoint_records.jsonl"]["records"] != expected["endpoint_records.jsonl"]["records"]
    ) or descriptors["directed_records.jsonl"]["records"] != PHASE_COUNTS[phase]["directions"]:
        raise RuntimeError("complete phase JSONL census changed")
    return descriptors


def publish_normal(
    stage: Path, result: Mapping[str, object], *, public_root: Path = ROOT,
) -> dict[str, object]:
    result_path = stage / "result.json"
    _write_bytes(result_path, _json_bytes(dict(result)))
    evidence = stage / "evidence"
    files = {}
    for path in sorted(evidence.rglob("*")):
        if not path.is_file():
            continue
        descriptor: dict[str, object] = {
            "byte_length": path.stat().st_size, "sha256": sha256_file(path)
        }
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            descriptor.update(dtype=str(array.dtype), shape=list(array.shape))
        elif path.suffix == ".jsonl":
            lines = path.read_bytes().splitlines()
            descriptor.update(
                records=len(lines),
                row_order_sha256=sha256_bytes(b"\n".join(lines)),
            )
        files[str(path.relative_to(evidence))] = descriptor
    receipt = {
        "schema": "induction_centered_fixed_geometry_rung592_receipt_v1",
        "result_sha256": sha256_file(result_path), "evidence_files": files,
    }
    receipt_path = stage / "receipt.json"
    _write_bytes(receipt_path, _json_bytes(receipt, pretty=True))
    targets = (public_root / NORMAL_EVIDENCE.name, public_root / NORMAL_RESULT.name, public_root / NORMAL_RECEIPT.name)
    if any(path.exists() for path in targets):
        raise RuntimeError("normal namespace already occupied")
    _fsync_directory(evidence); _fsync_directory(stage)
    os.replace(evidence, targets[0]); _fsync_directory(public_root)
    os.replace(result_path, targets[1]); _fsync_directory(public_root)
    os.replace(receipt_path, targets[2]); _fsync_directory(public_root)
    return receipt


def _empty_scientific_failures() -> dict[str, list[str]]:
    return {name: [] for name in (
        "invalid_instrument", "native_denominator_or_scale_null", "factor_capacity_null",
        "factorization_not_identified", "insufficient_active_controls",
        "broad_contextual_equality_write",
    )}


def run_science(*, public_root: Path = ROOT) -> dict[str, object]:
    """Run the full approved FIT-first path.  Never called by model-free tests."""
    started = time.time()
    implementation_sha256 = globals().get("__r592_immutable_sha256__")
    if implementation_sha256 is None:
        implementation_sha256 = sha256_file(Path(__file__))
    public = tuple(public_root / path.name for path in PUBLIC_NAMESPACES)
    occupied = [str(path) for path in public if path.exists()]
    if occupied:
        raise RuntimeError(f"R592 public namespace occupied: {occupied}")
    r585, execution = load_authority()
    fit_bundle = build_phase_manifest(execution, "FIT")
    select_bundle = build_phase_manifest(execution, "SELECT")
    runtime = _immutable_module(RUNTIME, SOURCE_HASHES[RUNTIME], "r592_pinned_model_runtime")
    # This constructor is the first permitted torch/checkpoint/CUDA boundary.
    executor = runtime.R592ModelExecutor(types.SimpleNamespace(**globals()), r585)
    r585 = executor.r585
    stage = Path(tempfile.mkdtemp(prefix=".r592-stage-", dir=public_root))
    (stage / "evidence").mkdir(exist_ok=True)
    fit_scales = None
    split_scores: dict[str, object] = {}
    failure_classes: dict[str, list[str]] = {}
    evaluated: list[str] = []
    calls = 0
    try:
        fit = run_manifest_calls(
            executor, fit_bundle, make_context_factory(execution, fit_bundle),
            stage=stage, public_root=public_root,
        )
        if fit["status"] == "invalid":
            calls = len(fit["diagnostic"]["executed_call_ids"])
            if stage.exists(): shutil.rmtree(stage)
            return {"status": "invalid_diagnostic", "model_forwards": calls}
        calls = PHASE_COUNTS["FIT"]["calls"]; evaluated.append("FIT")
        fit_records = derive_scientific_records(execution, fit_bundle, fit["outputs"])
        fit_scales = r585.compute_fit_scales(fit_records, execution["manifests"])
        fit_report, fit_failures = r585.score_split(
            fit_records, "FIT", execution["manifests"], fit_scales, replicates=2_000
        )
        split_scores["FIT"] = fit_report
        failure_classes = {name: list(values) for name, values in fit_failures.items()}
        write_complete_phase_evidence(
            stage / "evidence", "FIT", execution, fit_bundle, fit["outputs"],
            fit_records, fit_report, fit_scales, r585,
        )
        shutil.rmtree(stage / "evidence" / "calls")
        if not any(failure_classes.values()):
            select = run_manifest_calls(
                executor, select_bundle, make_context_factory(execution, select_bundle),
                stage=stage, public_root=public_root,
            )
            if select["status"] == "invalid":
                calls = PHASE_COUNTS["FIT"]["calls"] + len(select["diagnostic"]["executed_call_ids"])
                if stage.exists(): shutil.rmtree(stage)
                return {"status": "invalid_diagnostic", "model_forwards": calls}
            calls = PHASE_COUNTS["FIT"]["calls"] + PHASE_COUNTS["SELECT"]["calls"]
            evaluated.append("SELECT")
            select_records = derive_scientific_records(execution, select_bundle, select["outputs"])
            select_report, select_failures = r585.score_split(
                select_records, "SELECT", execution["manifests"], fit_scales, replicates=2_000
            )
            split_scores["SELECT"] = select_report
            for name, values in select_failures.items():
                failure_classes["select_" + name] = list(values)
            write_complete_phase_evidence(
                stage / "evidence", "SELECT", execution, select_bundle, select["outputs"],
                select_records, select_report, fit_scales, r585,
            )
            shutil.rmtree(stage / "evidence" / "calls")
        terminal = r585.terminal_from_failures(evaluated, failure_classes)
        result = {
            "schema": SCHEMA, "rung": 592,
            "stage": "centered_fixed_geometry_equality_factor_interchange",
            "evidence_level": "prospective_partial_output_factor_identification",
            "terminal": terminal,
            "failure_classes": failure_classes,
            "failed_clauses": [
                value for name in sorted(failure_classes) for value in failure_classes[name]
            ],
            "split_scores": split_scores, "evaluated_splits": evaluated,
            "operational_arm_labels": OPERATIONAL_ARM_LABELS,
            "machine_arm_order": list(MACHINE_ARMS),
            "difference_order": list(DIFFERENCE_ORDER),
            "claim_boundary": "partial_output_space_equality_factor_interchange_only",
            "literal_remove_insert_claimed": False,
            "select_opened": evaluated == ["FIT", "SELECT"],
            "final_opened": False, "ood_opened": False,
            "model_forwards": calls, "model_backwards": 0, "model_weights_updated": False,
            "checkpoint_weights_sha256": executor.checkpoint_sha256,
            "implementation_sha256": implementation_sha256,
            "source_sha256": verify_sources(),
            "call_manifest_sha256": {
                "FIT": fit_bundle["call_manifest_sha256"],
                **({"SELECT": select_bundle["call_manifest_sha256"]} if "SELECT" in evaluated else {}),
            },
            "authority_sha256": {
                key: execution[key] for key in execution if key.endswith("_sha256")
            },
            "elapsed_seconds": float(time.time() - started),
        }
        publish_normal(stage, result, public_root=public_root)
        if stage.exists(): stage.rmdir()
        return result
    except Exception:
        # A raised/incomplete call is a hard abort.  No temporary bytes are
        # promoted; best-effort cleanup leaves all public namespaces absent.
        if stage.exists(): shutil.rmtree(stage)
        raise


def build_dryrun() -> dict[str, object]:
    r585, execution = load_authority()
    fit = build_phase_manifest(execution, "FIT")
    select = build_phase_manifest(execution, "SELECT")
    manifests = execution["manifests"]
    bootstrap = execution["bootstrap_cells"]
    machine_ids = sorted({
        arm for cell in bootstrap for arm in MACHINE_ARMS
        if f"|{arm}|" in str(cell.get("cell_id", ""))
    })
    # Directly execute the exact centered formula on a planted non-degenerate fixture.
    rng = np.random.default_rng(592)
    ex = rng.normal(size=(3, 4, 2)).astype("<f4")
    ux = rng.normal(size=(3, 4, 2, RESIDUAL)).astype("<f4")
    ey = rng.normal(size=(3, 4, 2)).astype("<f4")
    uy = rng.normal(size=(3, 4, 2, RESIDUAL)).astype("<f4")
    deltas = centered_deltas(ex, ux, ey, uy)
    if not np.array_equal(deltas[:, 0], np.zeros_like(deltas[:, 0])):
        raise RuntimeError("replay is not bitwise zero")
    return {
        "schema": DRYRUN_SCHEMA,
        "status": "prospective_model_free_only",
        "scientific_terminal": None,
        "source_sha256": verify_sources(),
        "authority_hashes": {
            key: execution[key] for key in execution if key.endswith("_sha256")
        },
        "phase_counts": PHASE_COUNTS,
        "fit_call_manifest_sha256": fit["call_manifest_sha256"],
        "select_call_manifest_sha256": select["call_manifest_sha256"],
        "fit_token_manifest_sha256": fit["token_manifest_sha256"],
        "select_token_manifest_sha256": select["token_manifest_sha256"],
        "select_tail_batch_sizes": [call["batch_size"] for call in select["calls"][-5:]],
        "operational_arm_labels": OPERATIONAL_ARM_LABELS,
        "machine_arm_order": list(MACHINE_ARMS),
        "difference_order": list(DIFFERENCE_ORDER),
        "bootstrap_cell_count_by_split": {
            phase: sum(str(cell["cell_id"]).startswith(phase + "|") for cell in bootstrap)
            for phase in PHASE_COUNTS
        },
        "manifest_cell_counts": {
            phase: {
                "targets": sum(cell["split"] == phase for cell in manifests["target_cells"]),
                "controls": sum(cell["split"] == phase for cell in manifests["control_cells"]),
            } for phase in PHASE_COUNTS
        },
        "legacy_machine_ids_observed": machine_ids,
        "centered_fixture_sha256": sha256_bytes(deltas.tobytes(order="C")),
        "centered_mixed_identity_max_abs": mixed_identity_error(deltas, ex, ux, ey, uy),
        "evidence_schemas": {phase: phase_evidence_schema(phase) for phase in PHASE_COUNTS},
        "invalid_predicate_order": list(PREDICATE_ORDER),
        "model_forwards": 0,
        "registered_max_model_forwards": 961,
        "model_backwards": 0,
        "model_weights_updated": False,
        "select_opened": False,
        "final_opened": False,
        "ood_opened": False,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(build_dryrun(), indent=2, sort_keys=True, allow_nan=False))
        return
    result = run_science()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
