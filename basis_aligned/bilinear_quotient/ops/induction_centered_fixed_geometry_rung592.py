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
import sys
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

SOURCE_HASHES = {
    R591: "fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc",
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    PREREG: "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a",
    AMENDMENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    DIAGNOSTIC_AMENDMENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    MASK_AMENDMENT: "f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc",
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
    return r591.load_authority()


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
    if os.environ.get("BQLIB_DRYRUN") != "1":
        raise SystemExit("R592 producer is outcome-locked; managed reviewed runtime required")
    print(json.dumps(build_dryrun(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
