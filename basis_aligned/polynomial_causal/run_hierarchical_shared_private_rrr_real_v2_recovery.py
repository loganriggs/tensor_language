#!/usr/bin/env python3
"""One-change recovery for hierarchical shared/private RRR v1 JSON containers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_hierarchical_shared_private_rrr_real_v1 as v1
import run_shared_output_rrr_real_v1 as base


RUNNER = HERE / "run_hierarchical_shared_private_rrr_real_v2_recovery.py"
TEST = HERE / "test_run_hierarchical_shared_private_rrr_real_v2_recovery.py"
PREREG = HERE / "HIERARCHICAL_SHARED_PRIVATE_RRR_REAL_V2_RECOVERY_PREREGISTRATION.md"

AUTHORITY = HERE / "hierarchical_shared_private_rrr_real_v2_recovery_authority.json"
RESULTS = HERE / "hierarchical_shared_private_rrr_real_v2_recovery_results.json"
FAILURE = HERE / "hierarchical_shared_private_rrr_real_v2_recovery_failure.json"
RECEIPT = HERE / "hierarchical_shared_private_rrr_real_v2_recovery_receipt.json"
LOCK = Path("/workspace/runs/.hierarchical_shared_private_rrr_real_v2_recovery.lock")
PROTOCOL_VERSION = "hierarchical_shared_private_v2_recovery"

V1_AUTHORITY = HERE / "hierarchical_shared_private_rrr_real_v1_authority.json"
V1_RESULTS = HERE / "hierarchical_shared_private_rrr_real_v1_results.json"
V1_FAILURE = HERE / "hierarchical_shared_private_rrr_real_v1_failure.json"
V1_RECEIPT = HERE / "hierarchical_shared_private_rrr_real_v1_receipt.json"
V1_HASHES = {
    str(V1_AUTHORITY.relative_to(ROOT)):
        "558d316eb5fdb4a4249eb58cdd5c2b80f0005873cdbccf517dfede7226c4d11c",
    str(V1_RESULTS.relative_to(ROOT)):
        "86315dcc855e9a27958b6abfd50ed5c6b7bb7108f00fe3684bfbf624405a772d",
    str(V1_FAILURE.relative_to(ROOT)):
        "054db06c03525b3f78eefdd9ed8e0fa3daf3868175460c76a95e39b875ebc35c",
}

SOURCE_PATHS = tuple(dict.fromkeys((
    *v1.SOURCE_PATHS,
    *(str(path.relative_to(ROOT)) for path in (RUNNER, TEST, PREREG)),
)))
FILE_PINS = {**v1.FILE_PINS, **V1_HASHES}
_INHERITED_V1_VERIFY_FROZEN_INPUTS = v1.verify_frozen_inputs


def _read_pinned_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    before = base.file_sha256(path)
    payload = path.read_bytes()
    payload_hash = hashlib.sha256(payload).hexdigest()
    after = base.file_sha256(path)
    if before != expected_sha256 or payload_hash != expected_sha256 or after != expected_sha256:
        raise RuntimeError(f"hierarchical RRR v1 parent changed while reading: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("hierarchical RRR v1 parent is not a JSON object")
    return value


def _semantic_replay_v1(
    authority: Mapping[str, Any], result: Mapping[str, Any], failure: Mapping[str, Any],
) -> None:
    saved = {name: getattr(base, name) for name in v1._BASE_DEFAULTS}
    try:
        for name, value in v1._BASE_DEFAULTS.items():
            setattr(base, name, value)
        v1.configure_base()
        base.semantic_validate_result(result, authority)
    finally:
        for name, value in saved.items():
            setattr(base, name, value)
    expected_failure = {
        "authority_exists": True,
        "authority_file_sha256": V1_HASHES[str(V1_AUTHORITY.relative_to(ROOT))],
        "error": "shared-RRR result replay changed",
        "error_type": "RuntimeError",
        "receipt_exists": False,
        "results_exists": True,
        "results_file_sha256": V1_HASHES[str(V1_RESULTS.relative_to(ROOT))],
        "schema": "shared_output_rrr_real_hierarchical_shared_private_v1_failure",
        "status": "terminal_failure_no_receipt",
    }
    if dict(failure) != expected_failure:
        raise RuntimeError("hierarchical RRR v1 failure semantics changed")


def _require_v1_receipt_absent() -> None:
    if V1_RECEIPT.exists():
        raise RuntimeError("hierarchical RRR v1 receipt unexpectedly exists")


def _verify_v1_terminal_files() -> None:
    """Close the cross-parent window after the three stable individual reads."""
    observed = {
        relative: base.file_sha256(ROOT / relative) for relative in V1_HASHES
    }
    if observed != V1_HASHES:
        raise RuntimeError("hierarchical RRR v1 terminal parent set changed")
    _require_v1_receipt_absent()


def verify_spent_v1() -> dict[str, Any]:
    _require_v1_receipt_absent()
    authority = _read_pinned_json(
        V1_AUTHORITY, V1_HASHES[str(V1_AUTHORITY.relative_to(ROOT))],
    )
    result = _read_pinned_json(
        V1_RESULTS, V1_HASHES[str(V1_RESULTS.relative_to(ROOT))],
    )
    failure = _read_pinned_json(
        V1_FAILURE, V1_HASHES[str(V1_FAILURE.relative_to(ROOT))],
    )
    if result.get("authority_sha256") != authority.get("authority_sha256") or authority.get(
        "schema"
    ) != "hierarchical_shared_private_rrr_real_v1_authority" or result.get(
        "schema"
    ) != "shared_output_rrr_real_hierarchical_shared_private_v1_results":
        raise RuntimeError("hierarchical RRR v1 authority/result join changed")
    _semantic_replay_v1(authority, result, failure)
    _verify_v1_terminal_files()
    return {
        "authority_file_sha256": V1_HASHES[str(V1_AUTHORITY.relative_to(ROOT))],
        "authority_sha256": authority["authority_sha256"],
        "results_file_sha256": V1_HASHES[str(V1_RESULTS.relative_to(ROOT))],
        "results_logical_sha256": base.logical_sha256(result),
        "failure_file_sha256": V1_HASHES[str(V1_FAILURE.relative_to(ROOT))],
        "receipt_absent": True,
        "failure_stage": "strict post-publication JSON equality before receipt",
        "scientific_values_used_for_recovery_selection": False,
        "only_change": "JSON-normalize each unchanged program diagnostics before result assembly",
    }


def json_normalize(value: Any) -> Any:
    normalized = json.loads(json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ))
    if json.loads(json.dumps(
        normalized, sort_keys=True, allow_nan=False, separators=(",", ":"),
    )) != normalized:
        raise RuntimeError("hierarchical RRR diagnostics are not JSON-idempotent")
    return normalized


def fit_program(descriptor: Mapping[str, Any], state: base.SpectralState):
    program = v1.fit_program(descriptor, state)
    program.diagnostics = json_normalize(program.diagnostics)
    v1.semantic_validate_diagnostics(program.diagnostics, descriptor)
    return program


def authority_payload(
    source: Mapping[str, Any], inputs: Mapping[str, str], checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = verify_spent_v1()
    parent = v1.authority_payload(source, inputs, checkpoint)
    protocol = dict(parent["protocol"])
    protocol["recovery_parent"] = lineage
    protocol["only_execution_change"] = (
        "json.loads(json.dumps(program.diagnostics, sort_keys=True, allow_nan=False))"
    )
    body = {
        "schema": "hierarchical_shared_private_rrr_real_v2_recovery_authority",
        "status": "frozen_before_any_row_tensor_or_model_load",
        "source_closure": dict(source),
        "input_file_sha256s": dict(inputs),
        "checkpoint": dict(checkpoint),
        "protocol": protocol,
        "outputs": {
            "results": str(RESULTS), "failure": str(FAILURE), "receipt": str(RECEIPT),
        },
    }
    return {**body, "authority_sha256": base.logical_sha256(body)}


def verify_frozen_inputs(
    value: Mapping[str, Any], *, verify_checkpoint_hash: bool,
) -> None:
    _INHERITED_V1_VERIFY_FROZEN_INPUTS(
        value, verify_checkpoint_hash=verify_checkpoint_hash,
    )
    verify_spent_v1()


_BASE_DEFAULTS = {name: getattr(base, name) for name in v1._BASE_DEFAULTS}


def configure_base() -> None:
    v1.configure_base()
    assignments = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "AUTHORITY": AUTHORITY, "RESULTS": RESULTS, "FAILURE": FAILURE,
        "RECEIPT": RECEIPT, "LOCK": LOCK,
        "SOURCE_PATHS": SOURCE_PATHS, "FILE_PINS": FILE_PINS,
        "authority_payload": authority_payload,
        "fit_program": fit_program,
        "verify_frozen_inputs": verify_frozen_inputs,
    }
    for name, value in assignments.items():
        setattr(base, name, value)


def restore_base_defaults() -> None:
    for name, value in _BASE_DEFAULTS.items():
        setattr(base, name, value)


def run(*, device: str = "cuda") -> dict[str, Any]:
    configure_base()
    try:
        verify_spent_v1()
        return base.run(device=device)
    finally:
        restore_base_defaults()


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
