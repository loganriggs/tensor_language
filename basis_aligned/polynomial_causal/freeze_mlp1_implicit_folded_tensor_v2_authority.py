#!/usr/bin/env python3
"""Freeze the isolated v2 dtype-retry authority without deserializing weights."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import freeze_mlp1_implicit_folded_tensor_v1_authority as v1_freeze  # noqa: E402
import tensor_bilin18_tangent_authority as lifecycle  # noqa: E402


PROTOCOL = HERE / "MLP1_IMPLICIT_FOLDED_TENSOR_V2_RETRY_PROTOCOL.json"
ERRATUM = HERE / "MLP1_IMPLICIT_FOLDED_TENSOR_V2_ERRATUM.md"
COLLECTOR = HERE / "collect_mlp1_implicit_folded_tensor_v2.py"
TEST = HERE / "test_mlp1_implicit_folded_tensor_v2_lifecycle.py"
AUTHORITY = HERE / "mlp1_implicit_folded_tensor_v2_authority.json"
RESULT = HERE / "mlp1_implicit_folded_tensor_v2_results.json"
OUTCOME_AUTHORITY = HERE / "mlp1_implicit_folded_tensor_v2_outcome_authority.json"
FAILURE = HERE / "mlp1_implicit_folded_tensor_v2_failure.json"
RUN_LOCK = HERE / ".mlp1_implicit_folded_tensor_v2.lock"
PARENT_AUTHORITY = v1_freeze.AUTHORITY
PARENT_FAILURE = v1_freeze.FAILURE
PARENT_RESULT = v1_freeze.RESULT
PARENT_OUTCOME_AUTHORITY = v1_freeze.OUTCOME_AUTHORITY
PARENT_AUTHORITY_SHA256 = "7b2cebe982559a3e232e073d238b984df6246461b92c622e0275bc9279b8b468"
PARENT_FAILURE_SHA256 = "350d7dc7bb4ec3207853abaa5db83da57b58b1613898553320824245c3f48526"


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


SOURCES = _deduplicate((Path(__file__), COLLECTOR, TEST, PROTOCOL, ERRATUM, *v1_freeze.SOURCES))

EXPECTED_PROTOCOL = {
    "schema_version": 1,
    "experiment_id": "bilin18_mlp1_implicit_folded_tensor_v2_dtype_retry",
    "status": "prospective_retry_after_v1_fail_closed_before_scientific_outcome",
    "parent_v1_authority_sha256": PARENT_AUTHORITY_SHA256,
    "parent_v1_failure_sha256": PARENT_FAILURE_SHA256,
    "parent_failure_scope": (
        "v1 stopped during full checkpoint state-tree metadata validation at "
        "transformer.wte.weight; no MLP1 tensor was extracted, no Gram/spectrum/core/result "
        "was computed, and no partial result exists"
    ),
    "scientific_plan": (
        "exactly inherited from MLP1 implicit folded-tensor v1; no rank, threshold, core, "
        "price, or claim rule changes"
    ),
    "native_dtype_schema": {
        "torch.bfloat16": [
            "transformer.wte.weight", "transformer.h.*.lambdas",
            "transformer.h.*.attn.lamb", "transformer.h.*.mlp.Down_bias",
        ],
        "torch.float32": "every other key in the exact 218-key TT.GPT state tree",
    },
    "state_key_counts": {
        "total": 218, "torch.bfloat16": 55, "torch.float32": 163,
    },
    "mlp1_factor_dtypes": {
        "Left.weight": "torch.float32", "Right.weight": "torch.float32",
        "Down.weight": "torch.float32", "Down_bias": "torch.bfloat16",
    },
    "bias_rule": (
        "Hash and record the owned native bf16 MLP1 Down_bias before conversion. Make a "
        "distinct float64 analysis copy, hash it separately, and pass only that copy to "
        "the bias-separated diagnostic. Bias is excluded from balancing, mode Grams, "
        "spectra, HOSVD bases, and projected cores."
    ),
    "checkpoint_load_rule": (
        "After v2 source/weight authority only: local "
        "torch.load(weights_only=True,mmap=True,map_location=cpu), exact 218-key shape and "
        "native-dtype validation, clone only MLP1 factors, then delete the state tree."
    ),
    "publication_order": [
        "v2_source_weight_authority", "v2_nonauthoritative_result",
        "v2_last_written_outcome_authority",
    ],
    "failure_rule": (
        "All v1 artifacts remain immutable. V2 failure is create-only, may bind a partial "
        "v2 result, and can never authorize or relabel either version."
    ),
    "claim_boundary": v1_freeze.EXPECTED_PROTOCOL["claim_boundary"],
}


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return lifecycle.canonical_sha256(value)


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text())
    if value != EXPECTED_PROTOCOL:
        raise RuntimeError("MLP1 folded-tensor v2 retry protocol changed")
    return value


def namespace_contract() -> dict[str, str]:
    return {
        "source_weight_authority": str(AUTHORITY.resolve().relative_to(ROOT)),
        "result": str(RESULT.resolve().relative_to(ROOT)),
        "outcome_authority": str(OUTCOME_AUTHORITY.resolve().relative_to(ROOT)),
        "failure": str(FAILURE.resolve().relative_to(ROOT)),
        "run_lock": str(RUN_LOCK.resolve().relative_to(ROOT)),
    }


def namespace_outputs() -> tuple[Path, ...]:
    return AUTHORITY, RESULT, OUTCOME_AUTHORITY, FAILURE


def validate_parent_failure() -> dict[str, Any]:
    if file_sha256(PARENT_AUTHORITY) != PARENT_AUTHORITY_SHA256:
        raise RuntimeError("preserved v1 authority changed")
    if file_sha256(PARENT_FAILURE) != PARENT_FAILURE_SHA256:
        raise RuntimeError("preserved v1 failure changed")
    if PARENT_RESULT.exists() or PARENT_OUTCOME_AUTHORITY.exists():
        raise RuntimeError("preserved v1 absent result namespace changed")
    authority = json.loads(PARENT_AUTHORITY.read_text())
    failure = json.loads(PARENT_FAILURE.read_text())
    expected_failure = {
        "error_message": "checkpoint state-tree metadata changed: transformer.wte.weight",
        "error_type": "RuntimeError",
        "partial_result_sha256": None,
        "result_authorized": False,
        "schema_version": 1,
        "source_weight_authority_sha256": PARENT_AUTHORITY_SHA256,
        "status": "failed_nonauthoritative",
    }
    if (
        authority.get("status") != "frozen_before_any_mlp1_checkpoint_tensor_deserialization"
        or authority.get("checkpoint_deserialized") is not False
        or authority.get("result_computed") is not False
        or failure != expected_failure
    ):
        raise RuntimeError("preserved v1 failure semantics changed")
    return {
        "authority_path": str(PARENT_AUTHORITY.resolve().relative_to(ROOT)),
        "authority_sha256": PARENT_AUTHORITY_SHA256,
        "failure_path": str(PARENT_FAILURE.resolve().relative_to(ROOT)),
        "failure_sha256": PARENT_FAILURE_SHA256,
        "result_absent": True,
        "outcome_authority_absent": True,
    }


def source_snapshot() -> dict[str, Any]:
    lifecycle.require_committed_sources(SOURCES)
    hashes = {str(path.resolve().relative_to(ROOT)): file_sha256(path) for path in SOURCES}
    if len(hashes) != len(SOURCES):
        raise RuntimeError("MLP1 folded-tensor v2 source closure contains duplicates")
    result = {"source_hashes": hashes, "git": lifecycle.git_identity(SOURCES)}
    result["fingerprint"] = canonical_sha256(result)
    return result


def protected_snapshot() -> dict[str, Any]:
    result = {
        "sources": source_snapshot(),
        "checkpoint": v1_freeze.checkpoint_snapshot(),
        "parent_v1": validate_parent_failure(),
        "protocol_sha256": file_sha256(PROTOCOL),
        "erratum_sha256": file_sha256(ERRATUM),
        "v1_execution_protocol_sha256": file_sha256(v1_freeze.PROTOCOL),
        "v1_preregistration_sha256": file_sha256(v1_freeze.PREREG),
        "decision_addendum_sha256": file_sha256(v1_freeze.DECISION_ADDENDUM),
    }
    result["fingerprint"] = canonical_sha256(result)
    return result


def configure_cpu_runtime() -> dict[str, Any]:
    return v1_freeze.configure_cpu_runtime()


def require_empty_namespace() -> None:
    if any(path.exists() for path in namespace_outputs()):
        raise RuntimeError("MLP1 folded-tensor v2 namespace is already frozen or spent")


def publication_guard(lock: lifecycle.RunLock, snapshot: Mapping[str, Any]) -> None:
    lock.assert_owned()
    require_empty_namespace()
    if protected_snapshot() != dict(snapshot):
        raise RuntimeError("MLP1 folded-tensor v2 protected state changed before authority")


def build_authority(lock: lifecycle.RunLock, runtime: Mapping[str, Any]) -> dict[str, Any]:
    lock.assert_owned()
    require_empty_namespace()
    snapshot = protected_snapshot()
    payload = {
        "schema_version": 1,
        "receipt_kind": "mlp1_implicit_folded_tensor_v2_source_weight_authority",
        "status": "frozen_before_any_v2_checkpoint_tensor_deserialization",
        "protected_snapshot": snapshot,
        "runtime": dict(runtime),
        "protocol": load_protocol(),
        "inherited_v1_execution_plan": v1_freeze.load_protocol(),
        "namespace": namespace_contract(),
        "authorized_operation": "one CPU v2 dtype-corrected MLP1 folded-tensor diagnostic",
        "rows_loaded": False,
        "checkpoint_deserialized": False,
        "mlp1_tensors_extracted": False,
        "mode_grams_computed": False,
        "spectra_computed": False,
        "projected_cores_computed": False,
        "result_computed": False,
    }
    lifecycle.publish_json_create_only(
        AUTHORITY, payload, ownership_check=lambda: publication_guard(lock, snapshot),
    )
    return payload


def validate_authority(
    value: Any, *, snapshot: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    required = {
        "schema_version", "receipt_kind", "status", "protected_snapshot", "runtime",
        "protocol", "inherited_v1_execution_plan", "namespace", "authorized_operation",
        "rows_loaded", "checkpoint_deserialized", "mlp1_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "projected_cores_computed",
        "result_computed",
    }
    spent = {
        "rows_loaded", "checkpoint_deserialized", "mlp1_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "projected_cores_computed",
        "result_computed",
    }
    if (
        not isinstance(value, dict) or set(value) != required
        or value["schema_version"] != 1
        or value["receipt_kind"] != "mlp1_implicit_folded_tensor_v2_source_weight_authority"
        or value["status"] != "frozen_before_any_v2_checkpoint_tensor_deserialization"
        or value["protected_snapshot"] != dict(snapshot)
        or value["runtime"] != dict(runtime)
        or value["protocol"] != load_protocol()
        or value["inherited_v1_execution_plan"] != v1_freeze.load_protocol()
        or value["namespace"] != namespace_contract()
        or value["authorized_operation"] != (
            "one CPU v2 dtype-corrected MLP1 folded-tensor diagnostic"
        )
        or any(value[field] is not False for field in spent)
    ):
        raise RuntimeError("MLP1 folded-tensor v2 authority is malformed")


def freeze() -> dict[str, Any]:
    runtime = configure_cpu_runtime()
    with lifecycle.exclusive_run_lock(RUN_LOCK) as lock:
        return build_authority(lock, runtime)


def main() -> None:
    payload = freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
