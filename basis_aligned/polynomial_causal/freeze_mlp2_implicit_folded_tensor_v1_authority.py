#!/usr/bin/env python3
"""Freeze MLP2 weight-diagnostic source/checkpoint authority without loading tensors."""

from __future__ import annotations

import json
from importlib.metadata import version as package_version
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# Fix standalone production BLAS configuration before NumPy or Torch imports.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from threadpoolctl import threadpool_info  # noqa: E402
import freeze_mlp1_implicit_folded_tensor_v2_authority as mlp1_freeze  # noqa: E402
import tensor_bilin18_tangent_authority as lifecycle  # noqa: E402


PROTOCOL = HERE / "MLP2_IMPLICIT_FOLDED_TENSOR_V1_EXECUTION_PROTOCOL.json"
PREREG = HERE / "MLP2_IMPLICIT_FOLDED_TENSOR_PREREGISTRATION.md"
COMMON_CONTRACT = HERE / "COMMON_EARLY_MLP_DECOMPOSITION_COMPARISON_CONTRACT.md"
MLP1_DECISION = HERE / "MLP1_IMPLICIT_FOLDED_TENSOR_DECISION_ADDENDUM.md"
MATH_SOURCE = HERE / "mlp2_implicit_folded_tensor.py"
MATH_TEST = HERE / "test_mlp2_implicit_folded_tensor.py"
COLLECTOR = HERE / "collect_mlp2_implicit_folded_tensor_v1.py"
TEST = HERE / "test_mlp2_implicit_folded_tensor_v1_lifecycle.py"
AUTHORITY = HERE / "mlp2_implicit_folded_tensor_v1_authority.json"
RESULT = HERE / "mlp2_implicit_folded_tensor_v1_results.json"
OUTCOME_AUTHORITY = HERE / "mlp2_implicit_folded_tensor_v1_outcome_authority.json"
FAILURE = HERE / "mlp2_implicit_folded_tensor_v1_failure.json"
RUN_LOCK = HERE / ".mlp2_implicit_folded_tensor_v1.lock"

MLP1_V1_FAILURE = mlp1_freeze.PARENT_FAILURE
MLP1_V2_AUTHORITY = mlp1_freeze.AUTHORITY
MLP1_V2_RESULT = mlp1_freeze.RESULT
MLP1_V2_OUTCOME = mlp1_freeze.OUTCOME_AUTHORITY
MLP1_V2_FAILURE = mlp1_freeze.FAILURE
EXPECTED_LESSON_SHA256 = {
    MLP1_V1_FAILURE: "350d7dc7bb4ec3207853abaa5db83da57b58b1613898553320824245c3f48526",
    MLP1_V2_AUTHORITY: "baa1eb9bb245c792f9a8ac473d5ca25f0012b55f9aef5b5a8e86f6c5f693ce92",
    MLP1_V2_RESULT: "2cbd5a745a6669d49da22ded23bd6df385f3cc057b686f814ad0157ef5fa8281",
    MLP1_V2_OUTCOME: "b96573ee373e6d4f0032667acb5c838d5205fc9c166eb9e5112d90396c56a9de",
    COMMON_CONTRACT: "db10c1f301b963f687863de2174f0077421005d2840b0ce3d92dac326922b656",
    MLP1_DECISION: "e8f6695e35e8de5746b3ce60b3f0164d8435aa610e117b32174e7c785e6f76c7",
}


EXPECTED_PROTOCOL = {
    "schema_version": 1,
    "experiment_id": "bilin18_mlp2_implicit_folded_tensor_v1",
    "status": "prospective_no_mlp2_checkpoint_tensor_loaded",
    "site": 2,
    "device": "cpu",
    "analysis_arithmetic": "numpy.float64",
    "blas_thread_count": 1,
    "frozen_mlp2_sources": {
        "preregistration_commit": "899f81d6e120ecedb4d7fbd3b0f819740fb5a034",
        "implementation_commit": "58b4040ddcab72d96965f792f0c2da787d100864",
        "preregistration_sha256": (
            "d97f8c29307c8199bd8846bca3a16356ccd1767e356f067f8f23b465ac081d12"
        ),
        "math_source_sha256": (
            "e6093e9ae6f66078cb49f5a479e5131be1d7e0b880c290e8a44ebe17d7af063e"
        ),
        "math_test_sha256": (
            "3bbb750170b46ebb8f16e115a49eaffe09cb4b5850e6f8968099bb9443b3de3e"
        ),
    },
    "energy_levels": [0.9, 0.95, 0.99, 0.999],
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
    "mlp2_factor_dtypes": {
        "Left.weight": "torch.float32", "Right.weight": "torch.float32",
        "Down.weight": "torch.float32", "Down_bias": "torch.bfloat16",
    },
    "bias_rule": (
        "Clone and hash the original MLP2 bf16 Down_bias before conversion; make and "
        "separately hash one disjoint float64 analysis copy; retain all 1152 values "
        "exactly; bias is not an input to balancing, mode Grams, spectra, HOSVD tails, "
        "or price-rank selection."
    ),
    "checkpoint_load_rule": (
        "Only after exact MLP2 source/weight authority: one local "
        "torch.load(weights_only=True,mmap=True,map_location=cpu), exact 218-key "
        "shape/native-dtype validation, clone only MLP2 Left/Right/Down/Down_bias, then "
        "delete the state tree."
    ),
    "publication_order": [
        "mlp2_source_weight_authority", "mlp2_nonauthoritative_result",
        "mlp2_last_written_outcome_authority",
    ],
    "failure_rule": (
        "Failure is create-only, may bind a partial MLP2 result, and can never "
        "overwrite or authorize an artifact."
    ),
    "claim_boundary": (
        "spectral_and_fixed_grammar_price_diagnostic_only; no CP-rank, CE, causal, "
        "semantic, removal, OOD, or cube credit"
    ),
}


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


SOURCES = _deduplicate((
    Path(__file__), COLLECTOR, TEST, PROTOCOL, PREREG, COMMON_CONTRACT,
    MLP1_DECISION, MATH_SOURCE, MATH_TEST, *mlp1_freeze.SOURCES,
))


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return lifecycle.canonical_sha256(value)


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text())
    if value != EXPECTED_PROTOCOL:
        raise RuntimeError("MLP2 folded-tensor execution protocol changed")
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


def configure_cpu_runtime() -> dict[str, Any]:
    protocol = load_protocol()
    threads = protocol["blas_thread_count"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    blas = [
        {
            key: row.get(key)
            for key in (
                "user_api", "internal_api", "num_threads", "prefix", "version",
                "threading_layer", "architecture",
            )
        }
        for row in threadpool_info()
        if row.get("user_api") == "blas"
    ]
    if not blas or any(row["num_threads"] != threads for row in blas):
        raise RuntimeError("MLP2 NumPy BLAS thread count differs from frozen runtime")
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "threadpoolctl": package_version("threadpoolctl"),
        "device": "cpu",
        "analysis_dtype": "numpy.float64",
        "torch_num_threads": torch.get_num_threads(),
        "blas_environment": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "blas_libraries": blas,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def source_snapshot() -> dict[str, Any]:
    lifecycle.require_committed_sources(SOURCES)
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in SOURCES}
    if len(hashes) != len(SOURCES):
        raise RuntimeError("MLP2 folded-tensor source closure contains duplicates")
    result = {"source_hashes": hashes, "git": lifecycle.git_identity(SOURCES)}
    result["fingerprint"] = canonical_sha256(result)
    return result


def validate_frozen_mlp2_sources() -> dict[str, Any]:
    registered = load_protocol()["frozen_mlp2_sources"]
    observed = {
        "preregistration_sha256": file_sha256(PREREG),
        "math_source_sha256": file_sha256(MATH_SOURCE),
        "math_test_sha256": file_sha256(MATH_TEST),
    }
    if observed != {
        key: registered[key]
        for key in ("preregistration_sha256", "math_source_sha256", "math_test_sha256")
    }:
        raise RuntimeError("frozen MLP2 preregistration/math/test bytes changed")
    for field in ("preregistration_commit", "implementation_commit"):
        commit = registered[field]
        if subprocess.run(
            ("git", "merge-base", "--is-ancestor", commit, "origin/main"),
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).returncode != 0:
            raise RuntimeError(f"frozen MLP2 {field} is not reachable from origin/main")
    return dict(registered)


def validate_mlp1_lessons() -> dict[str, Any]:
    observed = {path: file_sha256(path) for path in EXPECTED_LESSON_SHA256}
    if observed != EXPECTED_LESSON_SHA256:
        raise RuntimeError("MLP1 dtype/decision lesson inputs changed")
    if MLP1_V2_FAILURE.exists():
        raise RuntimeError("successful MLP1 v2 diagnostic unexpectedly has a failure")
    parent = mlp1_freeze.validate_parent_failure()
    authority = json.loads(MLP1_V2_AUTHORITY.read_text())
    result = json.loads(MLP1_V2_RESULT.read_text())
    final = json.loads(MLP1_V2_OUTCOME.read_text())
    if (
        authority.get("status") != "frozen_before_any_v2_checkpoint_tensor_deserialization"
        or result.get("checkpoint_factor_receipt", {}).get("state_tree") != {
            "keys": 218, "torch_bfloat16_keys": 55, "torch_float32_keys": 163,
        }
        or result.get("checkpoint_factor_receipt", {}).get("native_bias", {}).get(
            "native_dtype"
        ) != "torch.bfloat16"
        or final.get("status") != "authoritative_weight_diagnostic_only"
        or final.get("result_sha256") != EXPECTED_LESSON_SHA256[MLP1_V2_RESULT]
        or final.get("v2_source_weight_authority_sha256") != EXPECTED_LESSON_SHA256[
            MLP1_V2_AUTHORITY
        ]
        or final.get("parent_v1_failure_sha256") != parent["failure_sha256"]
        or final.get("failure_absent") is not True
    ):
        raise RuntimeError("MLP1 dtype/decision lesson semantics changed")
    return {
        "hashes": {str(path.relative_to(ROOT)): digest for path, digest in observed.items()},
        "v1_failure_preserved": parent,
        "v2_dtype_census": result["checkpoint_factor_receipt"]["state_tree"],
        "v2_original_bias_dtype": "torch.bfloat16",
        "v2_outcome_authority_status": final["status"],
    }


def protected_snapshot() -> dict[str, Any]:
    result = {
        "sources": source_snapshot(),
        "checkpoint": mlp1_freeze.v1_freeze.checkpoint_snapshot(),
        "frozen_mlp2_sources": validate_frozen_mlp2_sources(),
        "mlp1_lessons": validate_mlp1_lessons(),
        "protocol_sha256": file_sha256(PROTOCOL),
        "preregistration_sha256": file_sha256(PREREG),
        "common_contract_sha256": file_sha256(COMMON_CONTRACT),
        "mlp1_decision_addendum_sha256": file_sha256(MLP1_DECISION),
        "math_source_sha256": file_sha256(MATH_SOURCE),
        "math_test_sha256": file_sha256(MATH_TEST),
    }
    result["fingerprint"] = canonical_sha256(result)
    return result


def require_empty_namespace() -> None:
    if any(path.exists() for path in namespace_outputs()):
        raise RuntimeError("MLP2 folded-tensor namespace is already frozen or spent")


def publication_guard(lock: lifecycle.RunLock, snapshot: Mapping[str, Any]) -> None:
    lock.assert_owned()
    require_empty_namespace()
    if protected_snapshot() != dict(snapshot):
        raise RuntimeError("MLP2 folded-tensor protected state changed before authority")


def build_authority(lock: lifecycle.RunLock, runtime: Mapping[str, Any]) -> dict[str, Any]:
    lock.assert_owned()
    require_empty_namespace()
    snapshot = protected_snapshot()
    payload = {
        "schema_version": 1,
        "receipt_kind": "mlp2_implicit_folded_tensor_v1_source_weight_authority",
        "status": "frozen_before_any_mlp2_checkpoint_tensor_deserialization",
        "protected_snapshot": snapshot,
        "runtime": dict(runtime), "protocol": load_protocol(),
        "namespace": namespace_contract(),
        "authorized_operation": "one CPU MLP2 implicit folded-tensor diagnostic",
        "rows_loaded": False, "checkpoint_deserialized": False,
        "mlp2_tensors_extracted": False, "mode_grams_computed": False,
        "spectra_computed": False, "result_computed": False,
    }
    lifecycle.publish_json_create_only(
        AUTHORITY, payload, ownership_check=lambda: publication_guard(lock, snapshot),
    )
    return payload


def validate_authority(
    value: Any, *, snapshot: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema_version", "receipt_kind", "status", "protected_snapshot", "runtime",
        "protocol", "namespace", "authorized_operation", "rows_loaded",
        "checkpoint_deserialized", "mlp2_tensors_extracted", "mode_grams_computed",
        "spectra_computed", "result_computed",
    }
    false_fields = {
        "rows_loaded", "checkpoint_deserialized", "mlp2_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "result_computed",
    }
    if (
        not isinstance(value, dict) or set(value) != expected_keys
        or value["schema_version"] != 1
        or value["receipt_kind"] != "mlp2_implicit_folded_tensor_v1_source_weight_authority"
        or value["status"] != "frozen_before_any_mlp2_checkpoint_tensor_deserialization"
        or value["protected_snapshot"] != dict(snapshot)
        or value["runtime"] != dict(runtime) or value["protocol"] != load_protocol()
        or value["namespace"] != namespace_contract()
        or value["authorized_operation"] != (
            "one CPU MLP2 implicit folded-tensor diagnostic"
        )
        or any(value[field] is not False for field in false_fields)
    ):
        raise RuntimeError("MLP2 folded-tensor authority is malformed")


def freeze() -> dict[str, Any]:
    runtime = configure_cpu_runtime()
    with lifecycle.exclusive_run_lock(RUN_LOCK) as lock:
        return build_authority(lock, runtime)


def main() -> None:
    payload = freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
