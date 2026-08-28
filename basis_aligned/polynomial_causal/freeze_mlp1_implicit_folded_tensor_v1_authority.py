#!/usr/bin/env python3
"""Freeze exact source and local checkpoint bytes before the MLP1 weight diagnostic.

This module hashes the checkpoint but never deserializes it.  It creates only the
pre-outcome authority receipt in a dedicated append-only namespace.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import bilin18_observed_model_facade as model_facade  # noqa: E402
import tensor_bilin18_tangent_authority as lifecycle  # noqa: E402


PROTOCOL = HERE / "MLP1_IMPLICIT_FOLDED_TENSOR_V1_EXECUTION_PROTOCOL.json"
PREREG = HERE / "MLP1_IMPLICIT_FOLDED_TENSOR_V1_PREREGISTRATION.md"
COMMON_CONTRACT = HERE / "COMMON_EARLY_MLP_DECOMPOSITION_COMPARISON_CONTRACT.md"
MATH_SOURCE = HERE / "mlp1_implicit_folded_tensor_v1.py"
MATH_TEST = HERE / "test_mlp1_implicit_folded_tensor_v1.py"
COLLECTOR = HERE / "collect_mlp1_implicit_folded_tensor_v1.py"
TEST = HERE / "test_mlp1_implicit_folded_tensor_v1_lifecycle.py"
AUTHORITY = HERE / "mlp1_implicit_folded_tensor_v1_authority.json"
RESULT = HERE / "mlp1_implicit_folded_tensor_v1_results.json"
OUTCOME_AUTHORITY = HERE / "mlp1_implicit_folded_tensor_v1_outcome_authority.json"
FAILURE = HERE / "mlp1_implicit_folded_tensor_v1_failure.json"
RUN_LOCK = HERE / ".mlp1_implicit_folded_tensor_v1.lock"
MODEL_SNAPSHOT = model_facade.DEFAULT_SNAPSHOT


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
    MATH_SOURCE, MATH_TEST,
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_bilin18_observed_model_facade.py",
    ROOT / "jacclust" / "tt_model.py",
    HERE / "tensor_bilin18_tangent_authority.py",
    HERE / "test_tensor_bilin18_tangent_pilot.py",
))

EXPECTED_PROTOCOL = {
    "schema_version": 1,
    "experiment_id": "bilin18_mlp1_implicit_folded_tensor_v1",
    "status": "prospective_no_checkpoint_tensor_loaded",
    "site": 1,
    "object": "bias-separated partially symmetric folded tensor T[o,i,j]",
    "device": "cpu",
    "arithmetic": "torch.float64",
    "thread_count": 8,
    "hidden_block": 64,
    "negative_gram_eigenvalue_relative_tolerance": 1e-10,
    "mode_trace_relative_tolerance": 1e-10,
    "numerical_rank_rule": (
        "singular_value > eps_float64 * max(unfolding_rows, unfolding_columns) * "
        "largest_singular_value"
    ),
    "energy_thresholds": [0.9, 0.95, 0.99, 0.999],
    "down_price_ranks": [16, 32, 64, 128, 256, 512, 1024, 1152],
    "cp_price_ranks": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4608],
    "projected_core_plan": {
        "16": [64, 256, 1024, 2176],
        "32": [128, 512, 2048, 8192, 16896],
        "64": [256, 1024, 4096, 16384, 65536, 133120],
    },
    "core_rule": (
        "For each registered r use the top-r output Gram eigenvectors and the shared "
        "top-r input Gram eigenvectors; project G[r,r,r], symmetrize input modes, "
        "rank unique COO entries by folded Frobenius mass, and use deterministic "
        "(output,b,c) ties."
    ),
    "checkpoint_load_rule": (
        "Only after exact source/checkpoint authority: local "
        "torch.load(weights_only=True,mmap=True,map_location=cpu), exact full "
        "state-tree schema replay against a meta TT.GPT, clone only MLP1 "
        "Left/Right/Down/Down_bias, then delete the state tree."
    ),
    "publication_order": [
        "source_weight_authority", "nonauthoritative_result",
        "last_written_outcome_authority",
    ],
    "failure_rule": (
        "A failure is create-only and may bind a partial result; it can never delete, "
        "overwrite, or authorize a result."
    ),
    "claim_boundary": (
        "Weight-space coefficient-Frobenius spectrum and fixed-grammar prices only; "
        "no CE, causal, semantic, CP-rank, product-minimality, OOD, edit, removal, "
        "or composition credit."
    ),
}


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def canonical_sha256(value: Any) -> str:
    return lifecycle.canonical_sha256(value)


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


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL.read_text())
    if value != EXPECTED_PROTOCOL:
        raise RuntimeError("MLP1 folded-tensor execution protocol bytes/semantics changed")
    return value


def configure_cpu_runtime() -> dict[str, Any]:
    protocol = load_protocol()
    torch.set_num_threads(protocol["thread_count"])
    torch.use_deterministic_algorithms(True)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "float_dtype": "torch.float64",
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def source_snapshot() -> dict[str, Any]:
    lifecycle.require_committed_sources(SOURCES)
    hashes = {str(path.resolve().relative_to(ROOT)): file_sha256(path) for path in SOURCES}
    if len(hashes) != len(SOURCES):
        raise RuntimeError("MLP1 folded-tensor source closure contains duplicates")
    result = {"source_hashes": hashes, "git": lifecycle.git_identity(SOURCES)}
    result["fingerprint"] = canonical_sha256(result)
    return result


def checkpoint_snapshot(snapshot: Path = MODEL_SNAPSHOT) -> dict[str, Any]:
    receipt = model_facade.validate_snapshot(snapshot, verify_weights_sha256=True)
    root = Path(receipt.snapshot)
    config = root / "config.json"
    weights = root / "pytorch_model.bin"
    result = {
        "model_repo": model_facade.MODEL_REPO,
        "revision": receipt.revision,
        "snapshot": receipt.snapshot,
        "config": {
            "path": str(config.resolve()), "sha256": receipt.config_sha256,
            "bytes": config.stat().st_size,
        },
        "weights": {
            "path": str(weights.resolve()), "sha256": receipt.weights_sha256,
            "bytes": receipt.weights_bytes,
        },
        "tokenizer_vocab": receipt.tokenizer_vocab,
        "logit_vocab": receipt.logit_vocab,
    }
    result["fingerprint"] = canonical_sha256(result)
    return result


def protected_snapshot() -> dict[str, Any]:
    result = {
        "sources": source_snapshot(),
        "checkpoint": checkpoint_snapshot(),
        "protocol_sha256": file_sha256(PROTOCOL),
        "preregistration_sha256": file_sha256(PREREG),
        "common_contract_sha256": file_sha256(COMMON_CONTRACT),
    }
    result["fingerprint"] = canonical_sha256(result)
    return result


def require_empty_authority_namespace() -> None:
    if any(path.exists() for path in namespace_outputs()):
        raise RuntimeError("MLP1 folded-tensor namespace is already frozen or spent")


def authority_publication_guard(
    lock: lifecycle.RunLock, expected_snapshot: Mapping[str, Any],
) -> None:
    lock.assert_owned()
    require_empty_authority_namespace()
    if protected_snapshot() != dict(expected_snapshot):
        raise RuntimeError("MLP1 folded-tensor protected state changed before authority write")


def build_authority(
    lock: lifecycle.RunLock, runtime: Mapping[str, Any],
) -> dict[str, Any]:
    lock.assert_owned()
    require_empty_authority_namespace()
    before = protected_snapshot()
    protocol = load_protocol()
    payload = {
        "schema_version": 1,
        "receipt_kind": "mlp1_implicit_folded_tensor_v1_source_weight_authority",
        "status": "frozen_before_any_mlp1_checkpoint_tensor_deserialization",
        "protected_snapshot": before,
        "runtime": dict(runtime),
        "protocol": protocol,
        "namespace": namespace_contract(),
        "authorized_operation": "one CPU weight-only MLP1 folded-tensor diagnostic",
        "rows_loaded": False,
        "checkpoint_deserialized": False,
        "mlp1_tensors_extracted": False,
        "mode_grams_computed": False,
        "spectra_computed": False,
        "projected_cores_computed": False,
        "result_computed": False,
    }
    lifecycle.publish_json_create_only(
        AUTHORITY, payload,
        ownership_check=lambda: authority_publication_guard(lock, before),
    )
    return payload


def validate_authority(
    value: Any, *, snapshot: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    keys = {
        "schema_version", "receipt_kind", "status", "protected_snapshot", "runtime",
        "protocol", "namespace", "authorized_operation", "rows_loaded",
        "checkpoint_deserialized", "mlp1_tensors_extracted", "mode_grams_computed",
        "spectra_computed", "projected_cores_computed", "result_computed",
    }
    false_fields = {
        "rows_loaded", "checkpoint_deserialized", "mlp1_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "projected_cores_computed",
        "result_computed",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or value["schema_version"] != 1
        or value["receipt_kind"] != "mlp1_implicit_folded_tensor_v1_source_weight_authority"
        or value["status"] != "frozen_before_any_mlp1_checkpoint_tensor_deserialization"
        or value["protected_snapshot"] != dict(snapshot)
        or value["runtime"] != dict(runtime)
        or value["protocol"] != load_protocol()
        or value["namespace"] != namespace_contract()
        or value["authorized_operation"] != "one CPU weight-only MLP1 folded-tensor diagnostic"
        or any(value[field] is not False for field in false_fields)
    ):
        raise RuntimeError("MLP1 folded-tensor source/weight authority is malformed")


def freeze() -> dict[str, Any]:
    runtime = configure_cpu_runtime()
    with lifecycle.exclusive_run_lock(RUN_LOCK) as lock:
        return build_authority(lock, runtime)


def main() -> None:
    payload = freeze()
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote create-only authority {AUTHORITY}")


if __name__ == "__main__":
    main()
