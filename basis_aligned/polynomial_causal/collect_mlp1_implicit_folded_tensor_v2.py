#!/usr/bin/env python3
"""Authority-gated v2 collector with checkpoint-native mixed dtype validation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping

import torch

import collect_mlp1_implicit_folded_tensor_v1 as v1_collector
import freeze_mlp1_implicit_folded_tensor_v2_authority as freeze
import tensor_bilin18_tangent_authority as lifecycle


MLP1_KEYS = dict(v1_collector.MLP1_KEYS)
EXPECTED_FACTOR_SHAPES = dict(v1_collector.EXPECTED_FACTOR_SHAPES)
EXPECTED_STATE_KEYS = 218
EXPECTED_BF16_KEYS = 55
EXPECTED_FLOAT32_KEYS = 163
_BLOCK_BF16 = re.compile(
    r"^transformer\.h\.(?:[0-9]|1[0-7])\.(?:lambdas|attn\.lamb|mlp\.Down_bias)$"
)


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash physical bytes, including dtypes NumPy cannot represent such as bf16."""
    byte_view = value.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes(order="C")).hexdigest()


def native_dtype_for_key(name: str) -> torch.dtype:
    if name == "transformer.wte.weight" or _BLOCK_BF16.fullmatch(name):
        return torch.bfloat16
    return torch.float32


def expected_state_schema(config_path: Path) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    meta = v1_collector._expected_state_schema(config_path)
    result = {name: (shape, native_dtype_for_key(name)) for name, (shape, _) in meta.items()}
    bf16 = sum(dtype == torch.bfloat16 for _, dtype in result.values())
    fp32 = sum(dtype == torch.float32 for _, dtype in result.values())
    if len(result) != EXPECTED_STATE_KEYS or bf16 != EXPECTED_BF16_KEYS or (
        fp32 != EXPECTED_FLOAT32_KEYS
    ):
        raise RuntimeError("derived v2 checkpoint dtype census differs from the frozen contract")
    return result


def validate_state_tree(
    state: Any, expected: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
) -> None:
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RuntimeError("v2 checkpoint state-tree keys differ from exact TT.GPT schema")
    observed_counts = {torch.bfloat16: 0, torch.float32: 0}
    for name, value in state.items():
        shape, dtype = expected[name]
        if (
            not torch.is_tensor(value) or tuple(value.shape) != shape
            or value.dtype != dtype or value.device.type != "cpu"
        ):
            raise RuntimeError(f"v2 checkpoint state-tree metadata changed: {name}")
        observed_counts[dtype] += 1
    if len(expected) == EXPECTED_STATE_KEYS and observed_counts != {
        torch.bfloat16: EXPECTED_BF16_KEYS, torch.float32: EXPECTED_FLOAT32_KEYS,
    }:
        raise RuntimeError("v2 checkpoint dtype census changed")


def load_mlp1_factors(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(checkpoint["weights"]["path"])
    expected_hash = checkpoint["weights"]["sha256"]
    expected_bytes = checkpoint["weights"]["bytes"]
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("v2 checkpoint changed immediately before deserialization")
    expected = expected_state_schema(Path(checkpoint["config"]["path"]))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_state_tree(state, expected)

    analysis: dict[str, torch.Tensor] = {}
    weights_receipt: dict[str, Any] = {}
    for role in ("left", "right", "down"):
        name = MLP1_KEYS[role]
        native = state[name].detach().cpu().contiguous().clone()
        if tuple(native.shape) != EXPECTED_FACTOR_SHAPES[role] or native.dtype != torch.float32:
            raise RuntimeError(f"v2 MLP1 {role} tensor metadata changed")
        if not bool(torch.isfinite(native).all()):
            raise RuntimeError(f"v2 MLP1 {role} contains nonfinite values")
        analysis[role] = native
        raw_hash = tensor_sha256(native)
        weights_receipt[role] = {
            "state_key": name, "shape": list(native.shape),
            "native_dtype": str(native.dtype), "native_raw_sha256": raw_hash,
            "analysis_dtype": str(native.dtype), "analysis_raw_sha256": raw_hash,
        }

    bias_name = MLP1_KEYS["bias"]
    native_bias = state[bias_name].detach().cpu().contiguous().clone()
    if tuple(native_bias.shape) != EXPECTED_FACTOR_SHAPES["bias"] or (
        native_bias.dtype != torch.bfloat16
    ) or not bool(torch.isfinite(native_bias.float()).all()):
        raise RuntimeError("v2 MLP1 native bias metadata/values changed")
    native_bias_hash = tensor_sha256(native_bias)
    analysis_bias = native_bias.to(dtype=torch.float64).contiguous()
    if analysis_bias.untyped_storage().data_ptr() == native_bias.untyped_storage().data_ptr():
        raise RuntimeError("v2 MLP1 analysis bias aliases the native bf16 bias")
    if not torch.equal(analysis_bias, native_bias.to(dtype=torch.float64)):
        raise RuntimeError("v2 MLP1 bf16-to-float64 bias conversion changed values")
    analysis_bias_hash = tensor_sha256(analysis_bias)
    analysis["bias"] = analysis_bias
    receipt = {
        "weights": weights_receipt,
        "native_bias": {
            "state_key": bias_name, "shape": list(native_bias.shape),
            "native_dtype": str(native_bias.dtype),
            "native_raw_sha256": native_bias_hash,
            "hashed_before_analysis_conversion": True,
        },
        "analysis_bias_copy": {
            "shape": list(analysis_bias.shape), "dtype": str(analysis_bias.dtype),
            "raw_sha256": analysis_bias_hash,
            "derived_from_native_raw_sha256": native_bias_hash,
            "storage_disjoint_from_native": True,
            "conversion": "exact elementwise torch.bfloat16 to torch.float64",
        },
        "state_tree": {
            "keys": EXPECTED_STATE_KEYS, "torch_bfloat16_keys": EXPECTED_BF16_KEYS,
            "torch_float32_keys": EXPECTED_FLOAT32_KEYS,
        },
    }
    del native_bias, state
    gc.collect()
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("v2 checkpoint changed during MLP1 extraction")
    return analysis, receipt


def result_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
) -> None:
    lock.assert_owned()
    if freeze.RESULT.exists() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("v2 result namespace changed before publication")
    if file_sha256(freeze.AUTHORITY) != authority_hash or freeze.protected_snapshot() != dict(
        snapshot
    ):
        raise RuntimeError("v2 result protected provenance changed")


def validate_result(
    value: Any, *, authority_hash: str, snapshot: Mapping[str, Any],
    runtime: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    keys = {
        "schema_version", "experiment_id", "status", "authority",
        "source_weight_authority_sha256", "protected_snapshot_fingerprint", "runtime",
        "checkpoint_factor_receipt", "diagnostic", "runtime_seconds",
        "raw_checkpoint_tensors_published", "materialized_folded_tensor", "rows_loaded",
        "model_forward_calls", "claim_boundary",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or value["schema_version"] != 2
        or value["experiment_id"] != protocol["experiment_id"]
        or value["status"] != "complete_pending_v2_last_written_outcome_authority"
        or value["authority"] != "none_until_v2_outcome_authority_exists"
        or value["source_weight_authority_sha256"] != authority_hash
        or value["protected_snapshot_fingerprint"] != snapshot["fingerprint"]
        or value["runtime"] != dict(runtime)
        or value["raw_checkpoint_tensors_published"] is not False
        or value["materialized_folded_tensor"] is not False
        or value["rows_loaded"] is not False or value["model_forward_calls"] != 0
        or value["claim_boundary"] != protocol["claim_boundary"]
        or isinstance(value["runtime_seconds"], bool)
        or not isinstance(value["runtime_seconds"], (int, float))
        or not math.isfinite(value["runtime_seconds"])
        or value["runtime_seconds"] < 0
    ):
        raise RuntimeError("v2 result schema/provenance changed")
    receipt = value["checkpoint_factor_receipt"]
    if not isinstance(receipt, dict) or set(receipt) != {
        "weights", "native_bias", "analysis_bias_copy", "state_tree",
    }:
        raise RuntimeError("v2 factor receipt schema changed")
    native_bias = receipt["native_bias"]
    analysis_bias = receipt["analysis_bias_copy"]
    weights = receipt["weights"]
    if not isinstance(weights, dict) or set(weights) != {"left", "right", "down"}:
        raise RuntimeError("v2 weight receipt roles changed")
    for role, row in weights.items():
        if (
            not isinstance(row, dict)
            or set(row) != {
                "state_key", "shape", "native_dtype", "native_raw_sha256",
                "analysis_dtype", "analysis_raw_sha256",
            }
            or row["state_key"] != MLP1_KEYS[role]
            or row["shape"] != list(EXPECTED_FACTOR_SHAPES[role])
            or row["native_dtype"] != "torch.float32"
            or row["analysis_dtype"] != "torch.float32"
            or row["native_raw_sha256"] != row["analysis_raw_sha256"]
            or not isinstance(row["native_raw_sha256"], str)
            or len(row["native_raw_sha256"]) != 64
        ):
            raise RuntimeError(f"v2 {role} provenance changed")
    if (
        not isinstance(native_bias, dict)
        or set(native_bias) != {
            "state_key", "shape", "native_dtype", "native_raw_sha256",
            "hashed_before_analysis_conversion",
        }
        or native_bias["state_key"] != MLP1_KEYS["bias"]
        or native_bias["shape"] != list(EXPECTED_FACTOR_SHAPES["bias"])
        or native_bias["native_dtype"] != "torch.bfloat16"
        or native_bias["hashed_before_analysis_conversion"] is not True
        or not isinstance(native_bias["native_raw_sha256"], str)
        or len(native_bias["native_raw_sha256"]) != 64
        or not isinstance(analysis_bias, dict)
        or set(analysis_bias) != {
            "shape", "dtype", "raw_sha256", "derived_from_native_raw_sha256",
            "storage_disjoint_from_native", "conversion",
        }
        or analysis_bias["shape"] != list(EXPECTED_FACTOR_SHAPES["bias"])
        or analysis_bias["dtype"] != "torch.float64"
        or analysis_bias["storage_disjoint_from_native"] is not True
        or not isinstance(analysis_bias["raw_sha256"], str)
        or len(analysis_bias["raw_sha256"]) != 64
        or analysis_bias["derived_from_native_raw_sha256"] != native_bias[
            "native_raw_sha256"
        ]
        or analysis_bias["conversion"] != "exact elementwise torch.bfloat16 to torch.float64"
        or value["diagnostic"].get("bias", {}).get("raw_sha256") != analysis_bias.get(
            "raw_sha256"
        )
        or receipt["state_tree"] != {
            "keys": EXPECTED_STATE_KEYS, "torch_bfloat16_keys": EXPECTED_BF16_KEYS,
            "torch_float32_keys": EXPECTED_FLOAT32_KEYS,
        }
        or value["diagnostic"].get("dimensions") != {
            "output": EXPECTED_FACTOR_SHAPES["down"][0],
            "hidden_products": EXPECTED_FACTOR_SHAPES["down"][1],
            "input": EXPECTED_FACTOR_SHAPES["left"][1],
        }
    ):
        raise RuntimeError("v2 original-bias provenance changed")
    diagnostic = value["diagnostic"]
    if (
        not isinstance(diagnostic, dict)
        or set(diagnostic) != {
            "dimensions", "bias", "balancing", "balanced_down", "folded_hosvd",
            "projected_cores", "prices",
        }
        or not isinstance(diagnostic["bias"], dict)
        or diagnostic["bias"].get("preserved_separately") is not True
        or diagnostic["bias"].get("shape") != list(EXPECTED_FACTOR_SHAPES["bias"])
        or diagnostic["bias"].get("dtype") != "torch.float64"
        or diagnostic["folded_hosvd"].get("materialized_folded_tensor") is not False
        or diagnostic["folded_hosvd"].get(
            "input_modes_shared_by_partial_symmetry"
        ) is not True
        or diagnostic["prices"].get("cp_fitted") is not False
        or set(diagnostic["projected_cores"]) != set(
            freeze.v1_freeze.load_protocol()["projected_core_plan"]
        )
    ):
        raise RuntimeError("v2 inherited diagnostic schema changed")


def validate_final(
    value: Any, *, authority_hash: str, result_hash: str, result_bytes: int,
    snapshot: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    namespace = freeze.namespace_contract()
    expected = {
        "schema_version": 2,
        "receipt_kind": "mlp1_implicit_folded_tensor_v2_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "v2_source_weight_authority_path": namespace["source_weight_authority"],
        "v2_source_weight_authority_sha256": authority_hash,
        "parent_v1_authority_sha256": freeze.PARENT_AUTHORITY_SHA256,
        "parent_v1_failure_sha256": freeze.PARENT_FAILURE_SHA256,
        "result_path": namespace["result"], "result_sha256": result_hash,
        "result_bytes": result_bytes,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "failure_absent": True, "claim_boundary": protocol["claim_boundary"],
    }
    if value != expected:
        raise RuntimeError("v2 final authority payload changed")


def final_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
    result_hash: str, expected_result: Mapping[str, Any], runtime: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    lock.assert_owned()
    if not freeze.RESULT.is_file() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("v2 final namespace changed")
    if (
        file_sha256(freeze.AUTHORITY) != authority_hash
        or file_sha256(freeze.RESULT) != result_hash
        or freeze.protected_snapshot() != dict(snapshot)
    ):
        raise RuntimeError("v2 final protected provenance changed")
    stored = json.loads(freeze.RESULT.read_text())
    if stored != dict(expected_result):
        raise RuntimeError("v2 serialized result differs from in-memory result")
    validate_result(
        stored, authority_hash=authority_hash, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )


def publish_failure(
    lock: lifecycle.RunLock, *, authority_hash: str, error: BaseException,
) -> None:
    lock.assert_owned()
    if freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        return
    if not freeze.AUTHORITY.is_file() or file_sha256(freeze.AUTHORITY) != authority_hash:
        return
    partial = file_sha256(freeze.RESULT) if freeze.RESULT.is_file() else None
    payload = {
        "schema_version": 2, "status": "v2_failed_nonauthoritative",
        "v2_source_weight_authority_sha256": authority_hash,
        "parent_v1_failure_sha256": freeze.PARENT_FAILURE_SHA256,
        "partial_v2_result_sha256": partial, "result_authorized": False,
        "error_type": type(error).__name__, "error_message": str(error),
    }

    def guard() -> None:
        lock.assert_owned()
        observed = file_sha256(freeze.RESULT) if freeze.RESULT.is_file() else None
        if (
            freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists()
            or file_sha256(freeze.AUTHORITY) != authority_hash or observed != partial
        ):
            raise RuntimeError("v2 failure provenance changed")

    lifecycle.publish_json_create_only(freeze.FAILURE, payload, ownership_check=guard)


def run_outcome(lock: lifecycle.RunLock, runtime: Mapping[str, Any]) -> dict[str, Any]:
    lock.assert_owned()
    if not freeze.AUTHORITY.is_file():
        raise RuntimeError("freeze v2 source/weight authority before outcome")
    if any(path.exists() for path in (freeze.RESULT, freeze.OUTCOME_AUTHORITY, freeze.FAILURE)):
        raise RuntimeError("v2 outcome namespace is already spent")
    authority_hash = file_sha256(freeze.AUTHORITY)
    snapshot = freeze.protected_snapshot()
    authority = json.loads(freeze.AUTHORITY.read_text())
    freeze.validate_authority(authority, snapshot=snapshot, runtime=runtime)
    protocol = authority["protocol"]
    inherited = authority["inherited_v1_execution_plan"]
    started = time.monotonic()
    raw, factor_receipt = load_mlp1_factors(snapshot["checkpoint"])
    diagnostic = v1_collector.analyze_factors(raw, inherited)
    del raw
    gc.collect()
    result = {
        "schema_version": 2,
        "experiment_id": protocol["experiment_id"],
        "status": "complete_pending_v2_last_written_outcome_authority",
        "authority": "none_until_v2_outcome_authority_exists",
        "source_weight_authority_sha256": authority_hash,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "runtime": dict(runtime), "checkpoint_factor_receipt": factor_receipt,
        "diagnostic": diagnostic, "runtime_seconds": time.monotonic() - started,
        "raw_checkpoint_tensors_published": False,
        "materialized_folded_tensor": False, "rows_loaded": False,
        "model_forward_calls": 0, "claim_boundary": protocol["claim_boundary"],
    }
    validate_result(
        result, authority_hash=authority_hash, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )
    lifecycle.publish_json_create_only(
        freeze.RESULT, result,
        ownership_check=lambda: result_guard(
            lock, snapshot=snapshot, authority_hash=authority_hash,
        ),
    )
    result_hash = file_sha256(freeze.RESULT)
    namespace = freeze.namespace_contract()
    final = {
        "schema_version": 2,
        "receipt_kind": "mlp1_implicit_folded_tensor_v2_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "v2_source_weight_authority_path": namespace["source_weight_authority"],
        "v2_source_weight_authority_sha256": authority_hash,
        "parent_v1_authority_sha256": freeze.PARENT_AUTHORITY_SHA256,
        "parent_v1_failure_sha256": freeze.PARENT_FAILURE_SHA256,
        "result_path": namespace["result"], "result_sha256": result_hash,
        "result_bytes": freeze.RESULT.stat().st_size,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "failure_absent": True, "claim_boundary": protocol["claim_boundary"],
    }
    validate_final(
        final, authority_hash=authority_hash, result_hash=result_hash,
        result_bytes=freeze.RESULT.stat().st_size, snapshot=snapshot, protocol=protocol,
    )
    lifecycle.publish_json_create_only(
        freeze.OUTCOME_AUTHORITY, final,
        ownership_check=lambda: final_guard(
            lock, snapshot=snapshot, authority_hash=authority_hash,
            result_hash=result_hash, expected_result=result, runtime=runtime,
            protocol=protocol,
        ),
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-outcome", action="store_true")
    args = parser.parse_args()
    if not args.run_outcome:
        parser.error("v2 scientific stage requires literal --run-outcome")
    runtime = freeze.configure_cpu_runtime()
    with lifecycle.exclusive_run_lock(freeze.RUN_LOCK) as lock:
        authority_hash = file_sha256(freeze.AUTHORITY) if freeze.AUTHORITY.is_file() else ""
        try:
            final = run_outcome(lock, runtime)
        except BaseException as error:
            if authority_hash:
                publish_failure(lock, authority_hash=authority_hash, error=error)
            raise
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
