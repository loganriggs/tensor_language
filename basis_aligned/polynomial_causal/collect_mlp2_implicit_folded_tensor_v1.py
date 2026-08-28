#!/usr/bin/env python3
"""Authority-gated CPU collector for the implicit MLP2 folded-tensor diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

# The production entry point fixes BLAS threads before importing NumPy.  The authority
# records and rechecks these values; tests may import NumPy earlier but never publish.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import collect_mlp1_implicit_folded_tensor_v2 as dtype_contract  # noqa: E402
import freeze_mlp2_implicit_folded_tensor_v1_authority as freeze  # noqa: E402
import mlp2_implicit_folded_tensor as algebra  # noqa: E402
import tensor_bilin18_tangent_authority as lifecycle  # noqa: E402


MLP2_KEYS = {
    "left": "transformer.h.2.mlp.Left.weight",
    "right": "transformer.h.2.mlp.Right.weight",
    "down": "transformer.h.2.mlp.Down.weight",
    "bias": "transformer.h.2.mlp.Down_bias",
}
EXPECTED_FACTOR_SHAPES = {
    "left": (4608, 1152), "right": (4608, 1152),
    "down": (1152, 4608), "bias": (1152,),
}
EXPECTED_STATE_KEYS = 218
EXPECTED_BF16_KEYS = 55
EXPECTED_FLOAT32_KEYS = 163


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def tensor_sha256(value: torch.Tensor) -> str:
    return dtype_contract.tensor_sha256(value)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes(order="C")).hexdigest()


def expected_state_schema(
    config_path: Path,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    result = dtype_contract.expected_state_schema(config_path)
    if len(result) != EXPECTED_STATE_KEYS:
        raise RuntimeError("MLP2 inherited exact state schema changed")
    return result


def validate_state_tree(
    state: Any, expected: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
) -> None:
    dtype_contract.validate_state_tree(state, expected)


def load_mlp2_factors(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(checkpoint["weights"]["path"])
    expected_hash = checkpoint["weights"]["sha256"]
    expected_bytes = checkpoint["weights"]["bytes"]
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("MLP2 checkpoint changed immediately before deserialization")
    schema = expected_state_schema(Path(checkpoint["config"]["path"]))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_state_tree(state, schema)

    analysis: dict[str, np.ndarray] = {}
    weights_receipt: dict[str, Any] = {}
    for role in ("left", "right", "down"):
        name = MLP2_KEYS[role]
        native = state[name].detach().cpu().contiguous().clone()
        if tuple(native.shape) != EXPECTED_FACTOR_SHAPES[role] or native.dtype != torch.float32:
            raise RuntimeError(f"MLP2 {role} tensor metadata changed")
        if not bool(torch.isfinite(native).all()):
            raise RuntimeError(f"MLP2 {role} contains nonfinite values")
        native_hash = tensor_sha256(native)
        converted = np.array(native.numpy(), dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(converted)):
            raise RuntimeError(f"MLP2 {role} float64 analysis copy contains nonfinite values")
        analysis[role] = converted
        weights_receipt[role] = {
            "state_key": name, "shape": list(native.shape),
            "native_dtype": str(native.dtype), "native_raw_sha256": native_hash,
            "analysis_shape": list(converted.shape), "analysis_dtype": str(converted.dtype),
            "analysis_raw_sha256": array_sha256(converted),
            "conversion": "exact elementwise torch.float32 to numpy.float64",
        }
        del native

    bias_name = MLP2_KEYS["bias"]
    native_bias = state[bias_name].detach().cpu().contiguous().clone()
    if (
        tuple(native_bias.shape) != EXPECTED_FACTOR_SHAPES["bias"]
        or native_bias.dtype != torch.bfloat16
        or not bool(torch.isfinite(native_bias.float()).all())
    ):
        raise RuntimeError("MLP2 native bias metadata/values changed")
    native_bias_hash = tensor_sha256(native_bias)
    analysis_bias = np.array(
        native_bias.to(dtype=torch.float64).numpy(), dtype=np.float64, order="C", copy=True,
    )
    if not np.array_equal(analysis_bias, native_bias.to(dtype=torch.float64).numpy()):
        raise RuntimeError("MLP2 bf16-to-float64 analysis bias conversion changed values")
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
            "raw_sha256": array_sha256(analysis_bias),
            "derived_from_native_raw_sha256": native_bias_hash,
            "storage_disjoint_from_native": True,
            "conversion": "exact elementwise torch.bfloat16 to numpy.float64",
        },
        "state_tree": {
            "keys": EXPECTED_STATE_KEYS, "torch_bfloat16_keys": EXPECTED_BF16_KEYS,
            "torch_float32_keys": EXPECTED_FLOAT32_KEYS,
        },
    }
    del native_bias, state
    gc.collect()
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("MLP2 checkpoint changed during factor extraction")
    return analysis, receipt


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_spectrum(
    value: Any, *, levels: list[float], exact_length: int | None = None,
    psd_gram_dimension: int | None = None,
) -> algebra.Spectrum:
    keys = {
        "singular_values", "eigenvalues", "total_energy", "numerical_rank",
        "energy_ranks",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("MLP2 spectrum schema changed")
    singular = value["singular_values"]
    eigen = value["eigenvalues"]
    if (
        not isinstance(singular, list) or not isinstance(eigen, list)
        or len(singular) != len(eigen)
        or (exact_length is not None and len(singular) != exact_length)
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0 for item in singular)
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0 for item in eigen)
        or isinstance(value["total_energy"], bool)
        or not isinstance(value["total_energy"], (int, float))
        or not math.isfinite(value["total_energy"]) or value["total_energy"] < 0
        or isinstance(value["numerical_rank"], bool)
        or not isinstance(value["numerical_rank"], int)
        or not 0 <= value["numerical_rank"] <= len(singular)
        or set(value["energy_ranks"]) != {"r90", "r95", "r99", "r99.9"}
        or not all(
            isinstance(rank, int) and not isinstance(rank, bool) and 0 <= rank <= len(singular)
            for rank in value["energy_ranks"].values()
        )
    ):
        raise RuntimeError("MLP2 spectrum values changed schema")
    singular_array = np.asarray(singular, dtype=np.float64)
    eigen_array = np.asarray(eigen, dtype=np.float64)
    if (
        not np.all(singular_array[:-1] >= singular_array[1:])
        or not np.all(eigen_array[:-1] >= eigen_array[1:])
        or not np.allclose(
            singular_array * singular_array, eigen_array, rtol=2e-12, atol=1e-14,
        )
        or not math.isclose(
            float(np.sum(eigen_array)), float(value["total_energy"]),
            rel_tol=2e-12, abs_tol=1e-12,
        )
    ):
        raise RuntimeError("MLP2 spectrum algebra changed")
    if value["total_energy"] == 0:
        expected_energy_ranks = {algebra._energy_key(level): 0 for level in levels}
    else:
        cumulative = np.cumsum(eigen_array) / float(np.sum(eigen_array))
        expected_energy_ranks = {
            algebra._energy_key(level): int(
                np.searchsorted(cumulative, level, side="left") + 1
            )
            for level in levels
        }
    if value["energy_ranks"] != expected_energy_ranks:
        raise RuntimeError("MLP2 spectrum energy ranks changed")
    if psd_gram_dimension is not None:
        if eigen_array.size == 0 or eigen_array[0] == 0:
            expected_numerical_rank = 0
        else:
            expected_numerical_rank = int(np.count_nonzero(
                eigen_array
                > eigen_array[0] * psd_gram_dimension * np.finfo(np.float64).eps
            ))
        if value["numerical_rank"] != expected_numerical_rank:
            raise RuntimeError("MLP2 mode numerical rank changed")
    return algebra.Spectrum(
        singular_values=singular_array, eigenvalues=eigen_array,
        total_energy=float(value["total_energy"]),
        numerical_rank=value["numerical_rank"], energy_ranks=dict(value["energy_ranks"]),
    )


def _validate_price(value: Any) -> None:
    keys = {
        "family", "stored_values", "bias_values", "bilinear_products_per_token",
        "linear_weight_multiplies_per_token", "metadata_included", "status",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or any(
            isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0
            for key in (
                "stored_values", "bias_values", "bilinear_products_per_token",
                "linear_weight_multiplies_per_token",
            )
        )
        or value["bias_values"] != 1152 or value["metadata_included"] is not False
        or not isinstance(value["family"], str) or not isinstance(value["status"], str)
    ):
        raise RuntimeError("MLP2 price schema changed")


def validate_diagnostic(value: Any, protocol: Mapping[str, Any]) -> None:
    keys = {
        "site", "input_dim", "output_dim", "declared_products", "active_products",
        "zero_products", "bias_preserved", "balanced_down", "folded_output",
        "folded_input", "native_price", "down_price_points", "hosvd_price_points",
        "claim_boundary",
    }
    if (
        not isinstance(value, dict) or set(value) != keys or value["site"] != 2
        or value["input_dim"] != 1152 or value["output_dim"] != 1152
        or value["declared_products"] != 4608
        or not isinstance(value["active_products"], int)
        or not isinstance(value["zero_products"], int)
        or value["active_products"] + value["zero_products"] != 4608
        or value["bias_preserved"] is not True
        or value["claim_boundary"] != protocol["claim_boundary"]
    ):
        raise RuntimeError("MLP2 diagnostic envelope changed")
    levels = protocol["energy_levels"]
    down_spectrum = _validate_spectrum(value["balanced_down"], levels=levels)
    if len(value["balanced_down"]["singular_values"]) > 1152:
        raise RuntimeError("MLP2 balanced-Down spectrum is too long")
    if down_spectrum.singular_values.size == 0:
        expected_down_numerical_rank = 0
    else:
        tolerance = (
            max(1152, value["active_products"])
            * np.finfo(np.float64).eps
            * down_spectrum.singular_values[0]
        )
        expected_down_numerical_rank = int(np.count_nonzero(
            down_spectrum.singular_values > tolerance
        ))
    if down_spectrum.numerical_rank != expected_down_numerical_rank:
        raise RuntimeError("MLP2 balanced-Down numerical rank changed")
    output_spectrum = _validate_spectrum(
        value["folded_output"], levels=levels, exact_length=1152,
        psd_gram_dimension=1152,
    )
    input_spectrum = _validate_spectrum(
        value["folded_input"], levels=levels, exact_length=1152,
        psd_gram_dimension=1152,
    )
    _validate_price(value["native_price"])
    expected_native = algebra.native_mlp_price(
        products=4608, input_dim=1152, output_dim=1152,
    ).as_dict()
    if value["native_price"] != expected_native:
        raise RuntimeError("MLP2 native price changed")
    down = value["down_price_points"]
    hosvd = value["hosvd_price_points"]
    if (
        not isinstance(down, list) or len(down) != len(levels)
        or [row.get("energy_level") for row in down] != levels
        or not isinstance(hosvd, list) or len(hosvd) != len(levels)
        or [row.get("energy_level") for row in hosvd] != levels
    ):
        raise RuntimeError("MLP2 registered price grid changed")
    for row in down:
        if not isinstance(row, dict) or set(row) != {"energy_level", "price"}:
            raise RuntimeError("MLP2 Down price point schema changed")
        _validate_price(row["price"])
        key = algebra._energy_key(row["energy_level"])
        expected_price = algebra.down_svd_price(
            products=value["active_products"], input_dim=1152, output_dim=1152,
            rank=down_spectrum.energy_ranks[key],
        ).as_dict()
        if row["price"] != expected_price:
            raise RuntimeError("MLP2 Down price derivation changed")
    hosvd_keys = {
        "energy_level", "output_rank", "input_rank",
        "relative_frobenius_error_upper_bound", "price",
        "fewer_products_than_native", "fewer_values_than_native",
    }
    for row in hosvd:
        if (
            not isinstance(row, dict) or set(row) != hosvd_keys
            or not isinstance(row["output_rank"], int)
            or not isinstance(row["input_rank"], int)
            or not 0 <= row["output_rank"] <= 1152
            or not 0 <= row["input_rank"] <= 1152
            or isinstance(row["relative_frobenius_error_upper_bound"], bool)
            or not isinstance(row["relative_frobenius_error_upper_bound"], (int, float))
            or not math.isfinite(row["relative_frobenius_error_upper_bound"])
            or not 0 <= row["relative_frobenius_error_upper_bound"] <= 1
            or type(row["fewer_products_than_native"]) is not bool
            or type(row["fewer_values_than_native"]) is not bool
        ):
            raise RuntimeError("MLP2 HOSVD price point schema changed")
        _validate_price(row["price"])
        key = algebra._energy_key(row["energy_level"])
        output_rank = output_spectrum.energy_ranks[key]
        input_rank = input_spectrum.energy_ranks[key]
        expected_price = algebra.symmetric_tucker_price(
            input_dim=1152, output_dim=1152,
            input_rank=input_rank, output_rank=output_rank,
        ).as_dict()
        expected_bound = algebra.hosvd_relative_error_upper_bound(
            output_spectrum, input_spectrum,
            output_rank=output_rank, input_rank=input_rank,
        )
        if (
            row["output_rank"] != output_rank or row["input_rank"] != input_rank
            or row["price"] != expected_price
            or not math.isclose(
                row["relative_frobenius_error_upper_bound"], expected_bound,
                rel_tol=2e-12, abs_tol=1e-14,
            )
            or row["fewer_products_than_native"] != (
                expected_price["bilinear_products_per_token"] < 4608
            )
            or row["fewer_values_than_native"] != (
                expected_price["stored_values"] < expected_native["stored_values"]
            )
        ):
            raise RuntimeError("MLP2 HOSVD derivation changed")


def validate_factor_receipt(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "weights", "native_bias", "analysis_bias_copy", "state_tree",
    }:
        raise RuntimeError("MLP2 factor receipt schema changed")
    weights = value["weights"]
    if not isinstance(weights, dict) or set(weights) != {"left", "right", "down"}:
        raise RuntimeError("MLP2 factor receipt roles changed")
    for role, row in weights.items():
        if (
            not isinstance(row, dict)
            or set(row) != {
                "state_key", "shape", "native_dtype", "native_raw_sha256",
                "analysis_shape", "analysis_dtype", "analysis_raw_sha256", "conversion",
            }
            or row["state_key"] != MLP2_KEYS[role]
            or row["shape"] != list(EXPECTED_FACTOR_SHAPES[role])
            or row["analysis_shape"] != list(EXPECTED_FACTOR_SHAPES[role])
            or row["native_dtype"] != "torch.float32"
            or row["analysis_dtype"] != "float64"
            or row["conversion"] != "exact elementwise torch.float32 to numpy.float64"
            or not _is_sha256(row["native_raw_sha256"])
            or not _is_sha256(row["analysis_raw_sha256"])
        ):
            raise RuntimeError(f"MLP2 {role} factor provenance changed")
    native = value["native_bias"]
    analysis = value["analysis_bias_copy"]
    if (
        not isinstance(native, dict)
        or set(native) != {
            "state_key", "shape", "native_dtype", "native_raw_sha256",
            "hashed_before_analysis_conversion",
        }
        or native["state_key"] != MLP2_KEYS["bias"]
        or native["shape"] != [1152] or native["native_dtype"] != "torch.bfloat16"
        or native["hashed_before_analysis_conversion"] is not True
        or not _is_sha256(native["native_raw_sha256"])
        or not isinstance(analysis, dict)
        or set(analysis) != {
            "shape", "dtype", "raw_sha256", "derived_from_native_raw_sha256",
            "storage_disjoint_from_native", "conversion",
        }
        or analysis["shape"] != [1152] or analysis["dtype"] != "float64"
        or not _is_sha256(analysis["raw_sha256"])
        or analysis["derived_from_native_raw_sha256"] != native["native_raw_sha256"]
        or analysis["storage_disjoint_from_native"] is not True
        or analysis["conversion"] != "exact elementwise torch.bfloat16 to numpy.float64"
        or value["state_tree"] != {
            "keys": 218, "torch_bfloat16_keys": 55, "torch_float32_keys": 163,
        }
    ):
        raise RuntimeError("MLP2 original-bias provenance changed")


def result_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
) -> None:
    lock.assert_owned()
    if freeze.RESULT.exists() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("MLP2 result namespace changed before publication")
    if (
        file_sha256(freeze.AUTHORITY) != authority_hash
        or freeze.protected_snapshot() != dict(snapshot)
    ):
        raise RuntimeError("MLP2 result protected provenance changed")


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
        not isinstance(value, dict) or set(value) != keys or value["schema_version"] != 1
        or value["experiment_id"] != protocol["experiment_id"]
        or value["status"] != "complete_pending_mlp2_last_written_outcome_authority"
        or value["authority"] != "none_until_mlp2_outcome_authority_exists"
        or value["source_weight_authority_sha256"] != authority_hash
        or value["protected_snapshot_fingerprint"] != snapshot["fingerprint"]
        or value["runtime"] != dict(runtime)
        or isinstance(value["runtime_seconds"], bool)
        or not isinstance(value["runtime_seconds"], (int, float))
        or not math.isfinite(value["runtime_seconds"]) or value["runtime_seconds"] < 0
        or value["raw_checkpoint_tensors_published"] is not False
        or value["materialized_folded_tensor"] is not False
        or value["rows_loaded"] is not False or value["model_forward_calls"] != 0
        or value["claim_boundary"] != protocol["claim_boundary"]
    ):
        raise RuntimeError("MLP2 result schema/provenance changed")
    validate_factor_receipt(value["checkpoint_factor_receipt"])
    validate_diagnostic(value["diagnostic"], protocol)


def final_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
    result_hash: str, expected_result: Mapping[str, Any], runtime: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    lock.assert_owned()
    if not freeze.RESULT.is_file() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("MLP2 final namespace changed")
    if (
        file_sha256(freeze.AUTHORITY) != authority_hash
        or file_sha256(freeze.RESULT) != result_hash
        or freeze.protected_snapshot() != dict(snapshot)
    ):
        raise RuntimeError("MLP2 final protected provenance changed")
    stored = json.loads(freeze.RESULT.read_text())
    if stored != dict(expected_result):
        raise RuntimeError("serialized MLP2 result differs from memory")
    validate_result(
        stored, authority_hash=authority_hash, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )


def validate_final(
    value: Any, *, authority_hash: str, result_hash: str, result_bytes: int,
    snapshot: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    namespace = freeze.namespace_contract()
    expected = {
        "schema_version": 1,
        "receipt_kind": "mlp2_implicit_folded_tensor_v1_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "source_weight_authority_path": namespace["source_weight_authority"],
        "source_weight_authority_sha256": authority_hash,
        "result_path": namespace["result"], "result_sha256": result_hash,
        "result_bytes": result_bytes,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "mlp1_v1_failure_sha256": freeze.EXPECTED_LESSON_SHA256[
            freeze.MLP1_V1_FAILURE
        ],
        "failure_absent": True, "claim_boundary": protocol["claim_boundary"],
    }
    if value != expected:
        raise RuntimeError("MLP2 final authority payload changed")


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
        "schema_version": 1, "status": "mlp2_failed_nonauthoritative",
        "source_weight_authority_sha256": authority_hash,
        "partial_result_sha256": partial, "result_authorized": False,
        "error_type": type(error).__name__, "error_message": str(error),
    }

    def guard() -> None:
        lock.assert_owned()
        observed = file_sha256(freeze.RESULT) if freeze.RESULT.is_file() else None
        if (
            freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists()
            or file_sha256(freeze.AUTHORITY) != authority_hash or observed != partial
        ):
            raise RuntimeError("MLP2 failure provenance changed")

    lifecycle.publish_json_create_only(freeze.FAILURE, payload, ownership_check=guard)


def run_outcome(lock: lifecycle.RunLock, runtime: Mapping[str, Any]) -> dict[str, Any]:
    lock.assert_owned()
    if not freeze.AUTHORITY.is_file():
        raise RuntimeError("freeze MLP2 source/weight authority before outcome")
    if any(path.exists() for path in (freeze.RESULT, freeze.OUTCOME_AUTHORITY, freeze.FAILURE)):
        raise RuntimeError("MLP2 outcome namespace is already spent")
    authority_hash = file_sha256(freeze.AUTHORITY)
    snapshot = freeze.protected_snapshot()
    authority = json.loads(freeze.AUTHORITY.read_text())
    freeze.validate_authority(authority, snapshot=snapshot, runtime=runtime)
    protocol = authority["protocol"]
    started = time.monotonic()
    factors, factor_receipt = load_mlp2_factors(snapshot["checkpoint"])
    diagnostic = algebra.analyze_bilin18_mlp2_factors(
        factors["down"], factors["left"], factors["right"], factors["bias"],
        energy_levels=protocol["energy_levels"],
    ).as_dict()
    del factors
    gc.collect()
    result = {
        "schema_version": 1, "experiment_id": protocol["experiment_id"],
        "status": "complete_pending_mlp2_last_written_outcome_authority",
        "authority": "none_until_mlp2_outcome_authority_exists",
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
        "schema_version": 1,
        "receipt_kind": "mlp2_implicit_folded_tensor_v1_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "source_weight_authority_path": namespace["source_weight_authority"],
        "source_weight_authority_sha256": authority_hash,
        "result_path": namespace["result"], "result_sha256": result_hash,
        "result_bytes": freeze.RESULT.stat().st_size,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "mlp1_v1_failure_sha256": freeze.EXPECTED_LESSON_SHA256[
            freeze.MLP1_V1_FAILURE
        ],
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
        parser.error("MLP2 scientific stage requires literal --run-outcome")
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
