#!/usr/bin/env python3
"""Authority-gated CPU collector for the MLP1 implicit folded-tensor diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys
sys.path.insert(0, str(ROOT))
import bilin18_observed_model_facade as model_facade  # noqa: E402
import freeze_mlp1_implicit_folded_tensor_v1_authority as freeze  # noqa: E402
import mlp1_implicit_folded_tensor_v1 as algebra  # noqa: E402
import tensor_bilin18_tangent_authority as lifecycle  # noqa: E402


MLP1_KEYS = {
    "left": "transformer.h.1.mlp.Left.weight",
    "right": "transformer.h.1.mlp.Right.weight",
    "down": "transformer.h.1.mlp.Down.weight",
    "bias": "transformer.h.1.mlp.Down_bias",
}
EXPECTED_FACTOR_SHAPES = {
    "left": (4608, 1152), "right": (4608, 1152),
    "down": (1152, 4608), "bias": (1152,),
}


def file_sha256(path: Path) -> str:
    return lifecycle.sha256_file(path)


def tensor_sha256(value: torch.Tensor) -> str:
    return lifecycle.tensor_raw_sha256(value)


def _expected_state_schema(config_path: Path) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    config = model_facade.validate_config(json.loads(config_path.read_text()))
    constructor = dict(config)
    constructor.pop("step")
    with torch.device("meta"):
        model = model_facade.TT.GPT(model_facade.TT.GPTConfig(**constructor))
    return {
        name: (tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
    }


def validate_state_tree(
    state: Any, expected: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
) -> None:
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise RuntimeError("checkpoint state-tree keys differ from the pinned TT.GPT model")
    for name, value in state.items():
        if not torch.is_tensor(value):
            raise RuntimeError(f"checkpoint state-tree value is not a tensor: {name}")
        shape, dtype = expected[name]
        if tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu":
            raise RuntimeError(f"checkpoint state-tree metadata changed: {name}")


def load_mlp1_factors(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(checkpoint["weights"]["path"])
    expected_hash = checkpoint["weights"]["sha256"]
    expected_bytes = checkpoint["weights"]["bytes"]
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("checkpoint changed immediately before deserialization")
    config_path = Path(checkpoint["config"]["path"])
    expected_schema = _expected_state_schema(config_path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_state_tree(state, expected_schema)
    factors: dict[str, torch.Tensor] = {}
    factor_receipt: dict[str, Any] = {}
    for role, name in MLP1_KEYS.items():
        value = state[name]
        if tuple(value.shape) != EXPECTED_FACTOR_SHAPES[role] or value.dtype != torch.float32:
            raise RuntimeError(f"MLP1 {role} tensor metadata changed")
        owned = value.detach().cpu().contiguous().clone()
        if not bool(torch.isfinite(owned).all()):
            raise RuntimeError(f"MLP1 {role} tensor contains nonfinite values")
        factors[role] = owned
        factor_receipt[role] = {
            "state_key": name, "shape": list(owned.shape), "dtype": str(owned.dtype),
            "raw_sha256": tensor_sha256(owned),
        }
    del state
    gc.collect()
    if path.stat().st_size != expected_bytes or file_sha256(path) != expected_hash:
        raise RuntimeError("checkpoint changed during MLP1 tensor extraction")
    return factors, factor_receipt


def _spectrum_report(gram: torch.Tensor, protocol: Mapping[str, Any]) -> dict[str, object]:
    return algebra._spectrum_from_gram(  # source-closed pure helper
        gram,
        negative_relative_tolerance=protocol[
            "negative_gram_eigenvalue_relative_tolerance"
        ],
    )


def _with_numerical_rank(
    report: Mapping[str, Any], *, rows: int, columns: int,
) -> dict[str, Any]:
    result = dict(report)
    squared = torch.tensor(result["squared_singular_values"], dtype=torch.float64)
    largest = 0.0 if squared.numel() == 0 else float(torch.sqrt(squared[0]))
    tolerance = torch.finfo(torch.float64).eps * max(rows, columns) * largest
    rank = int((torch.sqrt(squared) > tolerance).sum())
    result["numerical_rank"] = rank
    result["numerical_rank_tolerance"] = tolerance
    result["unfolding_shape"] = [rows, columns]
    return result


def _down_replacement_only_price(output: int, hidden: int, rank: int) -> dict[str, int]:
    multiply_adds = rank * (hidden + output)
    return {
        "float_storage": multiply_adds + output,
        "integer_storage": 0,
        "multiply_adds_per_token": multiply_adds,
        "bilinear_products_per_token": hidden,
        "bias_additions_per_token": output,
        "scalar_multiplications_per_token": multiply_adds + hidden,
    }


def analyze_factors(
    raw: Mapping[str, torch.Tensor], protocol: Mapping[str, Any],
) -> dict[str, Any]:
    factors = algebra.balance_factors(raw["down"], raw["left"], raw["right"], raw["bias"])
    gout, gin = algebra.exact_folded_mode_grams(
        factors, hidden_block=protocol["hidden_block"],
    )
    output_spectrum = _with_numerical_rank(
        _spectrum_report(gout, protocol), rows=gout.shape[0],
        columns=factors.left.shape[1] ** 2,
    )
    input_spectrum = _with_numerical_rank(
        _spectrum_report(gin, protocol), rows=gin.shape[0],
        columns=factors.down.shape[0] * factors.left.shape[1],
    )
    trace_out = float(torch.trace(gout))
    trace_in = float(torch.trace(gin))
    trace_scale = max(1.0, abs(trace_out), abs(trace_in))
    trace_residual = abs(trace_out - trace_in) / trace_scale
    if trace_residual > protocol["mode_trace_relative_tolerance"]:
        raise RuntimeError("output/input mode Gram traces disagree")

    output_dim, hidden = factors.down.shape
    width = factors.left.shape[1]
    cores: dict[str, Any] = {}
    for rank_text, keep_counts in protocol["projected_core_plan"].items():
        rank = int(rank_text)
        uo, ui = algebra.hosvd_bases(
            gout, gin, output_rank=rank, input_rank=rank,
        )
        core = algebra.project_symmetric_hosvd_core(factors, uo, ui)
        symmetry = float((core - core.transpose(1, 2)).abs().max())
        if symmetry != 0.0:
            raise RuntimeError("projected core lost exact input symmetry")
        core_energy = float(core.square().sum())
        if core_energy > trace_out * (1.0 + 1e-10) + 1e-10:
            raise RuntimeError("projected core energy exceeds folded tensor energy")
        cores[rank_text] = {
            "shape": list(core.shape),
            "dtype": str(core.dtype),
            "raw_sha256": tensor_sha256(core),
            "frobenius_squared": core_energy,
            "retained_full_tensor_frobenius_fraction": (
                1.0 if trace_out == 0.0 else core_energy / trace_out
            ),
            "input_symmetry_max_abs": symmetry,
            "dense_price": algebra.dense_tucker_price(output_dim, width, rank, rank),
            "sparse_curve": algebra.sparse_core_curve(
                core, keep_counts, ambient_output=output_dim, ambient_width=width,
            ),
        }
        del core, uo, ui

    down_spectrum = _with_numerical_rank(
        algebra.balanced_down_svd(factors),
        rows=output_dim, columns=hidden,
    )
    result = {
        "dimensions": {"output": output_dim, "hidden_products": hidden, "input": width},
        "bias": {
            "preserved_separately": True, "shape": list(factors.bias.shape),
            "dtype": str(factors.bias.dtype), "raw_sha256": tensor_sha256(factors.bias),
            "l2_norm": float(torch.linalg.vector_norm(factors.bias)),
        },
        "balancing": {
            "rule": "equal positive factor norms per nonzero product term",
            "dead_units": list(factors.dead_units),
            "max_log_defect_before": factors.max_log_defect_before,
            "weighted_log_defect_before": factors.weighted_log_defect_before,
            "max_log_defect_after": factors.max_log_defect_after,
            "weighted_log_defect_after": factors.weighted_log_defect_after,
        },
        "balanced_down": down_spectrum,
        "folded_hosvd": {
            "output_mode": output_spectrum,
            "input_mode_1": input_spectrum,
            "input_mode_2": dict(input_spectrum),
            "input_modes_shared_by_partial_symmetry": True,
            "output_gram_trace": trace_out,
            "input_gram_trace": trace_in,
            "relative_trace_residual": trace_residual,
            "hidden_block": protocol["hidden_block"],
            "materialized_folded_tensor": False,
        },
        "projected_cores": cores,
        "prices": {
            "native": algebra.native_price(output_dim, hidden, width),
            "down_rank": {
                str(rank): {
                    "standalone": algebra.down_rank_price(output_dim, hidden, width, rank),
                    "replacement_only_inherited_left_right": (
                        _down_replacement_only_price(output_dim, hidden, rank)
                    ),
                }
                for rank in protocol["down_price_ranks"]
            },
            "cp_contract_only": {
                str(rank): algebra.cp_price(output_dim, width, rank)
                for rank in protocol["cp_price_ranks"]
            },
            "cp_fitted": False,
        },
    }
    del gout, gin, factors
    return result


def result_publication_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
) -> None:
    lock.assert_owned()
    if freeze.RESULT.exists() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("MLP1 folded-tensor result namespace changed before publication")
    if file_sha256(freeze.AUTHORITY) != authority_hash:
        raise RuntimeError("MLP1 folded-tensor source/weight authority changed")
    if freeze.protected_snapshot() != dict(snapshot):
        raise RuntimeError("MLP1 folded-tensor protected state changed before result")


def validate_result_payload(
    value: Any, *, authority_hash: str, snapshot: Mapping[str, Any],
    runtime: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    keys = {
        "schema_version", "experiment_id", "status", "authority",
        "source_weight_authority_sha256", "protected_snapshot_fingerprint",
        "runtime", "checkpoint_factor_receipt", "diagnostic", "runtime_seconds",
        "raw_checkpoint_tensors_published", "materialized_folded_tensor",
        "rows_loaded", "model_forward_calls", "claim_boundary",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or value["schema_version"] != 1
        or value["experiment_id"] != protocol["experiment_id"]
        or value["status"] != "complete_pending_last_written_outcome_authority"
        or value["authority"] != "none_until_outcome_authority_exists"
        or value["source_weight_authority_sha256"] != authority_hash
        or value["protected_snapshot_fingerprint"] != snapshot["fingerprint"]
        or value["runtime"] != dict(runtime)
        or value["raw_checkpoint_tensors_published"] is not False
        or value["materialized_folded_tensor"] is not False
        or value["rows_loaded"] is not False
        or value["model_forward_calls"] != 0
        or value["claim_boundary"] != protocol["claim_boundary"]
        or isinstance(value["runtime_seconds"], bool)
        or not isinstance(value["runtime_seconds"], (int, float))
        or not math.isfinite(value["runtime_seconds"])
        or value["runtime_seconds"] < 0
    ):
        raise RuntimeError("MLP1 folded-tensor result schema/provenance changed")
    receipts = value["checkpoint_factor_receipt"]
    if not isinstance(receipts, dict) or set(receipts) != set(MLP1_KEYS):
        raise RuntimeError("MLP1 folded-tensor factor receipt schema changed")
    for role, receipt in receipts.items():
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"state_key", "shape", "dtype", "raw_sha256"}
            or receipt["state_key"] != MLP1_KEYS[role]
            or receipt["shape"] != list(EXPECTED_FACTOR_SHAPES[role])
            or receipt["dtype"] != "torch.float32"
            or not isinstance(receipt["raw_sha256"], str)
            or len(receipt["raw_sha256"]) != 64
        ):
            raise RuntimeError(f"MLP1 folded-tensor {role} receipt changed")
    diagnostic = value["diagnostic"]
    expected_dimensions = {
        "output": EXPECTED_FACTOR_SHAPES["down"][0],
        "hidden_products": EXPECTED_FACTOR_SHAPES["down"][1],
        "input": EXPECTED_FACTOR_SHAPES["left"][1],
    }
    if (
        not isinstance(diagnostic, dict)
        or set(diagnostic) != {
            "dimensions", "bias", "balancing", "balanced_down", "folded_hosvd",
            "projected_cores", "prices",
        }
        or diagnostic["dimensions"] != expected_dimensions
        or diagnostic["bias"].get("preserved_separately") is not True
        or diagnostic["folded_hosvd"].get("materialized_folded_tensor") is not False
        or diagnostic["folded_hosvd"].get("input_modes_shared_by_partial_symmetry") is not True
        or diagnostic["prices"].get("cp_fitted") is not False
        or set(diagnostic["projected_cores"]) != set(protocol["projected_core_plan"])
    ):
        raise RuntimeError("MLP1 folded-tensor diagnostic schema changed")


def outcome_authority_publication_guard(
    lock: lifecycle.RunLock, *, snapshot: Mapping[str, Any], authority_hash: str,
    result_hash: str, expected_result: Mapping[str, Any], runtime: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    lock.assert_owned()
    if not freeze.RESULT.is_file() or freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
        raise RuntimeError("MLP1 folded-tensor final authority namespace changed")
    if (
        file_sha256(freeze.AUTHORITY) != authority_hash
        or file_sha256(freeze.RESULT) != result_hash
        or freeze.protected_snapshot() != dict(snapshot)
    ):
        raise RuntimeError("MLP1 folded-tensor final authority inputs changed")
    stored_result = json.loads(freeze.RESULT.read_text())
    if stored_result != dict(expected_result):
        raise RuntimeError("MLP1 folded-tensor serialized result differs from memory")
    validate_result_payload(
        stored_result, authority_hash=authority_hash, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )


def validate_outcome_authority_payload(
    value: Any, *, authority_hash: str, result_hash: str,
    result_bytes: int, snapshot: Mapping[str, Any], protocol: Mapping[str, Any],
) -> None:
    namespace = freeze.namespace_contract()
    expected = {
        "schema_version": 1,
        "receipt_kind": "mlp1_implicit_folded_tensor_v1_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "source_weight_authority_path": namespace["source_weight_authority"],
        "source_weight_authority_sha256": authority_hash,
        "result_path": namespace["result"],
        "result_sha256": result_hash,
        "result_bytes": result_bytes,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "failure_absent": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    if value != expected:
        raise RuntimeError("MLP1 folded-tensor final authority payload changed")


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
        "schema_version": 1,
        "status": "failed_nonauthoritative",
        "source_weight_authority_sha256": authority_hash,
        "partial_result_sha256": partial,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "result_authorized": False,
    }

    def guard() -> None:
        lock.assert_owned()
        if freeze.OUTCOME_AUTHORITY.exists() or freeze.FAILURE.exists():
            raise RuntimeError("MLP1 folded-tensor failure namespace changed")
        observed = file_sha256(freeze.RESULT) if freeze.RESULT.is_file() else None
        if file_sha256(freeze.AUTHORITY) != authority_hash or observed != partial:
            raise RuntimeError("MLP1 folded-tensor failure provenance changed")

    lifecycle.publish_json_create_only(freeze.FAILURE, payload, ownership_check=guard)


def run_outcome(lock: lifecycle.RunLock, runtime: Mapping[str, Any]) -> dict[str, Any]:
    lock.assert_owned()
    if not freeze.AUTHORITY.is_file():
        raise RuntimeError("freeze source/weight authority before the MLP1 outcome")
    if any(path.exists() for path in (freeze.RESULT, freeze.OUTCOME_AUTHORITY, freeze.FAILURE)):
        raise RuntimeError("MLP1 folded-tensor outcome namespace is already spent")
    authority_hash = file_sha256(freeze.AUTHORITY)
    snapshot = freeze.protected_snapshot()
    authority = json.loads(freeze.AUTHORITY.read_text())
    freeze.validate_authority(authority, snapshot=snapshot, runtime=runtime)
    protocol = authority["protocol"]
    started = time.monotonic()
    raw, factor_receipt = load_mlp1_factors(snapshot["checkpoint"])
    analysis = analyze_factors(raw, protocol)
    del raw
    gc.collect()
    result = {
        "schema_version": 1,
        "experiment_id": protocol["experiment_id"],
        "status": "complete_pending_last_written_outcome_authority",
        "authority": "none_until_outcome_authority_exists",
        "source_weight_authority_sha256": authority_hash,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "runtime": dict(runtime),
        "checkpoint_factor_receipt": factor_receipt,
        "diagnostic": analysis,
        "runtime_seconds": time.monotonic() - started,
        "raw_checkpoint_tensors_published": False,
        "materialized_folded_tensor": False,
        "rows_loaded": False,
        "model_forward_calls": 0,
        "claim_boundary": protocol["claim_boundary"],
    }
    validate_result_payload(
        result, authority_hash=authority_hash, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )
    lifecycle.publish_json_create_only(
        freeze.RESULT, result,
        ownership_check=lambda: result_publication_guard(
            lock, snapshot=snapshot, authority_hash=authority_hash,
        ),
    )
    result_hash = file_sha256(freeze.RESULT)
    namespace = freeze.namespace_contract()
    final = {
        "schema_version": 1,
        "receipt_kind": "mlp1_implicit_folded_tensor_v1_outcome_authority",
        "status": "authoritative_weight_diagnostic_only",
        "source_weight_authority_path": namespace["source_weight_authority"],
        "source_weight_authority_sha256": authority_hash,
        "result_path": namespace["result"],
        "result_sha256": result_hash,
        "result_bytes": freeze.RESULT.stat().st_size,
        "protected_snapshot_fingerprint": snapshot["fingerprint"],
        "failure_absent": True,
        "claim_boundary": protocol["claim_boundary"],
    }
    validate_outcome_authority_payload(
        final, authority_hash=authority_hash, result_hash=result_hash,
        result_bytes=freeze.RESULT.stat().st_size, snapshot=snapshot, protocol=protocol,
    )
    lifecycle.publish_json_create_only(
        freeze.OUTCOME_AUTHORITY, final,
        ownership_check=lambda: outcome_authority_publication_guard(
            lock, snapshot=snapshot, authority_hash=authority_hash,
            result_hash=result_hash, expected_result=result, runtime=runtime,
            protocol=protocol,
        ),
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-outcome", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run_outcome:
        parser.error("the scientific stage requires the literal --run-outcome flag")
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
