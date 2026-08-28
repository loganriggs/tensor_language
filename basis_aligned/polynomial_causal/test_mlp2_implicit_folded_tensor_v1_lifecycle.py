from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import collect_mlp2_implicit_folded_tensor_v1 as collector
import freeze_mlp2_implicit_folded_tensor_v1_authority as freeze
import mlp2_implicit_folded_tensor as algebra


class FakeLock:
    def __init__(self) -> None:
        self.live = True

    def assert_owned(self) -> None:
        if not self.live:
            raise RuntimeError("lost synthetic MLP2 lock")


def _runtime() -> dict[str, object]:
    return {
        "python": "test", "torch": "test", "device": "cpu",
        "analysis_dtype": "numpy.float64", "torch_num_threads": 1,
        "blas_environment": {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "deterministic_algorithms": True,
    }


def _redirect(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(freeze, "AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(freeze, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(freeze, "OUTCOME_AUTHORITY", tmp_path / "outcome.json")
    monkeypatch.setattr(freeze, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(freeze, "RUN_LOCK", tmp_path / ".lock")
    monkeypatch.setattr(freeze, "namespace_contract", lambda: {
        "source_weight_authority": "authority.json", "result": "result.json",
        "outcome_authority": "outcome.json", "failure": "failure.json",
        "run_lock": ".lock",
    })


def _factor_receipt() -> dict[str, object]:
    return {
        "weights": {
            role: {
                "state_key": collector.MLP2_KEYS[role],
                "shape": list(collector.EXPECTED_FACTOR_SHAPES[role]),
                "native_dtype": "torch.float32", "native_raw_sha256": "a" * 64,
                "analysis_shape": list(collector.EXPECTED_FACTOR_SHAPES[role]),
                "analysis_dtype": "float64", "analysis_raw_sha256": "b" * 64,
                "conversion": "exact elementwise torch.float32 to numpy.float64",
            }
            for role in ("left", "right", "down")
        },
        "native_bias": {
            "state_key": collector.MLP2_KEYS["bias"], "shape": [1152],
            "native_dtype": "torch.bfloat16", "native_raw_sha256": "c" * 64,
            "hashed_before_analysis_conversion": True,
        },
        "analysis_bias_copy": {
            "shape": [1152], "dtype": "float64", "raw_sha256": "d" * 64,
            "derived_from_native_raw_sha256": "c" * 64,
            "storage_disjoint_from_native": True,
            "conversion": "exact elementwise torch.bfloat16 to numpy.float64",
        },
        "state_tree": {
            "keys": 218, "torch_bfloat16_keys": 55, "torch_float32_keys": 163,
        },
    }


def _spectrum(length: int) -> dict[str, object]:
    return {
        "singular_values": [0.0] * length, "eigenvalues": [0.0] * length,
        "total_energy": 0.0, "numerical_rank": 0,
        "energy_ranks": {"r90": 0, "r95": 0, "r99": 0, "r99.9": 0},
    }


def _diagnostic() -> dict[str, object]:
    levels = freeze.EXPECTED_PROTOCOL["energy_levels"]
    down_price = algebra.down_svd_price(
        products=4608, input_dim=1152, output_dim=1152, rank=0,
    ).as_dict()
    tucker_price = algebra.symmetric_tucker_price(
        input_dim=1152, output_dim=1152, input_rank=0, output_rank=0,
    ).as_dict()
    return {
        "site": 2, "input_dim": 1152, "output_dim": 1152,
        "declared_products": 4608, "active_products": 4608, "zero_products": 0,
        "bias_preserved": True, "balanced_down": _spectrum(1),
        "folded_output": _spectrum(1152), "folded_input": _spectrum(1152),
        "native_price": algebra.native_mlp_price(
            products=4608, input_dim=1152, output_dim=1152,
        ).as_dict(),
        "down_price_points": [
            {"energy_level": level, "price": down_price} for level in levels
        ],
        "hosvd_price_points": [
            {
                "energy_level": level, "output_rank": 0, "input_rank": 0,
                "relative_frobenius_error_upper_bound": 0.0,
                "price": tucker_price, "fewer_products_than_native": True,
                "fewer_values_than_native": True,
            }
            for level in levels
        ],
        "claim_boundary": freeze.EXPECTED_PROTOCOL["claim_boundary"],
    }


def test_source_closure_is_unique_transitive_and_namespace_is_empty() -> None:
    assert len(freeze.SOURCES) == len(set(freeze.SOURCES))
    assert set(freeze.mlp1_freeze.SOURCES) < set(freeze.SOURCES)
    assert {
        "MLP2_IMPLICIT_FOLDED_TENSOR_V1_EXECUTION_PROTOCOL.json",
        "MLP2_IMPLICIT_FOLDED_TENSOR_PREREGISTRATION.md",
        "mlp2_implicit_folded_tensor.py", "test_mlp2_implicit_folded_tensor.py",
        "freeze_mlp2_implicit_folded_tensor_v1_authority.py",
        "collect_mlp2_implicit_folded_tensor_v1.py",
        "test_mlp2_implicit_folded_tensor_v1_lifecycle.py",
        "COMMON_EARLY_MLP_DECOMPOSITION_COMPARISON_CONTRACT.md",
        "MLP1_IMPLICIT_FOLDED_TENSOR_DECISION_ADDENDUM.md",
    } <= {path.name for path in freeze.SOURCES}
    assert not any(path.exists() for path in freeze.namespace_outputs())


def test_freezer_has_no_checkpoint_deserialization_capability() -> None:
    tree = ast.parse(Path(freeze.__file__).read_text())
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
        and node.func.attr == "load"
        for node in ast.walk(tree)
    )


def test_mlp1_failure_and_successful_dtype_lessons_are_exactly_bound() -> None:
    frozen = freeze.validate_frozen_mlp2_sources()
    assert frozen["preregistration_commit"].startswith("899f81d6")
    assert frozen["implementation_commit"].startswith("58b4040d")
    lessons = freeze.validate_mlp1_lessons()
    assert lessons["v1_failure_preserved"]["result_absent"] is True
    assert lessons["v2_dtype_census"] == {
        "keys": 218, "torch_bfloat16_keys": 55, "torch_float32_keys": 163,
    }
    assert lessons["v2_original_bias_dtype"] == "torch.bfloat16"


def test_exact_mlp2_keys_use_mixed_contract() -> None:
    schema = collector.expected_state_schema(
        freeze.mlp1_freeze.v1_freeze.MODEL_SNAPSHOT / "config.json"
    )
    assert len(schema) == 218
    assert sum(dtype == torch.bfloat16 for _, dtype in schema.values()) == 55
    assert sum(dtype == torch.float32 for _, dtype in schema.values()) == 163
    assert schema[collector.MLP2_KEYS["left"]][1] == torch.float32
    assert schema[collector.MLP2_KEYS["right"]][1] == torch.float32
    assert schema[collector.MLP2_KEYS["down"]][1] == torch.float32
    assert schema[collector.MLP2_KEYS["bias"]][1] == torch.bfloat16


@pytest.mark.parametrize(
    "role,wrong",
    [
        ("left", torch.bfloat16), ("right", torch.bfloat16),
        ("down", torch.bfloat16), ("bias", torch.float32),
    ],
)
def test_wrong_mlp2_native_dtype_is_rejected(role: str, wrong: torch.dtype) -> None:
    shapes = {"left": (4, 2), "right": (4, 2), "down": (3, 4), "bias": (3,)}
    dtypes = {"left": torch.float32, "right": torch.float32, "down": torch.float32,
              "bias": torch.bfloat16}
    schema = {
        collector.MLP2_KEYS[name]: (shapes[name], dtype)
        for name, dtype in dtypes.items()
    }
    state = {
        collector.MLP2_KEYS[name]: torch.zeros(shapes[name], dtype=dtype)
        for name, dtype in dtypes.items()
    }
    state[collector.MLP2_KEYS[role]] = torch.zeros(shapes[role], dtype=wrong)
    with pytest.raises(RuntimeError, match="metadata changed"):
        collector.validate_state_tree(state, schema)


def test_loader_records_original_bf16_bias_before_disjoint_analysis_copy(
    monkeypatch, tmp_path: Path,
) -> None:
    checkpoint_file = tmp_path / "weights.pt"
    checkpoint_file.write_bytes(b"synthetic checkpoint")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    shapes = {"left": (4, 2), "right": (4, 2), "down": (3, 4), "bias": (3,)}
    state = {
        collector.MLP2_KEYS["left"]: torch.arange(8, dtype=torch.float32).view(4, 2),
        collector.MLP2_KEYS["right"]: torch.arange(8, dtype=torch.float32).view(4, 2) + 1,
        collector.MLP2_KEYS["down"]: torch.arange(12, dtype=torch.float32).view(3, 4),
        collector.MLP2_KEYS["bias"]: torch.tensor(
            [1.25, -2.5, 3.75], dtype=torch.bfloat16,
        ),
    }
    schema = {name: (tuple(value.shape), value.dtype) for name, value in state.items()}
    checkpoint = {
        "weights": {
            "path": str(checkpoint_file), "sha256": "e" * 64,
            "bytes": checkpoint_file.stat().st_size,
        },
        "config": {"path": str(config_file)},
    }
    monkeypatch.setattr(collector, "expected_state_schema", lambda path: schema)
    monkeypatch.setattr(collector, "file_sha256", lambda path: "e" * 64)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(collector, "EXPECTED_FACTOR_SHAPES", shapes)
    native_hash = collector.tensor_sha256(state[collector.MLP2_KEYS["bias"]])
    factors, receipt = collector.load_mlp2_factors(checkpoint)
    assert receipt["native_bias"]["native_raw_sha256"] == native_hash
    assert receipt["native_bias"]["hashed_before_analysis_conversion"] is True
    assert factors["bias"].dtype == np.float64
    assert receipt["analysis_bias_copy"]["raw_sha256"] == collector.array_sha256(
        factors["bias"]
    )
    state[collector.MLP2_KEYS["bias"]].add_(100)
    np.testing.assert_array_equal(factors["bias"], np.array([1.25, -2.5, 3.75]))


def test_bias_is_excluded_from_every_folded_spectrum_and_price() -> None:
    rng = np.random.default_rng(92)
    output = rng.normal(size=(3, 5))
    left = rng.normal(size=(5, 4))
    right = rng.normal(size=(5, 4))
    first = algebra.analyze_folded_factors(output, left, right, np.array([1.0, 2.0, 3.0]))
    second = algebra.analyze_folded_factors(
        output, left, right, np.array([-100.0, 20.0, 0.0]),
    )
    assert first.as_dict() == second.as_dict()


def test_authority_is_create_only_and_all_outcome_flags_are_false(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "mlp2"}
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    payload = freeze.build_authority(FakeLock(), _runtime())
    assert json.loads(freeze.AUTHORITY.read_text()) == payload
    for field in (
        "rows_loaded", "checkpoint_deserialized", "mlp2_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "result_computed",
    ):
        assert payload[field] is False
    with pytest.raises(RuntimeError, match="already frozen or spent"):
        freeze.build_authority(FakeLock(), _runtime())


def test_outcome_refuses_before_authority_without_checkpoint_load(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded"))
    with pytest.raises(RuntimeError, match="freeze MLP2"):
        collector.run_outcome(FakeLock(), _runtime())


def test_success_is_result_first_then_exact_last_written_authority(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "mlp2", "checkpoint": {"synthetic": True}}
    runtime = _runtime()
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    freeze.build_authority(FakeLock(), runtime)
    monkeypatch.setattr(
        collector, "load_mlp2_factors",
        lambda checkpoint: (
            {"down": None, "left": None, "right": None, "bias": None},
            _factor_receipt(),
        ),
    )

    class FakeDiagnostic:
        def as_dict(self):
            return _diagnostic()

    monkeypatch.setattr(
        collector.algebra, "analyze_bilin18_mlp2_factors",
        lambda *args, **kwargs: FakeDiagnostic(),
    )
    final = collector.run_outcome(FakeLock(), runtime)
    assert freeze.RESULT.is_file() and freeze.OUTCOME_AUTHORITY.is_file()
    assert not freeze.FAILURE.exists()
    assert json.loads(freeze.OUTCOME_AUTHORITY.read_text()) == final
    assert final["result_sha256"] == collector.file_sha256(freeze.RESULT)
    assert json.loads(freeze.RESULT.read_text())["authority"] == (
        "none_until_mlp2_outcome_authority_exists"
    )


def test_failure_is_create_only_and_cannot_authorize_partial_result(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    freeze.AUTHORITY.write_text("authority")
    authority_hash = collector.file_sha256(freeze.AUTHORITY)
    freeze.RESULT.write_text("partial")
    collector.publish_failure(
        FakeLock(), authority_hash=authority_hash, error=RuntimeError("synthetic"),
    )
    failure = json.loads(freeze.FAILURE.read_text())
    assert failure["result_authorized"] is False
    assert failure["partial_result_sha256"] == collector.file_sha256(freeze.RESULT)
    assert not freeze.OUTCOME_AUTHORITY.exists()
    before = freeze.FAILURE.read_bytes()
    collector.publish_failure(
        FakeLock(), authority_hash=authority_hash, error=RuntimeError("different"),
    )
    assert freeze.FAILURE.read_bytes() == before


def test_result_validator_rejects_native_bias_or_diagnostic_corruption() -> None:
    protocol = dict(freeze.EXPECTED_PROTOCOL)
    runtime = _runtime()
    base = {
        "schema_version": 1, "experiment_id": protocol["experiment_id"],
        "status": "complete_pending_mlp2_last_written_outcome_authority",
        "authority": "none_until_mlp2_outcome_authority_exists",
        "source_weight_authority_sha256": "e" * 64,
        "protected_snapshot_fingerprint": "mlp2", "runtime": runtime,
        "checkpoint_factor_receipt": _factor_receipt(), "diagnostic": _diagnostic(),
        "runtime_seconds": 1.0, "raw_checkpoint_tensors_published": False,
        "materialized_folded_tensor": False, "rows_loaded": False,
        "model_forward_calls": 0, "claim_boundary": protocol["claim_boundary"],
    }
    collector.validate_result(
        base, authority_hash="e" * 64, snapshot={"fingerprint": "mlp2"},
        runtime=runtime, protocol=protocol,
    )
    bad_bias = json.loads(json.dumps(base))
    bad_bias["checkpoint_factor_receipt"]["native_bias"]["native_dtype"] = "torch.float32"
    with pytest.raises(RuntimeError, match="original-bias provenance"):
        collector.validate_result(
            bad_bias, authority_hash="e" * 64, snapshot={"fingerprint": "mlp2"},
            runtime=runtime, protocol=protocol,
        )
    bad_diagnostic = json.loads(json.dumps(base))
    bad_diagnostic["diagnostic"]["site"] = 1
    with pytest.raises(RuntimeError, match="diagnostic envelope"):
        collector.validate_result(
            bad_diagnostic, authority_hash="e" * 64,
            snapshot={"fingerprint": "mlp2"}, runtime=runtime, protocol=protocol,
        )
    bad_rank = json.loads(json.dumps(base))
    bad_rank["diagnostic"]["folded_input"]["energy_ranks"]["r95"] = 1
    with pytest.raises(RuntimeError, match="energy ranks"):
        collector.validate_result(
            bad_rank, authority_hash="e" * 64,
            snapshot={"fingerprint": "mlp2"}, runtime=runtime, protocol=protocol,
        )
    bad_price = json.loads(json.dumps(base))
    bad_price["diagnostic"]["hosvd_price_points"][0]["price"]["stored_values"] += 1
    with pytest.raises(RuntimeError, match="HOSVD derivation"):
        collector.validate_result(
            bad_price, authority_hash="e" * 64,
            snapshot={"fingerprint": "mlp2"}, runtime=runtime, protocol=protocol,
        )
