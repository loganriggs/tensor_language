from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

import collect_mlp1_implicit_folded_tensor_v1 as v1_collector
import collect_mlp1_implicit_folded_tensor_v2 as collector
import freeze_mlp1_implicit_folded_tensor_v2_authority as freeze


class FakeLock:
    def __init__(self) -> None:
        self.live = True

    def assert_owned(self) -> None:
        if not self.live:
            raise RuntimeError("lost v2 synthetic lock")


def _redirect(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(freeze, "AUTHORITY", tmp_path / "v2_authority.json")
    monkeypatch.setattr(freeze, "RESULT", tmp_path / "v2_result.json")
    monkeypatch.setattr(freeze, "OUTCOME_AUTHORITY", tmp_path / "v2_outcome_authority.json")
    monkeypatch.setattr(freeze, "FAILURE", tmp_path / "v2_failure.json")
    monkeypatch.setattr(freeze, "RUN_LOCK", tmp_path / ".v2.lock")
    monkeypatch.setattr(freeze, "namespace_contract", lambda: {
        "source_weight_authority": "v2_authority.json", "result": "v2_result.json",
        "outcome_authority": "v2_outcome_authority.json", "failure": "v2_failure.json",
        "run_lock": ".v2.lock",
    })


def _runtime() -> dict[str, object]:
    return {
        "python": "test", "torch": "test", "device": "cpu",
        "float_dtype": "torch.float64", "torch_num_threads": 8,
        "deterministic_algorithms": True,
    }


def _tiny_analysis_protocol() -> dict[str, object]:
    protocol = dict(freeze.v1_freeze.EXPECTED_PROTOCOL)
    protocol["hidden_block"] = 2
    protocol["down_price_ranks"] = [1, 2]
    protocol["cp_price_ranks"] = [1, 2]
    protocol["projected_core_plan"] = {"1": [1], "2": [1, 6]}
    return protocol


def _tiny_factors(bias: torch.Tensor) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    return {
        "down": torch.randn(3, 4, generator=generator),
        "left": torch.randn(4, 2, generator=generator),
        "right": torch.randn(4, 2, generator=generator),
        "bias": bias.double().clone(),
    }


def _factor_receipt() -> dict[str, object]:
    digest = "a" * 64
    return {
        "weights": {
            role: {
                "state_key": collector.MLP1_KEYS[role],
                "shape": list(collector.EXPECTED_FACTOR_SHAPES[role]),
                "native_dtype": "torch.float32", "native_raw_sha256": digest,
                "analysis_dtype": "torch.float32", "analysis_raw_sha256": digest,
            }
            for role in ("left", "right", "down")
        },
        "native_bias": {
            "state_key": collector.MLP1_KEYS["bias"],
            "shape": list(collector.EXPECTED_FACTOR_SHAPES["bias"]),
            "native_dtype": "torch.bfloat16", "native_raw_sha256": "b" * 64,
            "hashed_before_analysis_conversion": True,
        },
        "analysis_bias_copy": {
            "shape": list(collector.EXPECTED_FACTOR_SHAPES["bias"]),
            "dtype": "torch.float64", "raw_sha256": "c" * 64,
            "derived_from_native_raw_sha256": "b" * 64,
            "storage_disjoint_from_native": True,
            "conversion": "exact elementwise torch.bfloat16 to torch.float64",
        },
        "state_tree": {
            "keys": 218, "torch_bfloat16_keys": 55, "torch_float32_keys": 163,
        },
    }


def _diagnostic() -> dict[str, object]:
    return {
        "dimensions": {
            "output": collector.EXPECTED_FACTOR_SHAPES["down"][0],
            "hidden_products": collector.EXPECTED_FACTOR_SHAPES["down"][1],
            "input": collector.EXPECTED_FACTOR_SHAPES["left"][1],
        },
        "bias": {
            "preserved_separately": True,
            "shape": list(collector.EXPECTED_FACTOR_SHAPES["bias"]),
            "dtype": "torch.float64", "raw_sha256": "c" * 64, "l2_norm": 1.0,
        },
        "balancing": {}, "balanced_down": {},
        "folded_hosvd": {
            "materialized_folded_tensor": False,
            "input_modes_shared_by_partial_symmetry": True,
        },
        "projected_cores": {
            key: {} for key in freeze.v1_freeze.load_protocol()["projected_core_plan"]
        },
        "prices": {"cp_fitted": False},
    }


def test_v1_authority_failure_are_exact_and_v1_outcomes_remain_absent() -> None:
    parent = freeze.validate_parent_failure()
    assert parent["authority_sha256"] == freeze.PARENT_AUTHORITY_SHA256
    assert parent["failure_sha256"] == freeze.PARENT_FAILURE_SHA256
    assert parent["result_absent"] and parent["outcome_authority_absent"]


def test_v2_source_closure_is_unique_transitive_and_outputs_absent() -> None:
    assert len(freeze.SOURCES) == len({path.resolve() for path in freeze.SOURCES})
    assert set(freeze.v1_freeze.SOURCES) < set(freeze.SOURCES)
    names = {path.name for path in freeze.SOURCES}
    assert {
        "MLP1_IMPLICIT_FOLDED_TENSOR_V2_RETRY_PROTOCOL.json",
        "MLP1_IMPLICIT_FOLDED_TENSOR_V2_ERRATUM.md",
        "freeze_mlp1_implicit_folded_tensor_v2_authority.py",
        "collect_mlp1_implicit_folded_tensor_v2.py",
        "test_mlp1_implicit_folded_tensor_v2_lifecycle.py",
    } <= names
    assert not any(path.exists() for path in freeze.namespace_outputs())


def test_v2_freezer_contains_no_torch_load_call() -> None:
    tree = ast.parse(Path(freeze.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert calls == []


def test_exact_meta_shape_tree_derives_registered_218_key_mixed_dtype_census() -> None:
    config = freeze.v1_freeze.MODEL_SNAPSHOT / "config.json"
    schema = collector.expected_state_schema(config)
    assert len(schema) == 218
    assert sum(dtype == torch.bfloat16 for _, dtype in schema.values()) == 55
    assert sum(dtype == torch.float32 for _, dtype in schema.values()) == 163
    assert schema["transformer.wte.weight"][1] == torch.bfloat16
    assert schema["transformer.h.1.lambdas"][1] == torch.bfloat16
    assert schema["transformer.h.1.attn.lamb"][1] == torch.bfloat16
    assert schema["transformer.h.1.mlp.Down_bias"][1] == torch.bfloat16
    assert schema["transformer.h.1.mlp.Left.weight"][1] == torch.float32
    assert schema["transformer.h.1.mlp.Right.weight"][1] == torch.float32
    assert schema["transformer.h.1.mlp.Down.weight"][1] == torch.float32


def _mixed_schema_and_state():
    schema = {
        "transformer.wte.weight": ((3, 2), torch.bfloat16),
        "transformer.h.0.lambdas": ((2,), torch.bfloat16),
        "transformer.h.0.attn.lamb": ((), torch.bfloat16),
        "transformer.h.0.mlp.Down_bias": ((2,), torch.bfloat16),
        "transformer.h.0.mlp.Left.weight": ((4, 2), torch.float32),
        "lm_head.weight": ((3, 2), torch.float32),
    }
    state = {name: torch.zeros(shape, dtype=dtype) for name, (shape, dtype) in schema.items()}
    return schema, state


def test_mixed_dtype_state_schema_is_accepted() -> None:
    schema, state = _mixed_schema_and_state()
    collector.validate_state_tree(state, schema)


@pytest.mark.parametrize(
    "name,wrong",
    [
        ("transformer.wte.weight", torch.float32),
        ("transformer.h.0.lambdas", torch.float32),
        ("transformer.h.0.attn.lamb", torch.float32),
        ("transformer.h.0.mlp.Down_bias", torch.float32),
        ("transformer.h.0.mlp.Left.weight", torch.bfloat16),
        ("lm_head.weight", torch.bfloat16),
    ],
)
def test_wrong_native_dtype_is_rejected(name: str, wrong: torch.dtype) -> None:
    schema, state = _mixed_schema_and_state()
    shape = schema[name][0]
    state[name] = torch.zeros(shape, dtype=wrong)
    with pytest.raises(RuntimeError, match="metadata changed"):
        collector.validate_state_tree(state, schema)


def test_bf16_physical_byte_hash_is_dtype_sensitive_and_supported() -> None:
    value = torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16)
    assert len(collector.tensor_sha256(value)) == 64
    assert collector.tensor_sha256(value) != collector.tensor_sha256(value.float())


def test_factor_loader_hashes_native_bf16_bias_before_disjoint_float64_copy(
    monkeypatch, tmp_path: Path,
) -> None:
    checkpoint_file = tmp_path / "weights.pt"
    checkpoint_file.write_bytes(b"synthetic checkpoint identity")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    state = {
        collector.MLP1_KEYS["left"]: torch.arange(8, dtype=torch.float32).view(4, 2),
        collector.MLP1_KEYS["right"]: torch.arange(8, dtype=torch.float32).view(4, 2) + 1,
        collector.MLP1_KEYS["down"]: torch.arange(12, dtype=torch.float32).view(3, 4),
        collector.MLP1_KEYS["bias"]: torch.tensor([1.25, -2.5, 3.75], dtype=torch.bfloat16),
    }
    schema = {name: (tuple(value.shape), value.dtype) for name, value in state.items()}
    expected_hash = "e" * 64
    checkpoint = {
        "weights": {
            "path": str(checkpoint_file), "sha256": expected_hash,
            "bytes": checkpoint_file.stat().st_size,
        },
        "config": {"path": str(config_file)},
    }
    monkeypatch.setattr(collector, "expected_state_schema", lambda path: schema)
    monkeypatch.setattr(collector, "file_sha256", lambda path: expected_hash)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    monkeypatch.setattr(collector, "EXPECTED_FACTOR_SHAPES", {
        "left": (4, 2), "right": (4, 2), "down": (3, 4), "bias": (3,),
    })
    native_hash = collector.tensor_sha256(state[collector.MLP1_KEYS["bias"]])
    analysis, receipt = collector.load_mlp1_factors(checkpoint)
    assert receipt["native_bias"] == {
        "state_key": collector.MLP1_KEYS["bias"], "shape": [3],
        "native_dtype": "torch.bfloat16", "native_raw_sha256": native_hash,
        "hashed_before_analysis_conversion": True,
    }
    assert analysis["bias"].dtype == torch.float64
    assert receipt["analysis_bias_copy"]["raw_sha256"] == collector.tensor_sha256(
        analysis["bias"]
    )
    assert receipt["analysis_bias_copy"]["derived_from_native_raw_sha256"] == native_hash
    state[collector.MLP1_KEYS["bias"]].add_(100)
    torch.testing.assert_close(
        analysis["bias"], torch.tensor([1.25, -2.5, 3.75], dtype=torch.float64),
    )


def test_bias_changes_only_bias_report_not_any_spectrum_core_or_price() -> None:
    first = v1_collector.analyze_factors(
        _tiny_factors(torch.tensor([1.0, 2.0, 3.0])), _tiny_analysis_protocol(),
    )
    second = v1_collector.analyze_factors(
        _tiny_factors(torch.tensor([-10.0, 20.0, 30.0])), _tiny_analysis_protocol(),
    )
    assert first["bias"] != second["bias"]
    assert {key: value for key, value in first.items() if key != "bias"} == {
        key: value for key, value in second.items() if key != "bias"
    }


def test_v2_authority_is_create_only_and_all_outcome_flags_false(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "v2", "checkpoint": {"synthetic": True}}
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(freeze, "load_protocol", lambda: dict(freeze.EXPECTED_PROTOCOL))
    monkeypatch.setattr(
        freeze.v1_freeze, "load_protocol", lambda: dict(freeze.v1_freeze.EXPECTED_PROTOCOL),
    )
    payload = freeze.build_authority(FakeLock(), _runtime())
    stored = json.loads(freeze.AUTHORITY.read_text())
    assert stored == payload
    for field in (
        "rows_loaded", "checkpoint_deserialized", "mlp1_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "projected_cores_computed",
        "result_computed",
    ):
        assert stored[field] is False
    with pytest.raises(RuntimeError, match="already frozen or spent"):
        freeze.build_authority(FakeLock(), _runtime())


def test_v2_outcome_refuses_before_v2_authority_without_deserialization(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded checkpoint"))
    with pytest.raises(RuntimeError, match="freeze v2 source/weight authority"):
        collector.run_outcome(FakeLock(), _runtime())


def test_v2_success_is_result_first_and_exact_last_written_authority(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "v2", "checkpoint": {"synthetic": True}}
    runtime = _runtime()
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(freeze, "load_protocol", lambda: dict(freeze.EXPECTED_PROTOCOL))
    monkeypatch.setattr(
        freeze.v1_freeze, "load_protocol", lambda: dict(freeze.v1_freeze.EXPECTED_PROTOCOL),
    )
    freeze.build_authority(FakeLock(), runtime)
    monkeypatch.setattr(
        collector, "load_mlp1_factors", lambda checkpoint: ({}, _factor_receipt()),
    )
    monkeypatch.setattr(collector.v1_collector, "analyze_factors", lambda raw, plan: _diagnostic())
    final = collector.run_outcome(FakeLock(), runtime)
    assert freeze.RESULT.is_file() and freeze.OUTCOME_AUTHORITY.is_file()
    assert not freeze.FAILURE.exists()
    assert json.loads(freeze.OUTCOME_AUTHORITY.read_text()) == final
    assert final["result_sha256"] == collector.file_sha256(freeze.RESULT)
    stored = json.loads(freeze.RESULT.read_text())
    assert stored["authority"] == "none_until_v2_outcome_authority_exists"
    assert stored["checkpoint_factor_receipt"]["native_bias"][
        "native_raw_sha256"
    ] == "b" * 64
    with pytest.raises(RuntimeError, match="already spent"):
        collector.run_outcome(FakeLock(), runtime)


def test_v2_failure_is_create_only_and_never_authorizes_partial_result(
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
    assert failure["partial_v2_result_sha256"] == collector.file_sha256(freeze.RESULT)
    assert not freeze.OUTCOME_AUTHORITY.exists()
    before = freeze.FAILURE.read_bytes()
    collector.publish_failure(
        FakeLock(), authority_hash=authority_hash, error=RuntimeError("different"),
    )
    assert freeze.FAILURE.read_bytes() == before


def test_v2_result_validator_rejects_bias_hash_or_diagnostic_envelope_corruption() -> None:
    protocol = dict(freeze.EXPECTED_PROTOCOL)
    snapshot = {"fingerprint": "v2"}
    runtime = _runtime()
    base = {
        "schema_version": 2, "experiment_id": protocol["experiment_id"],
        "status": "complete_pending_v2_last_written_outcome_authority",
        "authority": "none_until_v2_outcome_authority_exists",
        "source_weight_authority_sha256": "d" * 64,
        "protected_snapshot_fingerprint": "v2", "runtime": runtime,
        "checkpoint_factor_receipt": _factor_receipt(), "diagnostic": _diagnostic(),
        "runtime_seconds": 1.0, "raw_checkpoint_tensors_published": False,
        "materialized_folded_tensor": False, "rows_loaded": False,
        "model_forward_calls": 0, "claim_boundary": protocol["claim_boundary"],
    }
    collector.validate_result(
        base, authority_hash="d" * 64, snapshot=snapshot,
        runtime=runtime, protocol=protocol,
    )
    bad_hash = json.loads(json.dumps(base))
    bad_hash["checkpoint_factor_receipt"]["native_bias"]["native_raw_sha256"] = "0"
    with pytest.raises(RuntimeError, match="original-bias provenance"):
        collector.validate_result(
            bad_hash, authority_hash="d" * 64, snapshot=snapshot,
            runtime=runtime, protocol=protocol,
        )
    bad_diagnostic = json.loads(json.dumps(base))
    bad_diagnostic["diagnostic"].pop("balanced_down")
    with pytest.raises(RuntimeError, match="diagnostic schema"):
        collector.validate_result(
            bad_diagnostic, authority_hash="d" * 64, snapshot=snapshot,
            runtime=runtime, protocol=protocol,
        )
