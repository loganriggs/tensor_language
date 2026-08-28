from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import collect_mlp1_implicit_folded_tensor_v1 as collector
import freeze_mlp1_implicit_folded_tensor_v1_authority as freeze


class FakeLock:
    def __init__(self) -> None:
        self.live = True

    def assert_owned(self) -> None:
        if not self.live:
            raise RuntimeError("lost synthetic lock")


def _redirect_namespace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(freeze, "AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(freeze, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(freeze, "OUTCOME_AUTHORITY", tmp_path / "outcome_authority.json")
    monkeypatch.setattr(freeze, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(freeze, "RUN_LOCK", tmp_path / ".run.lock")
    monkeypatch.setattr(freeze, "namespace_contract", lambda: {
        "source_weight_authority": "authority.json", "result": "result.json",
        "outcome_authority": "outcome_authority.json", "failure": "failure.json",
        "run_lock": ".run.lock",
    })


def _runtime() -> dict[str, object]:
    return {
        "python": "test", "torch": "test", "device": "cpu",
        "float_dtype": "torch.float64", "torch_num_threads": 8,
        "deterministic_algorithms": True,
    }


def _small_protocol() -> dict[str, object]:
    result = dict(freeze.EXPECTED_PROTOCOL)
    result["hidden_block"] = 2
    result["down_price_ranks"] = [1, 2]
    result["cp_price_ranks"] = [1, 2]
    result["projected_core_plan"] = {"1": [1], "2": [1, 6]}
    return result


def _small_factors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return {
        "down": torch.randn(3, 4, generator=generator),
        "left": torch.randn(4, 2, generator=generator),
        "right": torch.randn(4, 2, generator=generator),
        "bias": torch.randn(3, generator=generator),
    }


def test_source_closure_is_unique_and_contains_model_loader_math_and_tests() -> None:
    assert len(freeze.SOURCES) == len({path.resolve() for path in freeze.SOURCES})
    names = {path.name for path in freeze.SOURCES}
    assert {
        "freeze_mlp1_implicit_folded_tensor_v1_authority.py",
        "collect_mlp1_implicit_folded_tensor_v1.py",
        "test_mlp1_implicit_folded_tensor_v1_lifecycle.py",
        "MLP1_IMPLICIT_FOLDED_TENSOR_V1_EXECUTION_PROTOCOL.json",
        "MLP1_IMPLICIT_FOLDED_TENSOR_V1_PREREGISTRATION.md",
        "COMMON_EARLY_MLP_DECOMPOSITION_COMPARISON_CONTRACT.md",
        "mlp1_implicit_folded_tensor_v1.py",
        "test_mlp1_implicit_folded_tensor_v1.py",
        "bilin18_observed_model_facade.py",
        "test_bilin18_observed_model_facade.py",
        "tt_model.py",
        "tensor_bilin18_tangent_authority.py",
        "test_tensor_bilin18_tangent_pilot.py",
    } <= names


def test_freezer_contains_no_checkpoint_deserialization_call() -> None:
    tree = ast.parse(Path(freeze.__file__).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]
    assert calls == []


def test_checkpoint_snapshot_hashes_without_deserializing(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / model_facade_revision()
    snapshot.mkdir()
    config = snapshot / "config.json"
    weights = snapshot / "pytorch_model.bin"
    config.write_text("{}")
    weights.write_bytes(b"checkpoint bytes only")
    receipt = SimpleNamespace(
        snapshot=str(snapshot), revision=model_facade_revision(),
        config_sha256="c" * 64, weights_sha256="w" * 64,
        weights_bytes=weights.stat().st_size, tokenizer_vocab=50257, logit_vocab=50304,
    )
    monkeypatch.setattr(freeze.model_facade, "validate_snapshot", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("authority deserialized checkpoint"))
    observed = freeze.checkpoint_snapshot(snapshot)
    assert observed["weights"]["sha256"] == "w" * 64
    assert observed["weights"]["bytes"] == len(b"checkpoint bytes only")


def model_facade_revision() -> str:
    return freeze.model_facade.MODEL_REVISION


def test_authority_is_create_only_last_guarded_and_all_outcome_flags_false(
    monkeypatch, tmp_path: Path,
) -> None:
    _redirect_namespace(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "s" * 64}
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(freeze, "load_protocol", lambda: dict(freeze.EXPECTED_PROTOCOL))
    payload = freeze.build_authority(FakeLock(), _runtime())
    stored = json.loads(freeze.AUTHORITY.read_text())
    assert stored == payload
    assert stored["status"] == "frozen_before_any_mlp1_checkpoint_tensor_deserialization"
    for field in (
        "rows_loaded", "checkpoint_deserialized", "mlp1_tensors_extracted",
        "mode_grams_computed", "spectra_computed", "projected_cores_computed",
        "result_computed",
    ):
        assert stored[field] is False
    assert not any(path.exists() for path in (freeze.RESULT, freeze.OUTCOME_AUTHORITY, freeze.FAILURE))
    with pytest.raises(RuntimeError, match="already frozen or spent"):
        freeze.build_authority(FakeLock(), _runtime())


def test_authority_final_snapshot_drift_leaves_no_receipt(monkeypatch, tmp_path: Path) -> None:
    _redirect_namespace(monkeypatch, tmp_path)
    snapshots = iter(({"fingerprint": "before"}, {"fingerprint": "after"}))
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(freeze, "load_protocol", lambda: dict(freeze.EXPECTED_PROTOCOL))
    with pytest.raises(RuntimeError, match="protected state changed"):
        freeze.build_authority(FakeLock(), _runtime())
    assert not freeze.AUTHORITY.exists()


def test_validate_authority_rejects_self_authorizing_or_spent_flags(monkeypatch, tmp_path: Path) -> None:
    _redirect_namespace(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "s"}
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(freeze, "load_protocol", lambda: dict(freeze.EXPECTED_PROTOCOL))
    payload = freeze.build_authority(FakeLock(), _runtime())
    payload["result_computed"] = True
    with pytest.raises(RuntimeError, match="malformed"):
        freeze.validate_authority(payload, snapshot=snapshot, runtime=_runtime())


def test_outcome_refuses_before_authority_without_checkpoint_load(monkeypatch, tmp_path: Path) -> None:
    _redirect_namespace(monkeypatch, tmp_path)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("checkpoint loaded"))
    with pytest.raises(RuntimeError, match="freeze source/weight authority"):
        collector.run_outcome(FakeLock(), _runtime())


def test_state_tree_requires_exact_keys_shapes_dtypes_and_cpu() -> None:
    state = {"a": torch.zeros(2, 3), "b": torch.ones(4)}
    schema = {"a": ((2, 3), torch.float32), "b": ((4,), torch.float32)}
    collector.validate_state_tree(state, schema)
    with pytest.raises(RuntimeError, match="keys"):
        collector.validate_state_tree({"a": state["a"]}, schema)
    corrupt = dict(state)
    corrupt["b"] = corrupt["b"].double()
    with pytest.raises(RuntimeError, match="metadata"):
        collector.validate_state_tree(corrupt, schema)


def test_small_analysis_replays_implicit_spectra_core_and_exact_prices() -> None:
    result = collector.analyze_factors(_small_factors(), _small_protocol())
    assert result["dimensions"] == {"output": 3, "hidden_products": 4, "input": 2}
    assert result["bias"]["preserved_separately"] is True
    assert result["folded_hosvd"]["materialized_folded_tensor"] is False
    assert result["folded_hosvd"]["relative_trace_residual"] < 1e-12
    assert set(result["projected_cores"]) == {"1", "2"}
    assert result["prices"]["native"]["bilinear_products_per_token"] == 4
    assert result["prices"]["down_rank"]["1"]["standalone"]["bilinear_products_per_token"] == 4
    assert result["prices"]["down_rank"]["1"]["replacement_only_inherited_left_right"][
        "float_storage"
    ] == 10
    assert result["prices"]["cp_contract_only"]["1"]["bilinear_products_per_token"] == 1
    assert result["prices"]["cp_fitted"] is False
    assert result["balanced_down"]["numerical_rank"] > 0
    assert result["folded_hosvd"]["output_mode"]["unfolding_shape"] == [3, 4]


def _install_synthetic_authority(monkeypatch, tmp_path: Path):
    _redirect_namespace(monkeypatch, tmp_path)
    snapshot = {"fingerprint": "p" * 64, "checkpoint": {"weights": {}}}
    runtime = _runtime()
    authority = {"protocol": _small_protocol()}
    freeze.AUTHORITY.write_text(json.dumps(authority))
    authority_hash = collector.file_sha256(freeze.AUTHORITY)
    monkeypatch.setattr(freeze, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(freeze, "validate_authority", lambda *args, **kwargs: None)
    factors = _small_factors()
    synthetic_shapes = {role: tuple(value.shape) for role, value in factors.items()}
    monkeypatch.setattr(collector, "EXPECTED_FACTOR_SHAPES", synthetic_shapes)
    receipt = {
        role: {
            "state_key": collector.MLP1_KEYS[role], "shape": list(value.shape),
            "dtype": str(value.dtype), "raw_sha256": collector.tensor_sha256(value),
        }
        for role, value in factors.items()
    }
    monkeypatch.setattr(collector, "load_mlp1_factors", lambda checkpoint: (factors, receipt))
    return snapshot, runtime, authority_hash


def test_result_is_nonauthoritative_until_last_written_receipt(monkeypatch, tmp_path: Path) -> None:
    _, runtime, authority_hash = _install_synthetic_authority(monkeypatch, tmp_path)
    final = collector.run_outcome(FakeLock(), runtime)
    result = json.loads(freeze.RESULT.read_text())
    stored_final = json.loads(freeze.OUTCOME_AUTHORITY.read_text())
    assert result["authority"] == "none_until_outcome_authority_exists"
    assert result["rows_loaded"] is False and result["model_forward_calls"] == 0
    assert final == stored_final
    assert stored_final["result_sha256"] == collector.file_sha256(freeze.RESULT)
    assert stored_final["source_weight_authority_sha256"] == authority_hash
    assert stored_final["failure_absent"] is True
    corrupt = dict(result)
    corrupt["model_forward_calls"] = 1
    with pytest.raises(RuntimeError, match="schema/provenance"):
        collector.validate_result_payload(
            corrupt, authority_hash=authority_hash,
            snapshot=freeze.protected_snapshot(), runtime=runtime,
            protocol=_small_protocol(),
        )


def test_post_result_final_guard_failure_preserves_and_binds_partial(
    monkeypatch, tmp_path: Path,
) -> None:
    snapshot, runtime, authority_hash = _install_synthetic_authority(monkeypatch, tmp_path)
    calls = 0

    def changing_snapshot():
        nonlocal calls
        calls += 1
        return snapshot if calls <= 2 else {"fingerprint": "changed"}

    monkeypatch.setattr(freeze, "protected_snapshot", changing_snapshot)
    lock = FakeLock()
    with pytest.raises(RuntimeError, match="final authority inputs changed") as failure:
        collector.run_outcome(lock, runtime)
    assert freeze.RESULT.is_file()
    assert not freeze.OUTCOME_AUTHORITY.exists()
    collector.publish_failure(lock, authority_hash=authority_hash, error=failure.value)
    record = json.loads(freeze.FAILURE.read_text())
    assert record["result_authorized"] is False
    assert record["partial_result_sha256"] == collector.file_sha256(freeze.RESULT)


def test_failure_never_relabels_a_completed_outcome(monkeypatch, tmp_path: Path) -> None:
    _, runtime, authority_hash = _install_synthetic_authority(monkeypatch, tmp_path)
    lock = FakeLock()
    collector.run_outcome(lock, runtime)
    collector.publish_failure(lock, authority_hash=authority_hash, error=RuntimeError("late"))
    assert not freeze.FAILURE.exists()
