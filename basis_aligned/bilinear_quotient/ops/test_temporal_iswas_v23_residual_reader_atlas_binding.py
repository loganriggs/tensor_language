import hashlib
import json
from pathlib import Path

import pytest

import temporal_iswas_v23_residual_reader_atlas_binding as target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_result():
    return {
        "schema": "temporal_iswas_v23_block11_residual_mlp_factorial_rescue_result_v1",
        "predictions": {key: True for key in target.PREDICTION_KEYS},
        "selected_component_or_threshold_after_outcome": None,
        "price": {"model_forwards_exact": 7, "sequence_evaluations_exact": 448},
    }


def test_real_binding_is_currently_fail_closed_while_result_is_pending():
    assert target.create_binding(dry_run=True) == {
        "status": "awaiting_result", "created": False
    }


def test_binding_payload_requires_dominant_residual_and_joint_closure():
    result = valid_result()
    payload = target.binding_payload(
        result, result_sha256="1" * 64,
        factorial_runner_sha256="2" * 64, atlas_runner_sha256="3" * 64,
    )
    assert payload["required_predictions"] == {
        key: True for key in target.REQUIRED
    }
    result["predictions"][target.PREDICTION_KEYS[1]] = False
    with pytest.raises(target.ResidualReaderBindingError):
        target.binding_payload(
            result, result_sha256="1" * 64,
            factorial_runner_sha256="2" * 64, atlas_runner_sha256="3" * 64,
        )


def test_create_binding_is_atomic_idempotent_and_refuses_stale_payload(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    factorial_runner = tmp_path / "factorial.py"
    atlas_runner = tmp_path / "atlas.py"
    binding = tmp_path / "binding.json"
    result_path.write_text(json.dumps(valid_result()))
    factorial_runner.write_text("factorial")
    atlas_runner.write_text("atlas")
    monkeypatch.setattr(target, "EXPECTED_FACTORIAL_RUNNER_SHA256", digest(factorial_runner))
    monkeypatch.setattr(target, "EXPECTED_ATLAS_RUNNER_SHA256", digest(atlas_runner))
    first = target.create_binding(
        factorial_result=result_path, factorial_runner=factorial_runner,
        atlas_runner=atlas_runner, binding=binding,
    )
    second = target.create_binding(
        factorial_result=result_path, factorial_runner=factorial_runner,
        atlas_runner=atlas_runner, binding=binding,
    )
    assert first["created"] is True and second["created"] is False
    binding.write_text(json.dumps({"schema": "stale"}))
    with pytest.raises(target.ResidualReaderBindingError):
        target.create_binding(
            factorial_result=result_path, factorial_runner=factorial_runner,
            atlas_runner=atlas_runner, binding=binding,
        )
