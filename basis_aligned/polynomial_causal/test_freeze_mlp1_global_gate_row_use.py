from __future__ import annotations

import json

import pytest

import freeze_mlp1_global_gate_row_use as use


def test_row_use_authority_is_deterministic_and_preserves_training_prohibition() -> None:
    first, second = use.build_authority(), use.build_authority()
    assert first == second
    assert first["parent_scored_experiments_authorized"] is True
    assert first["parent_training_authorized"] is False
    assert first["model_training_forbidden"] is True
    assert first["wave_A_fit_authorized"]["select_physical_gate_support"] is True
    assert first["wave_A_fit_authorized"]["optimize_any_model_parameter_or_buffer"] is False
    assert first["wave_B_evaluation_only"]["select_or_modify_support"] is False
    assert first["wave_B_evaluation_only"]["fit_or_modify_coefficients"] is False
    assert first["authority_fingerprint"] == use.canonical_sha256({
        key: value for key, value in first.items() if key != "authority_fingerprint"
    })


def test_row_use_authority_refuses_after_an_outcome_exists(tmp_path, monkeypatch) -> None:
    result = tmp_path / "result.json"
    result.write_text("outcome")
    monkeypatch.setattr(use, "RESULT", result)
    monkeypatch.setattr(use, "BUNDLE", tmp_path / "bundle.pt")
    with pytest.raises(RuntimeError, match="outcomes exist"):
        use.build_authority()


def test_serialized_row_use_authority_equals_builder() -> None:
    assert json.loads(use.OUT.read_text()) == use.build_authority()
