import hashlib
import json
from pathlib import Path

import pytest

import temporal_iswas_v24_ood_confirmation_binding as target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_result():
    return {
        "schema": "tense_auxiliary_is_was_fresh_lexicon_v24_capability_result_v1",
        "predictions": {key: True for key in target.PREDICTION_KEYS},
        "terminal": "screen", "causal_outcomes_opened": False,
        "rows_sha256": target.ROWS_SHA256,
        "jointly_capable_row_ids": {family: [] for family in target.FAMILIES},
        "family_capable": {family: True for family in target.FAMILIES},
        "price": {"model_forwards": 2, "example_evaluations": 128,
                  "interventions": 0, "transformer_backwards": 0,
                  "model_updates": 0},
    }


def test_real_binding_waits_without_opening_causal_outcomes():
    assert target.create_binding(dry_run=True) == {
        "status": "awaiting_result", "created": False
    }


def test_payload_requires_every_family_and_closed_causal_outcomes():
    result = valid_result()
    payload = target.binding_payload(
        result, result_sha256="1" * 64,
        capability_runner_sha256="2" * 64,
        confirmation_runner_sha256="3" * 64,
    )
    assert payload["all_families_capable"] is True
    assert payload["causal_outcomes_opened"] is False
    result["family_capable"]["P"] = False
    with pytest.raises(target.V24ConfirmationBindingError):
        target.binding_payload(
            result, result_sha256="1" * 64,
            capability_runner_sha256="2" * 64,
            confirmation_runner_sha256="3" * 64,
        )


def test_create_binding_is_atomic_idempotent_and_hash_bound(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    capability_runner = tmp_path / "capability.py"
    confirmation_runner = tmp_path / "confirmation.py"
    binding = tmp_path / "binding.json"
    result_path.write_text(json.dumps(valid_result()))
    capability_runner.write_text("capability")
    confirmation_runner.write_text("confirmation")
    monkeypatch.setattr(target, "EXPECTED_CAPABILITY_RUNNER_SHA256", digest(capability_runner))
    monkeypatch.setattr(target, "EXPECTED_CONFIRMATION_RUNNER_SHA256", digest(confirmation_runner))
    first = target.create_binding(
        capability_result=result_path, capability_runner=capability_runner,
        confirmation_runner=confirmation_runner, binding=binding,
    )
    second = target.create_binding(
        capability_result=result_path, capability_runner=capability_runner,
        confirmation_runner=confirmation_runner, binding=binding,
    )
    assert first["created"] is True and second["created"] is False
    confirmation_runner.write_text("changed")
    with pytest.raises(target.V24ConfirmationBindingError):
        target.create_binding(
            capability_result=result_path, capability_runner=capability_runner,
            confirmation_runner=confirmation_runner, binding=binding,
        )
