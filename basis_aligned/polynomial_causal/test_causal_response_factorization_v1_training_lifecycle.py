import hashlib
import inspect
import os

import pytest

import causal_response_factorization_v1_training_lifecycle as lifecycle
from test_causal_response_factorization_v1_fit_adapter import (
    _analysis_input,
    _parent_binding,
)


def _redirect(monkeypatch, tmp_path):
    for name, filename in {
        "AUTHORITY": "authority.json", "INPUT": "input.pt",
        "MANIFEST": "manifest.json", "RECEIPT": "receipt.json",
        "FAILURE": "failure.json", "TERMINAL": "terminal.json",
        "AUDIT": "audit.json", "LOCK": "lock",
    }.items():
        monkeypatch.setattr(lifecycle, name, tmp_path / filename)


def test_training_lifecycle_protocol_exposes_no_validation_or_eval():
    value = lifecycle.protocol()
    assert value["role"] == "FIT_TRAINING"
    assert value["training_documents"] == 229
    assert value["validation_documents_exposed"] == 0
    assert value["eval_documents_exposed"] == 0
    assert value["authorized_for_validation"] is False
    assert value["authorized_for_eval"] is False
    assert len(inspect.signature(lifecycle.execute_training_input_v1).parameters) == 0


def test_claim_rejects_replacement_and_preserves_attacker_file(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original = tmp_path / "original"
    lifecycle.LOCK.rename(original)
    lifecycle.LOCK.write_text("attacker\n")
    try:
        with pytest.raises(RuntimeError, match="claim changed"):
            lifecycle.require_claim(claim)
    finally:
        lifecycle.release_claim(claim)
    assert lifecycle.LOCK.read_text() == "attacker\n"


def _mock_transaction(monkeypatch, tmp_path, *, loader_error=None):
    _redirect(monkeypatch, tmp_path)
    payload, value = _analysis_input()
    fit_parent = _parent_binding(payload)
    closure = {
        "commit": "1" * 40,
        "paths": {"one": "2" * 64},
        "sha256": "3" * 64,
    }
    audit = {
        "reviewer": "fixture", "audited_source_commit": closure["commit"],
    }
    monkeypatch.setattr(lifecycle, "stable_audit", lambda: (audit, "4" * 64))
    monkeypatch.setattr(lifecycle, "source_closure", lambda _commit: closure)
    monkeypatch.setattr(
        lifecycle.parent, "fit_parent_binding_without_tensor_load", lambda: fit_parent
    )

    events = []

    class FakeLoader:
        def load_once(self, *, parent_binding, analysis_authority):
            assert lifecycle.AUTHORITY.exists()
            assert parent_binding == fit_parent
            assert analysis_authority["authorized_for_training_input"] is True
            events.append("loader_after_authority")
            if loader_error is not None:
                raise loader_error
            return value

    monkeypatch.setattr(lifecycle, "OneUseFitTrainingLoader", FakeLoader)

    def fake_publish(path, built, *, before_link, **_kwargs):
        before_link()
        path.write_bytes(b"synthetic training input")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fake_replay(path, **_kwargs):
        return value, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        lifecycle.training_input, "publish_training_input", fake_publish
    )
    monkeypatch.setattr(
        lifecycle.training_input, "replay_training_input", fake_replay
    )
    return events


def test_mocked_execute_freezes_authority_before_loader_and_receipt_last(
    monkeypatch, tmp_path,
):
    events = _mock_transaction(monkeypatch, tmp_path)
    digest = lifecycle.execute_training_input_v1()
    assert len(digest) == 64
    assert events == ["loader_after_authority"]
    assert lifecycle.AUTHORITY.is_file()
    assert lifecycle.INPUT.is_file()
    assert lifecycle.MANIFEST.is_file()
    assert lifecycle.RECEIPT.is_file() and lifecycle.TERMINAL.is_file()
    assert os.stat(lifecycle.RECEIPT).st_ino == os.stat(lifecycle.TERMINAL).st_ino
    assert not lifecycle.FAILURE.exists() and not lifecycle.LOCK.exists()
    receipt, observed = lifecycle.stable_json(lifecycle.RECEIPT)
    assert observed == digest
    assert receipt["payload"]["validation_values_read"] is False
    assert receipt["payload"]["eval_values_read"] is False


def test_mocked_loader_failure_publishes_failure_not_receipt(monkeypatch, tmp_path):
    _mock_transaction(
        monkeypatch, tmp_path, loader_error=RuntimeError("synthetic loader failure")
    )
    with pytest.raises(RuntimeError, match="synthetic loader failure"):
        lifecycle.execute_training_input_v1()
    assert lifecycle.AUTHORITY.is_file()
    assert lifecycle.FAILURE.is_file() and lifecycle.TERMINAL.is_file()
    assert os.stat(lifecycle.FAILURE).st_ino == os.stat(lifecycle.TERMINAL).st_ino
    assert not lifecycle.RECEIPT.exists()
    failure, _ = lifecycle.stable_json(lifecycle.FAILURE)
    assert failure["payload"]["status"] == "failed_no_training_receipt"
    assert failure["payload"]["authorized_for_validation"] is False
    assert failure["payload"]["authorized_for_eval"] is False


def test_spent_namespace_prevents_second_execution(monkeypatch, tmp_path):
    _mock_transaction(monkeypatch, tmp_path)
    lifecycle.execute_training_input_v1()
    with pytest.raises(RuntimeError, match="namespace is spent"):
        lifecycle.execute_training_input_v1()
