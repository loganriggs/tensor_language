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
    terminal_directory = tmp_path / "terminal"
    monkeypatch.setattr(lifecycle, "TERMINAL_DIR", terminal_directory)
    for name, filename in {
        "AUTHORITY": "authority.json", "INPUT": "input.pt",
        "MANIFEST": "manifest.json",
        "AUDIT": "audit.json", "LOCK": "lock",
    }.items():
        monkeypatch.setattr(lifecycle, name, tmp_path / filename)
    monkeypatch.setattr(lifecycle, "RECEIPT", terminal_directory / "receipt.json")
    monkeypatch.setattr(lifecycle, "FAILURE", terminal_directory / "failure.json")
    monkeypatch.setattr(lifecycle, "TERMINAL", terminal_directory / "terminal.json")


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


def _mock_transaction(
    monkeypatch, tmp_path, *, loader_error=None,
    before_loader_error=None, after_input_publish=None,
):
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
        def load_once(
            self, *, parent_binding, expected_analysis_authority_artifact_sha256
        ):
            assert lifecycle.AUTHORITY.exists()
            assert parent_binding == fit_parent
            assert expected_analysis_authority_artifact_sha256 == lifecycle.file_sha256(
                lifecycle.AUTHORITY
            )
            events.append("loader_after_authority")
            if loader_error is not None:
                if before_loader_error is not None:
                    before_loader_error()
                raise loader_error
            return value

    monkeypatch.setattr(lifecycle, "OneUseFitTrainingLoader", FakeLoader)

    def fake_publish(path, built, *, before_link, **_kwargs):
        before_link()
        path.write_bytes(b"synthetic training input")
        if after_input_publish is not None:
            after_input_publish()
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


def test_input_drift_after_manifest_cannot_publish_success(monkeypatch, tmp_path):
    _mock_transaction(monkeypatch, tmp_path)
    original = lifecycle._publish_json

    def publish_then_mutate(value, target, guard):
        digest = original(value, target, guard)
        if target == lifecycle.MANIFEST:
            lifecycle.INPUT.write_bytes(b"mutated after manifest")
        return digest

    monkeypatch.setattr(lifecycle, "_publish_json", publish_then_mutate)
    with pytest.raises(RuntimeError, match="changed before receipt"):
        lifecycle.execute_training_input_v1()
    assert lifecycle.FAILURE.is_file() and lifecycle.TERMINAL.is_file()
    assert not lifecycle.RECEIPT.exists()


def test_failed_success_terminal_install_falls_back_to_complete_failure_pair(
    monkeypatch, tmp_path,
):
    _mock_transaction(monkeypatch, tmp_path)
    original = lifecycle._rename_directory_noreplace
    injected = {"done": False}

    def fail_success_once(source, target):
        if target == lifecycle.TERMINAL_DIR and not injected["done"]:
            injected["done"] = True
            raise OSError("injected success terminal rename failure")
        return original(source, target)

    monkeypatch.setattr(lifecycle, "_rename_directory_noreplace", fail_success_once)
    with pytest.raises(OSError, match="injected"):
        lifecycle.execute_training_input_v1()
    assert lifecycle.FAILURE.is_file() and lifecycle.TERMINAL.is_file()
    assert os.stat(lifecycle.FAILURE).st_ino == os.stat(lifecycle.TERMINAL).st_ino
    assert not lifecycle.RECEIPT.exists()


def test_terminal_install_is_create_only_against_empty_rival_directory(
    monkeypatch, tmp_path,
):
    _redirect(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    lifecycle.TERMINAL_DIR.mkdir()
    rival_inode = lifecycle.TERMINAL_DIR.stat().st_ino
    try:
        with pytest.raises(OSError):
            lifecycle._publish_terminal_pair(
                {"kind": "receipt"}, kind="receipt", claim=claim,
                final_guard=lambda: None,
            )
    finally:
        lifecycle.release_claim(claim)
    assert lifecycle.TERMINAL_DIR.is_dir()
    assert lifecycle.TERMINAL_DIR.stat().st_ino == rival_inode
    assert list(lifecycle.TERMINAL_DIR.iterdir()) == []


def test_terminal_success_has_no_post_install_filesystem_work(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    events = []
    original = lifecycle._rename_directory_noreplace

    def sync(_path):
        events.append("sync_before")

    def guard():
        events.append("guard")

    def install(source, target):
        events.append("install")
        original(source, target)

    monkeypatch.setattr(lifecycle, "_fsync_directory_best_effort", sync)
    monkeypatch.setattr(lifecycle, "_rename_directory_noreplace", install)
    try:
        lifecycle._publish_terminal_pair(
            {"kind": "receipt"}, kind="receipt", claim=claim,
            final_guard=guard,
        )
    finally:
        lifecycle.release_claim(claim)
    assert events == ["sync_before", "guard", "install"]
    assert lifecycle.RECEIPT.is_file() and lifecycle.TERMINAL.is_file()


def test_mutated_authority_failure_binds_current_bytes_not_stale_digest(
    monkeypatch, tmp_path,
):
    def mutate_authority():
        lifecycle.AUTHORITY.write_text('{"mutated":true}\n')

    _mock_transaction(
        monkeypatch, tmp_path,
        loader_error=RuntimeError("synthetic post-authority failure"),
        before_loader_error=mutate_authority,
    )
    with pytest.raises(RuntimeError, match="post-authority"):
        lifecycle.execute_training_input_v1()
    failure, _ = lifecycle.stable_json(lifecycle.FAILURE)
    current = lifecycle.file_sha256(lifecycle.AUTHORITY)
    assert failure["protected_observation"]["authority"]["sha256"] == current
    assert failure["attempt_authority_artifact_sha256"] != current


def test_source_drift_after_input_link_cannot_publish_success(monkeypatch, tmp_path):
    state = {"drifted": False}
    _mock_transaction(
        monkeypatch, tmp_path, after_input_publish=lambda: state.update(drifted=True)
    )
    original = lifecycle.source_closure

    def drifting_closure(commit):
        value = original(commit)
        if not state["drifted"]:
            return value
        return {**value, "sha256": "f" * 64}

    monkeypatch.setattr(lifecycle, "source_closure", drifting_closure)
    with pytest.raises(RuntimeError, match="protected state changed"):
        lifecycle.execute_training_input_v1()
    assert lifecycle.FAILURE.is_file() and not lifecycle.RECEIPT.exists()


def test_authority_post_link_replay_failure_still_publishes_failure_pair(
    monkeypatch, tmp_path,
):
    _mock_transaction(monkeypatch, tmp_path)
    original = lifecycle.stable_json
    injected = {"done": False}

    def replay_then_fail(path):
        value = original(path)
        if path == lifecycle.AUTHORITY and lifecycle.AUTHORITY.exists() and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("injected authority post-link replay failure")
        return value

    monkeypatch.setattr(lifecycle, "stable_json", replay_then_fail)
    with pytest.raises(RuntimeError, match="post-link replay"):
        lifecycle.execute_training_input_v1()
    assert lifecycle.AUTHORITY.is_file()
    assert lifecycle.FAILURE.is_file() and lifecycle.TERMINAL.is_file()
    assert not lifecycle.RECEIPT.exists()


def test_late_authority_mutation_cannot_publish_stale_success(monkeypatch, tmp_path):
    _mock_transaction(monkeypatch, tmp_path)
    original = lifecycle.parent.fit_parent_binding_without_tensor_load
    calls = {"count": 0, "mutated": False}

    def replay_then_mutate():
        value = original()
        calls["count"] += 1
        # The receipt guard is the first phase with a manifest already installed.
        if lifecycle.MANIFEST.exists() and not calls["mutated"]:
            lifecycle.AUTHORITY.write_text('{"late_mutation":true}\n')
            calls["mutated"] = True
        return value

    monkeypatch.setattr(
        lifecycle.parent, "fit_parent_binding_without_tensor_load", replay_then_mutate
    )
    with pytest.raises(RuntimeError, match="terminal boundary"):
        lifecycle.execute_training_input_v1()
    assert not lifecycle.RECEIPT.exists()
