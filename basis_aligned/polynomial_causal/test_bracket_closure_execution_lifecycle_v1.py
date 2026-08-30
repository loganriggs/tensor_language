from __future__ import annotations

import json
import dataclasses

import pytest

import bracket_closure_execution_lifecycle_v1 as lifecycle
from test_bracket_closure_execution_v1 import _authority


def test_transaction_is_no_go_before_backend_access() -> None:
    class Poison:
        def __getattribute__(self, _name):
            raise AssertionError("backend touched")
    with pytest.raises(RuntimeError, match="lacks source-bound"):
        lifecycle.run_transaction(_authority(), Poison())


def test_ready_transaction_replays_source_before_rejecting_generic_backend(monkeypatch) -> None:
    authority = dataclasses.replace(
        _authority(), authorized_for_forward=True,
        inference_ruling_sha256="a" * 64, independent_audit_sha256="b" * 64,
    )
    calls = []
    monkeypatch.setattr(lifecycle, "verify_source_binding", lambda value: calls.append(value))
    with pytest.raises(RuntimeError, match="generic backend injection"):
        lifecycle.run_transaction(authority, object())
    assert calls == [authority]


def test_receipt_is_linked_last_only_after_guard_and_owned_lock(tmp_path) -> None:
    lock_path, receipt = tmp_path / "run.lock", tmp_path / "receipt.json"
    with lifecycle.RunLock(lock_path) as lock:
        lifecycle.publish_json_receipt_last(
            {"schema": "known", "complete": True}, receipt,
            lock=lock, final_guard=lambda: None,
        )
    assert json.loads(receipt.read_text()) == {"schema": "known", "complete": True}


def test_guard_failure_or_lock_swap_cannot_publish(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    with lifecycle.RunLock(tmp_path / "lock") as lock:
        with pytest.raises(RuntimeError, match="injected"):
            lifecycle.publish_json_receipt_last(
                {"schema": "known"}, receipt, lock=lock,
                final_guard=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
            )
    assert not receipt.exists()


def test_result_is_bound_then_receipt_is_linked_last(tmp_path) -> None:
    result, receipt = tmp_path / "result.json", tmp_path / "receipt.json"
    failure, terminal = tmp_path / "failure.json", tmp_path / "terminal.json"
    events = []
    with lifecycle.RunLock(tmp_path / "result.lock") as lock:
        lifecycle.publish_result_receipt_last(
            {"schema": "known", "promoted": False}, result, receipt,
            failure_path=failure, terminal_path=terminal,
            authority_sha256="a" * 64, lock=lock,
            final_guard=lambda: events.append((result.exists(), receipt.exists())),
        )
    assert events == [(False, False), (True, False), (True, False)]
    assert json.loads(receipt.read_text())["status"] == "complete_nonpromotion"
    assert json.loads(terminal.read_text())["status"] == "success_claimed_before_receipt"


def test_result_mutation_before_receipt_fails_closed(tmp_path) -> None:
    result, receipt = tmp_path / "result.json", tmp_path / "receipt.json"
    failure, terminal = tmp_path / "failure.json", tmp_path / "terminal.json"
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 2:
            result.write_text("{}")
    with lifecycle.RunLock(tmp_path / "result.lock") as lock:
        with pytest.raises(RuntimeError, match="aggregate changed"):
            lifecycle.publish_result_receipt_last(
                {"schema": "known", "promoted": True}, result, receipt,
                failure_path=failure, terminal_path=terminal,
                authority_sha256="a" * 64, lock=lock, final_guard=guard,
            )
    assert result.exists() and not receipt.exists() and not terminal.exists()
    lock_path = tmp_path / "lock2"
    with lifecycle.RunLock(lock_path) as lock:
        lock_path.unlink(); lock_path.write_text(lock.claim[2])
        with pytest.raises(RuntimeError, match="ownership changed"):
            lifecycle.publish_json_receipt_last(
                {"schema": "known"}, receipt, lock=lock, final_guard=lambda: None,
            )
    assert not receipt.exists()


def test_rival_failure_cannot_coexist_with_success_receipt(tmp_path) -> None:
    result, receipt = tmp_path / "result.json", tmp_path / "receipt.json"
    failure, terminal = tmp_path / "failure.json", tmp_path / "terminal.json"
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 2:
            failure.write_text("rival")
    with lifecycle.RunLock(tmp_path / "lock") as lock:
        with pytest.raises(RuntimeError, match="aggregate changed"):
            lifecycle.publish_result_receipt_last(
                {"schema": "known", "promoted": True}, result, receipt,
                failure_path=failure, terminal_path=terminal,
                authority_sha256="a" * 64, lock=lock, final_guard=guard,
            )
    assert result.exists() and failure.exists()
    assert not receipt.exists() and not terminal.exists()


def test_failure_binds_partial_result_and_common_terminal_claim(tmp_path) -> None:
    result, receipt = tmp_path / "result.json", tmp_path / "receipt.json"
    failure, terminal = tmp_path / "failure.json", tmp_path / "terminal.json"
    result.write_text('{"partial": true}\n')
    with lifecycle.RunLock(tmp_path / "lock") as lock:
        lifecycle.publish_failure_receipt_last(
            {"schema": "failed", "authority_sha256": "a" * 64}, failure,
            result_path=result, receipt_path=receipt, terminal_path=terminal,
            lock=lock, final_guard=lambda: None,
        )
    payload = json.loads(failure.read_text())
    assert payload["aggregate"]["result_present"] is True
    assert payload["aggregate"]["result_sha256"] == lifecycle.file_sha256(result)
    assert json.loads(terminal.read_text())["status"] == (
        "failure_claimed_before_failure_receipt"
    )
    with lifecycle.RunLock(tmp_path / "lock2") as lock:
        with pytest.raises(RuntimeError, match="spent"):
            lifecycle.publish_result_receipt_last(
                {"schema": "known", "promoted": False}, tmp_path / "other-result",
                receipt, failure_path=failure, terminal_path=terminal,
                authority_sha256="a" * 64, lock=lock, final_guard=lambda: None,
            )


def test_failure_guard_or_partial_result_race_cannot_publish(tmp_path) -> None:
    result, receipt = tmp_path / "result.json", tmp_path / "receipt.json"
    failure, terminal = tmp_path / "failure.json", tmp_path / "terminal.json"
    with lifecycle.RunLock(tmp_path / "lock") as lock:
        with pytest.raises(RuntimeError, match="protected input changed"):
            lifecycle.publish_failure_receipt_last(
                {"schema": "failed", "authority_sha256": "a" * 64}, failure,
                result_path=result, receipt_path=receipt, terminal_path=terminal,
                lock=lock, final_guard=lambda: (_ for _ in ()).throw(
                    RuntimeError("protected input changed")
                ),
            )
    assert not failure.exists() and not terminal.exists()
    with lifecycle.RunLock(tmp_path / "lock2") as lock:
        with pytest.raises(RuntimeError, match="aggregate changed"):
            lifecycle.publish_failure_receipt_last(
                {"schema": "failed", "authority_sha256": "a" * 64}, failure,
                result_path=result, receipt_path=receipt, terminal_path=terminal,
                lock=lock, final_guard=lambda: result.write_text("late"),
            )
    assert not failure.exists() and not terminal.exists()


def test_source_closure_binds_no_go_and_tests() -> None:
    assert "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_NO_GO.md" in (
        lifecycle.SOURCE_CLOSURE
    )
    assert {
        "basis_aligned/polynomial_causal/test_bracket_closure_execution_v1.py",
        "basis_aligned/polynomial_causal/test_bracket_closure_execution_lifecycle_v1.py",
        "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
        "basis_aligned/polynomial_causal/test_run_bracket_closure_execution_v1.py",
        "jacclust/__init__.py",
        "jacclust/tt_model.py",
    }.issubset(lifecycle.SOURCE_CLOSURE)
