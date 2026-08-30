from __future__ import annotations

import json

import pytest

import bracket_closure_execution_lifecycle_v1 as lifecycle
from test_bracket_closure_execution_v1 import _authority


def test_transaction_is_no_go_before_backend_access() -> None:
    class Poison:
        def __getattribute__(self, _name):
            raise AssertionError("backend touched")
    with pytest.raises(RuntimeError, match="prospectively NO-GO"):
        lifecycle.run_transaction(_authority(), Poison())


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
    lock_path = tmp_path / "lock2"
    with lifecycle.RunLock(lock_path) as lock:
        lock_path.unlink(); lock_path.write_text(lock.claim[2])
        with pytest.raises(RuntimeError, match="ownership changed"):
            lifecycle.publish_json_receipt_last(
                {"schema": "known"}, receipt, lock=lock, final_guard=lambda: None,
            )
    assert not receipt.exists()


def test_source_closure_binds_no_go_and_tests() -> None:
    assert "basis_aligned/polynomial_causal/BRACKET_CLOSURE_EXECUTION_V1_NO_GO.md" in (
        lifecycle.SOURCE_CLOSURE
    )
    assert lifecycle.SOURCE_CLOSURE[-2:] == (
        "basis_aligned/polynomial_causal/test_bracket_closure_execution_v1.py",
        "basis_aligned/polynomial_causal/test_bracket_closure_execution_lifecycle_v1.py",
    )
