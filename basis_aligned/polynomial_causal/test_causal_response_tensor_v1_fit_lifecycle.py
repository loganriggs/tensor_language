import hashlib
import json
from pathlib import Path

import pytest

import causal_response_tensor_v1_fit_lifecycle as lifecycle


def _redirect_namespace(monkeypatch, root: Path) -> None:
    for name, filename in {
        "AUTHORITY": "authority.json", "BUNDLE": "bundle.pt",
        "MANIFEST": "manifest.json", "RECEIPT": "receipt.json",
        "FAILURE": "failure.json", "TERMINAL": "terminal.json",
        "LOCK": "run.lock", "AUDIT": "audit.json",
    }.items():
        monkeypatch.setattr(lifecycle, name, root / filename)


def test_protocol_is_fit_only_and_binds_every_input_identity():
    value = lifecycle.protocol()
    assert value["role"] == "FIT"
    assert value["authorized_for_eval"] is False
    assert value["authorized_for_factor_selection"] is False
    assert value["rows"] == 496
    assert value["source_documents"] == 343
    assert value["sources"] == value["targets"] == 49
    assert value["projection_event_shape"] == [2, 49, 124]
    assert value["fit_document_ids_sha256"] == (
        "0f514805a7615e5ef3fe862eb8bf37bebfe8c57b8b7e781fbb25907c729b808d"
    )


def test_source_closure_canonicalizes_short_and_full_commit_names(monkeypatch):
    # Isolate commit-name canonicalization from the intentional dirty source under test.
    monkeypatch.setattr(lifecycle, "SOURCE_PATHS", ())
    full = lifecycle.source_closure("b117acbb")
    assert len(full["commit"]) == 40
    assert lifecycle.source_closure(full["commit"]) == full


def test_parent_snapshot_hashes_bytes_without_deserializing(monkeypatch):
    monkeypatch.setattr(
        lifecycle.fit_inputs.torch, "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority phase deserialized a tensor")
        ),
    )
    snapshot = lifecycle.parent_snapshot_without_tensor_load()
    assert snapshot["weights_sha256"] == (
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    )


def test_claim_rejects_replacement_and_does_not_delete_attacker_file(tmp_path):
    lock = tmp_path / "run.lock"
    claim = lifecycle.acquire_claim(lock)
    original = tmp_path / "original.lock"
    lock.rename(original)
    lock.write_text("attacker\n")
    try:
        with pytest.raises(RuntimeError, match="claim changed"):
            lifecycle.require_claim(claim)
    finally:
        lifecycle.release_claim(claim)
    assert lock.read_text() == "attacker\n"


def test_claim_rejects_nonce_mutation(tmp_path):
    lock = tmp_path / "run.lock"
    claim = lifecycle.acquire_claim(lock)
    try:
        lock.write_text("changed\n")
        with pytest.raises(RuntimeError, match="claim changed"):
            lifecycle.require_claim(claim)
    finally:
        lifecycle.release_claim(claim)


def test_authority_draft_is_explicitly_nonauthorizing(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(
        lifecycle, "parent_snapshot_without_tensor_load", lambda: parents
    )
    draft = lifecycle.build_authority_draft()
    assert draft["authorized_for_fit_execution"] is False
    assert draft["authorized_for_eval"] is False
    assert draft["source_closure"] == closure
    assert draft["parents"] == parents
    assert not lifecycle.AUTHORITY.exists()


def test_freeze_publishes_authority_create_only_before_parent_tensor_load(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(
        lifecycle, "parent_snapshot_without_tensor_load", lambda: parents
    )
    monkeypatch.setattr(lifecycle, "_stable_audit", lambda: (audit, "e" * 64))
    monkeypatch.setattr(
        lifecycle.fit_inputs.torch, "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authority phase deserialized a tensor")
        ),
    )

    authority = lifecycle.freeze_fit_authority()
    replay, digest = lifecycle.stable_json(lifecycle.AUTHORITY)
    assert replay == authority
    assert digest == hashlib.sha256(lifecycle.AUTHORITY.read_bytes()).hexdigest()
    assert authority["authorized_for_fit_execution"] is True
    assert authority["authorized_for_eval"] is False
    assert not lifecycle.LOCK.exists()
    with pytest.raises(RuntimeError, match="namespace is spent"):
        lifecycle.freeze_fit_authority()


def test_independent_audit_must_be_exact_source_bound_go(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    bad = {
        "schema": "causal_response_tensor_v1_fit_lifecycle_independent_audit",
        "status": "NO-GO",
        "approved": False,
        "outcome_access": False,
        "reviewer": "auditor",
        "audited_source_commit": "1" * 40,
        "audited_source_hashes": {},
        "tests_passed": 1,
        "remaining_execution_blockers": ["one"],
    }
    lifecycle.AUDIT.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="not an exact execution GO"):
        lifecycle._stable_audit()
