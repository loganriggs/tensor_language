import hashlib
import json
from pathlib import Path

import pytest

import causal_response_tensor_v1_fit_lifecycle as lifecycle
from test_bilin18_observed_model_facade import tiny_model


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


def test_late_terminal_race_prevents_authority_publication(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    parent_calls = 0

    def replay_parents():
        nonlocal parent_calls
        parent_calls += 1
        if parent_calls == 2:
            lifecycle.FAILURE.write_text("injected concurrent failure\n")
        return parents

    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(lifecycle, "parent_snapshot_without_tensor_load", replay_parents)
    monkeypatch.setattr(lifecycle, "_stable_audit", lambda: (audit, "e" * 64))
    with pytest.raises(RuntimeError, match="raced publication"):
        lifecycle.freeze_fit_authority()
    assert lifecycle.FAILURE.exists()
    assert not lifecycle.AUTHORITY.exists()


def test_late_independent_audit_change_prevents_authority_publication(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    calls = 0

    def changing_audit():
        nonlocal calls
        calls += 1
        return audit, ("e" if calls == 1 else "f") * 64

    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(
        lifecycle, "parent_snapshot_without_tensor_load", lambda: parents
    )
    monkeypatch.setattr(lifecycle, "_stable_audit", changing_audit)
    with pytest.raises(RuntimeError, match="protected state changed"):
        lifecycle.freeze_fit_authority()
    assert not lifecycle.AUTHORITY.exists()


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


def test_authority_semantic_reload_joins_every_protected_binding(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda value: None)
    monkeypatch.setattr(
        lifecycle, "parent_snapshot_without_tensor_load", lambda: parents
    )
    monkeypatch.setattr(lifecycle, "_stable_audit", lambda: (audit, "e" * 64))
    body = {
        "schema": "causal_response_tensor_v1_fit_authority",
        "status": "frozen_before_any_parent_tensor_or_bilin18_model_load",
        "source_closure": closure,
        "independent_audit": {
            "path": str(lifecycle.AUDIT), "sha256": "e" * 64,
            "reviewer": audit["reviewer"],
        },
        "parents": parents,
        "protocol": lifecycle.protocol(),
        "output_paths": lifecycle.output_paths(),
        "outcome_access_before_authority": {
            "parent_tensors_loaded": False, "model_loaded": False,
            "model_forward_calls": 0, "scientific_outcomes_read": False,
        },
        "authorized_for_fit_execution": True,
        "authorized_for_eval": False,
    }
    authority = {**body, "authority_sha256": lifecycle.logical_sha256(body)}
    lifecycle.AUTHORITY.write_text(json.dumps(authority, sort_keys=True))
    replay, artifact_sha256 = lifecycle.validate_fit_authority()
    assert replay == authority
    assert artifact_sha256 == lifecycle.file_sha256(lifecycle.AUTHORITY)

    changed = dict(authority)
    changed["authorized_for_eval"] = True
    lifecycle.AUTHORITY.write_text(json.dumps(changed, sort_keys=True))
    with pytest.raises(RuntimeError, match="schema or role changed"):
        lifecycle.validate_fit_authority()


def test_model_state_logical_hash_detects_and_replays_mutation():
    model = tiny_model().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = lifecycle.model_state_sha256(model, require_production=False)
    assert before == lifecycle.model_state_sha256(model, require_production=False)
    with lifecycle.torch.no_grad():
        first = next(model.parameters())
        first.reshape(-1)[0] += 1
    after = lifecycle.model_state_sha256(model, require_production=False)
    assert after != before


def _terminal_record(kind):
    return {
        "schema": "causal_response_tensor_v1_fit_terminal",
        "kind": kind,
        "authority_artifact_sha256": "a" * 64,
        "authority_logical_sha256": "b" * 64,
        "aggregate": {"bundle_present": kind == "receipt"},
        "payload": {"status": "complete" if kind == "receipt" else "failed"},
    }


def test_success_terminal_and_receipt_are_one_create_only_inode(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    try:
        digest = lifecycle._publish_terminal_record(
            _terminal_record("receipt"), kind="receipt", claim=claim,
            final_guard=lambda: None,
        )
        assert lifecycle.file_sha256(lifecycle.TERMINAL) == digest
        assert lifecycle.file_sha256(lifecycle.RECEIPT) == digest
        assert lifecycle.TERMINAL.stat().st_ino == lifecycle.RECEIPT.stat().st_ino
        assert not lifecycle.FAILURE.exists()
        with pytest.raises(RuntimeError, match="already spent"):
            lifecycle._publish_terminal_record(
                _terminal_record("failure"), kind="failure", claim=claim,
                final_guard=lambda: None,
            )
    finally:
        lifecycle.release_claim(claim)


def test_failure_terminal_wins_the_same_aggregate(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    try:
        digest = lifecycle._publish_terminal_record(
            _terminal_record("failure"), kind="failure", claim=claim,
            final_guard=lambda: None,
        )
        assert lifecycle.file_sha256(lifecycle.TERMINAL) == digest
        assert lifecycle.file_sha256(lifecycle.FAILURE) == digest
        assert lifecycle.TERMINAL.stat().st_ino == lifecycle.FAILURE.stat().st_ino
        assert not lifecycle.RECEIPT.exists()
    finally:
        lifecycle.release_claim(claim)


def test_second_link_failure_leaves_complete_recoverable_terminal(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original_link = lifecycle.os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected receipt link failure")
        return original_link(source, target)

    monkeypatch.setattr(lifecycle.os, "link", fail_second_link)
    record = _terminal_record("receipt")
    try:
        with pytest.raises(OSError, match="injected receipt link failure"):
            lifecycle._publish_terminal_record(
                record, kind="receipt", claim=claim, final_guard=lambda: None,
            )
        assert json.loads(lifecycle.TERMINAL.read_text()) == record
        assert not lifecycle.RECEIPT.exists()
        assert not lifecycle.FAILURE.exists()
    finally:
        lifecycle.release_claim(claim)


def test_fallible_guard_runs_before_any_terminal_link(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    try:
        with pytest.raises(RuntimeError, match="late protected drift"):
            lifecycle._publish_terminal_record(
                _terminal_record("receipt"), kind="receipt", claim=claim,
                final_guard=lambda: (_ for _ in ()).throw(
                    RuntimeError("late protected drift")
                ),
            )
        assert not lifecycle.TERMINAL.exists()
        assert not lifecycle.RECEIPT.exists()
        assert not lifecycle.FAILURE.exists()
    finally:
        lifecycle.release_claim(claim)


def test_lock_replacement_during_terminal_aggregate_blocks_both_links(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original_exists = Path.exists
    attacked = False

    def replace_lock_on_aggregate(path):
        nonlocal attacked
        if path == lifecycle.TERMINAL and not attacked:
            attacked = True
            lifecycle.LOCK.unlink()
            # Preserve the nonce deliberately; only the device/inode check catches it.
            lifecycle.LOCK.write_text(claim.nonce + "\n")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", replace_lock_on_aggregate)
    try:
        with pytest.raises(RuntimeError, match="claim changed"):
            lifecycle._publish_terminal_record(
                _terminal_record("receipt"), kind="receipt", claim=claim,
                final_guard=lambda: None,
            )
        assert attacked
        assert not original_exists(lifecycle.TERMINAL)
        assert not original_exists(lifecycle.RECEIPT)
        assert not original_exists(lifecycle.FAILURE)
    finally:
        lifecycle.release_claim(claim)


def test_lock_loss_between_terminal_and_receipt_leaves_complete_terminal_only(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original_fsync = lifecycle._fsync_parent_best_effort
    attacked = False

    def replace_lock_after_terminal(path):
        nonlocal attacked
        original_fsync(path)
        if path == lifecycle.TERMINAL and not attacked:
            attacked = True
            lifecycle.LOCK.unlink()
            lifecycle.LOCK.write_text(claim.nonce + "\n")

    monkeypatch.setattr(lifecycle, "_fsync_parent_best_effort", replace_lock_after_terminal)
    record = _terminal_record("receipt")
    try:
        with pytest.raises(RuntimeError, match="claim changed"):
            lifecycle._publish_terminal_record(
                record, kind="receipt", claim=claim, final_guard=lambda: None,
            )
        assert attacked
        assert json.loads(lifecycle.TERMINAL.read_text()) == record
        assert not lifecycle.RECEIPT.exists()
        assert not lifecycle.FAILURE.exists()
    finally:
        lifecycle.release_claim(claim)


def test_in_place_terminal_mutation_blocks_final_receipt_link(monkeypatch, tmp_path):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original_fsync = lifecycle._fsync_parent_best_effort
    attacked = False

    def mutate_terminal_after_first_link(path):
        nonlocal attacked
        original_fsync(path)
        if path == lifecycle.TERMINAL and not attacked:
            attacked = True
            # TERMINAL and the staged temporary are hard links to this same inode.
            lifecycle.TERMINAL.write_text('{"mutated":true}\n')

    monkeypatch.setattr(
        lifecycle, "_fsync_parent_best_effort", mutate_terminal_after_first_link
    )
    try:
        with pytest.raises(RuntimeError, match="JSON hash changed"):
            lifecycle._publish_terminal_record(
                _terminal_record("receipt"), kind="receipt", claim=claim,
                final_guard=lambda: None,
            )
        assert attacked
        assert json.loads(lifecycle.TERMINAL.read_text()) == {"mutated": True}
        assert not lifecycle.RECEIPT.exists()
        assert not lifecycle.FAILURE.exists()
    finally:
        lifecycle.release_claim(claim)


def test_target_absence_check_cannot_mutate_terminal_after_final_replay(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    claim = lifecycle.acquire_claim()
    original_exists = Path.exists
    attacked = False

    def mutate_from_target_check(path):
        nonlocal attacked
        if path == lifecycle.RECEIPT and lifecycle.TERMINAL.is_file() and not attacked:
            attacked = True
            lifecycle.TERMINAL.write_text('{"mutated":"after-replay"}\n')
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", mutate_from_target_check)
    try:
        with pytest.raises(RuntimeError, match="JSON hash changed"):
            lifecycle._publish_terminal_record(
                _terminal_record("receipt"), kind="receipt", claim=claim,
                final_guard=lambda: None,
            )
        assert attacked
        assert not original_exists(lifecycle.RECEIPT)
        assert not original_exists(lifecycle.FAILURE)
    finally:
        lifecycle.release_claim(claim)
