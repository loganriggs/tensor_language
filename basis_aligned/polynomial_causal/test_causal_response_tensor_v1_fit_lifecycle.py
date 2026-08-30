import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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


def test_terminal_race_during_absence_scan_prevents_authority_publication(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    parents = {"weights_sha256": "d" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    original_exists = Path.exists
    attacked = False
    parent_calls = 0

    def replay_parents():
        nonlocal parent_calls
        parent_calls += 1
        return parents

    def inject_failure_during_scan(path):
        nonlocal attacked
        if path == lifecycle.AUTHORITY and parent_calls == 1 and not attacked:
            attacked = True
            lifecycle.FAILURE.write_text("injected concurrent failure\n")
        return original_exists(path)

    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(
        lifecycle, "parent_snapshot_without_tensor_load", replay_parents
    )
    monkeypatch.setattr(lifecycle, "_stable_audit", lambda: (audit, "e" * 64))
    monkeypatch.setattr(Path, "exists", inject_failure_during_scan)
    with pytest.raises(RuntimeError, match="raced publication"):
        lifecycle.freeze_fit_authority()
    assert attacked
    assert lifecycle.FAILURE.exists()
    assert not lifecycle.AUTHORITY.exists()


def test_authority_absence_lookup_precedes_final_parent_replay(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    closure = {"commit": "1" * 40, "paths": {"a": "b" * 64}, "sha256": "c" * 64}
    audit = {
        "reviewer": "independent-test-auditor",
        "audited_source_commit": closure["commit"],
    }
    live_parents = {"weights_sha256": "d" * 64}
    parent_calls = 0
    original_exists = Path.exists
    attacked = False

    def replay_parents():
        nonlocal parent_calls
        parent_calls += 1
        return dict(live_parents)

    def mutate_parent_from_authority_lookup(path):
        nonlocal attacked
        # require_pristine_namespace scans before the first parent snapshot; attack
        # only the publication scan after the staged authority has bound d...d.
        if path == lifecycle.AUTHORITY and parent_calls == 1 and not attacked:
            attacked = True
            live_parents["weights_sha256"] = "f" * 64
        return original_exists(path)

    monkeypatch.setattr(lifecycle, "source_closure", lambda *_args: closure)
    monkeypatch.setattr(lifecycle, "parent_snapshot_without_tensor_load", replay_parents)
    monkeypatch.setattr(lifecycle, "_stable_audit", lambda: (audit, "e" * 64))
    monkeypatch.setattr(Path, "exists", mutate_parent_from_authority_lookup)
    with pytest.raises(RuntimeError, match="protected state changed"):
        lifecycle.freeze_fit_authority()
    assert attacked and parent_calls == 2
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


def test_artifact_record_rejects_same_size_mutation_during_final_path_stat(
    monkeypatch, tmp_path
):
    target = tmp_path / "protected.bin"
    target.write_bytes(b"before!!")
    original_stat = lifecycle.os.stat
    attacked = False

    def mutate_during_path_stat(path, *args, **kwargs):
        nonlocal attacked
        if Path(path) == target and not attacked:
            attacked = True
            target.write_bytes(b"after!!!")  # exactly the same number of bytes
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(lifecycle.os, "stat", mutate_during_path_stat)
    with pytest.raises(RuntimeError, match="changed during stable observation"):
        lifecycle._artifact_record(target)
    assert attacked


def test_artifact_record_rejects_path_replacement_after_middle_stat(
    monkeypatch, tmp_path
):
    target = tmp_path / "protected.bin"
    target.write_bytes(b"old-bytes")
    original_stat = lifecycle.os.stat
    attacked = False

    def replace_after_obtaining_old_stat(path, *args, **kwargs):
        nonlocal attacked
        observed = original_stat(path, *args, **kwargs)
        if Path(path) == target and not attacked:
            attacked = True
            target.unlink()
            target.write_bytes(b"new-bytes")  # same size, different inode and content
        return observed

    monkeypatch.setattr(lifecycle.os, "stat", replace_after_obtaining_old_stat)
    with pytest.raises(RuntimeError, match="changed during stable observation"):
        lifecycle._artifact_record(target)
    assert attacked


def test_artifact_record_rejects_creation_after_absence_stat(monkeypatch, tmp_path):
    target = tmp_path / "absent.bin"
    original_stat = lifecycle.os.stat
    attacked = False

    def create_then_preserve_missing_result(path, *args, **kwargs):
        nonlocal attacked
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if Path(path) == target and not attacked:
                attacked = True
                target.write_bytes(b"appeared")
            raise

    monkeypatch.setattr(lifecycle.os, "stat", create_then_preserve_missing_result)
    with pytest.raises(RuntimeError, match="appeared during absence observation"):
        lifecycle._artifact_record(target)
    assert attacked and target.exists()


def test_artifact_record_binds_exact_descriptor_identity(tmp_path):
    target = tmp_path / "protected.bin"
    target.write_bytes(b"stable bytes")
    record = lifecycle._artifact_record(target)
    observed = target.stat()
    assert record == {
        "path": str(target), "present": True,
        "sha256": hashlib.sha256(b"stable bytes").hexdigest(),
        "bytes": len(b"stable bytes"), "device": observed.st_dev,
        "inode": observed.st_ino, "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def test_failure_guard_rechecks_absent_outputs_after_protected_observation(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    lifecycle.AUTHORITY.write_bytes(b"authority")
    protected = {"observed": "failure state"}
    monkeypatch.setattr(
        lifecycle, "_failure_protected_observation", lambda _model: protected
    )
    aggregate = {
        "authority": lifecycle._artifact_record(lifecycle.AUTHORITY),
        "bundle": lifecycle._artifact_record(lifecycle.BUNDLE),
        "manifest": lifecycle._artifact_record(lifecycle.MANIFEST),
        "protected_state": protected,
    }
    authority_digest = aggregate["authority"]["sha256"]
    original_record = lifecycle._artifact_record
    attacked = False

    def create_bundle_after_its_record(path):
        nonlocal attacked
        record = original_record(path)
        if path == lifecycle.MANIFEST and not attacked:
            attacked = True
            lifecycle.BUNDLE.write_bytes(b"late bundle")
        return record

    monkeypatch.setattr(lifecycle, "_artifact_record", create_bundle_after_its_record)
    claim = lifecycle.acquire_claim()
    try:
        with pytest.raises(RuntimeError, match="bundle appeared"):
            lifecycle._failure_guard(
                claim=claim, authority={},
                authority_artifact_sha256=authority_digest,
                aggregate=aggregate, model=None,
            )
        assert attacked
    finally:
        lifecycle.release_claim(claim)


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


def test_no_path_lookup_can_mutate_bundle_after_terminal_final_guard(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    lifecycle.BUNDLE.write_bytes(b"protected bundle")
    expected = lifecycle.file_sha256(lifecycle.BUNDLE)
    claim = lifecycle.acquire_claim()
    original_exists = Path.exists
    guarded = False
    attacked = False

    def mutate_from_post_guard_terminal_lookup(path):
        nonlocal attacked
        if path == lifecycle.TERMINAL and guarded and not attacked:
            attacked = True
            lifecycle.BUNDLE.write_bytes(b"mutated after guard")
        return original_exists(path)

    def final_guard():
        nonlocal guarded
        assert lifecycle.file_sha256(lifecycle.BUNDLE) == expected
        lifecycle.require_claim(claim)
        guarded = True

    monkeypatch.setattr(Path, "exists", mutate_from_post_guard_terminal_lookup)
    try:
        lifecycle._publish_terminal_record(
            _terminal_record("receipt"), kind="receipt", claim=claim,
            final_guard=final_guard,
        )
        assert not attacked
        assert lifecycle.file_sha256(lifecycle.BUNDLE) == expected
    finally:
        lifecycle.release_claim(claim)


def test_no_path_lookup_can_mutate_bundle_after_manifest_final_guard(
    monkeypatch, tmp_path
):
    _redirect_namespace(monkeypatch, tmp_path)
    lifecycle.BUNDLE.write_bytes(b"protected bundle")
    expected = lifecycle.file_sha256(lifecycle.BUNDLE)
    claim = lifecycle.acquire_claim()
    original_exists = Path.exists
    guarded = False
    attacked = False
    body = {
        "schema": "causal_response_tensor_v1_fit_manifest",
        "status": "complete_fit_bundle_semantically_replayed",
        "authority_artifact_sha256": "a" * 64,
        "authority_logical_sha256": "b" * 64,
        "bundle": {}, "bundle_summary": {}, "protocol": {},
        "authorized_for_eval": False,
    }
    manifest = {**body, "manifest_sha256": lifecycle.logical_sha256(body)}

    def mutate_from_post_guard_manifest_lookup(path):
        nonlocal attacked
        if path == lifecycle.MANIFEST and guarded and not attacked:
            attacked = True
            lifecycle.BUNDLE.write_bytes(b"mutated after guard")
        return original_exists(path)

    def final_guard():
        nonlocal guarded
        assert lifecycle.file_sha256(lifecycle.BUNDLE) == expected
        lifecycle.require_claim(claim)
        guarded = True

    monkeypatch.setattr(Path, "exists", mutate_from_post_guard_manifest_lookup)
    try:
        lifecycle.publish_fit_manifest(
            manifest, claim=claim, final_guard=final_guard
        )
        assert not attacked
        assert lifecycle.file_sha256(lifecycle.BUNDLE) == expected
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


def _install_owner_fakes(monkeypatch, tmp_path, *, collector_failure=False):
    _redirect_namespace(monkeypatch, tmp_path)
    events = []
    authority = {
        "authority_sha256": "a" * 64,
        "source_closure": {"sha256": "b" * 64},
        "parents": {
            "config_sha256": "c" * 64,
            "weights_sha256": "d" * 64,
        },
    }

    def freeze(claim):
        lifecycle.require_claim(claim)
        events.append("authority")
        lifecycle.AUTHORITY.write_text('{"frozen":true}\n')
        return authority

    monkeypatch.setattr(lifecycle, "_freeze_fit_authority_under_claim", freeze)
    monkeypatch.setattr(
        lifecycle, "validate_fit_authority",
        lambda: (authority, lifecycle.file_sha256(lifecycle.AUTHORITY)),
    )
    inputs = SimpleNamespace(
        rows=torch.zeros((1, 2), dtype=torch.int64),
        row_document_ids=torch.zeros(1, dtype=torch.int64),
        fit_row_indices=torch.zeros(1, dtype=torch.int64),
        specs=(),
        parent_sha256s={
            "census_state_diverse": "1" * 64,
            "curated_rows": "2" * 64,
            "battery": "3" * 64,
            "split": "4" * 64,
        },
        model_rows_sha256="5" * 64,
        fit_role_sha256="6" * 64,
        fit_document_ids_sha256="7" * 64,
        support_hashes={},
    )

    def reconstruct(guard):
        events.append("inputs")
        guard()
        guard()
        return inputs

    monkeypatch.setattr(
        lifecycle.fit_inputs, "_reconstruct_production_fit_inputs_after_authority",
        reconstruct,
    )
    model = tiny_model().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint = lifecycle.facade.CheckpointReceipt(
        revision="revision", snapshot="snapshot",
        config_sha256="c" * 64, weights_sha256="d" * 64,
        weights_bytes=123, tokenizer_vocab=32, logit_vocab=32,
    )

    def load_model(**kwargs):
        assert kwargs == {
            "device": "cuda", "dtype": torch.float32,
            "snapshot": lifecycle.facade.DEFAULT_SNAPSHOT,
            "verify_weights_sha256": True,
        }
        events.append("model")
        return model, checkpoint

    monkeypatch.setattr(lifecycle.facade, "load_bilin18", load_model)
    monkeypatch.setattr(lifecycle.facade, "validate_snapshot", lambda: checkpoint)
    monkeypatch.setattr(lifecycle, "model_state_sha256", lambda _model: "8" * 64)
    monkeypatch.setattr(
        lifecycle, "_failure_protected_observation",
        lambda _model: {"test_protected_state": "stable"},
    )

    class Collector:
        def __init__(self, *args, **kwargs):
            assert kwargs == {"batch_size": 4, "require_production": True}
            events.append("collector")

        def fit_stage(self, rows):
            assert rows is inputs.fit_row_indices
            events.append("fit")
            if collector_failure:
                raise RuntimeError("injected preregistered fit failure")
            return {"preimage": True}

    monkeypatch.setattr(lifecycle, "ObservedResponseCollector", Collector)
    payload = {"call_ledger": {"outer_forwards": 12_400}}
    monkeypatch.setattr(
        lifecycle.fit_bundle, "build_fit_bundle_payload",
        lambda preimage, binding, require_production: payload,
    )

    def publish_bundle(path, value, **kwargs):
        assert path == lifecycle.BUNDLE and value is payload
        assert kwargs["require_production"] is True
        events.append("bundle")
        kwargs["before_link"]()
        path.write_bytes(b"exact bundle bytes")
        return lifecycle.file_sha256(path)

    monkeypatch.setattr(lifecycle.fit_bundle, "publish_fit_bundle", publish_bundle)

    def replay_bundle(path, **kwargs):
        assert path == lifecycle.BUNDLE
        assert kwargs["expected_artifact_sha256"] == lifecycle.file_sha256(path)
        return kwargs["expected_artifact_sha256"]

    monkeypatch.setattr(
        lifecycle.fit_bundle, "semantic_replay_fit_bundle", replay_bundle
    )
    summary = {
        "schema": "causal_response_tensor_v1_fit_bundle_summary",
        "binding": {}, "scientific_contract": {}, "axes": {},
        "support_hashes": {}, "tensor_hashes": {}, "ledger": {},
    }
    monkeypatch.setattr(
        lifecycle.fit_bundle, "fit_bundle_manifest_summary", lambda *args, **kwargs: summary
    )
    return events


def test_production_owner_has_no_arguments_and_publishes_receipt_last(
    monkeypatch, tmp_path
):
    assert not inspect.signature(
        lifecycle.execute_causal_response_fit_v1
    ).parameters
    events = _install_owner_fakes(monkeypatch, tmp_path)
    digest = lifecycle.execute_causal_response_fit_v1()
    assert events[:5] == ["authority", "inputs", "model", "collector", "fit"]
    assert events[5] == "bundle"
    assert digest == lifecycle.file_sha256(lifecycle.RECEIPT)
    assert lifecycle.TERMINAL.stat().st_ino == lifecycle.RECEIPT.stat().st_ino
    assert lifecycle.BUNDLE.exists() and lifecycle.MANIFEST.exists()
    assert not lifecycle.FAILURE.exists() and not lifecycle.LOCK.exists()
    receipt = json.loads(lifecycle.RECEIPT.read_text())
    assert receipt["payload"]["outer_forwards"] == 12_400
    assert receipt["payload"]["authorized_for_eval"] is False
    assert receipt["aggregate"]["bundle"]["sha256"] == lifecycle.file_sha256(
        lifecycle.BUNDLE
    )


def test_production_owner_preserves_collector_failure_as_terminal(
    monkeypatch, tmp_path
):
    _install_owner_fakes(monkeypatch, tmp_path, collector_failure=True)
    with pytest.raises(RuntimeError, match="injected preregistered fit failure"):
        lifecycle.execute_causal_response_fit_v1()
    assert lifecycle.TERMINAL.stat().st_ino == lifecycle.FAILURE.stat().st_ino
    assert not lifecycle.RECEIPT.exists()
    failure = json.loads(lifecycle.FAILURE.read_text())
    assert failure["payload"]["error_type"] == "RuntimeError"
    assert failure["payload"]["authorized_for_eval"] is False
    assert not failure["aggregate"]["bundle"]["present"]
    assert not lifecycle.LOCK.exists()


def test_post_link_authority_drift_is_bound_in_failure_terminal(
    monkeypatch, tmp_path
):
    _install_owner_fakes(monkeypatch, tmp_path)
    drifted = False
    original_fsync = lifecycle._fsync_parent_best_effort
    authority = {
        "authority_sha256": "a" * 64,
        "source_closure": {"sha256": "b" * 64},
        "parents": {
            "config_sha256": "c" * 64,
            "weights_sha256": "d" * 64,
        },
    }

    def drift_during_authority_fsync(path):
        nonlocal drifted
        if path == lifecycle.AUTHORITY:
            drifted = True
        original_fsync(path)

    def reject_drifted_authority():
        if drifted:
            raise RuntimeError("FIT authority parent binding changed")
        return authority, lifecycle.file_sha256(lifecycle.AUTHORITY)

    monkeypatch.setattr(
        lifecycle, "_fsync_parent_best_effort", drift_during_authority_fsync
    )
    monkeypatch.setattr(lifecycle, "validate_fit_authority", reject_drifted_authority)
    monkeypatch.setattr(
        lifecycle, "_failure_protected_observation",
        lambda _model: {"test_parent_sha256": ("f" if drifted else "d") * 64},
    )
    with pytest.raises(RuntimeError, match="parent binding changed"):
        lifecycle.execute_causal_response_fit_v1()
    assert drifted
    failure = json.loads(lifecycle.FAILURE.read_text())
    assert failure["aggregate"]["protected_state"]["test_parent_sha256"] == "f" * 64
    assert lifecycle.TERMINAL.stat().st_ino == lifecycle.FAILURE.stat().st_ino
    assert not lifecycle.RECEIPT.exists()
