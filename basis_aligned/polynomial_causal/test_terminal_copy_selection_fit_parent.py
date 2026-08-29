import pytest

import terminal_copy_fit_mean_lifecycle as life
import terminal_copy_selection_fit_parent as parent


def life_state():
    names = ("AUTHORITY", "BANK", "RESULT", "MANIFEST", "RECEIPT", "FAILURE", "LOCK")
    return (
        tuple((name, getattr(life, name)) for name in names),
        life.SOURCE_PATHS,
        life.PROTECTED_PATHS,
        life.protected_snapshot,
    )


def test_exact_v3_fit_parent_semantically_replays_without_self_authorization():
    before = life_state()
    binding = parent.replay_fit_parent()
    assert binding["document_count"] == 192
    assert binding["fit_receipt_self_authorizes_selection"] is False
    assert binding["requires_separate_selection_authority"] is True
    assert binding["fit_receipt_file_sha256"] == parent.V3_RECEIPT_SHA256
    assert life_state() == before


def test_fit_parent_rejects_any_bound_file_hash_change(monkeypatch):
    before = life_state()
    monkeypatch.setattr(parent, "V3_RECEIPT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="bytes changed"):
        parent.replay_fit_parent()
    assert life_state() == before


def test_fit_parent_restores_lifecycle_after_semantic_load_failure(monkeypatch):
    before = life_state()
    monkeypatch.setattr(
        life, "load_bank_semantically",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("load")),
    )
    with pytest.raises(RuntimeError, match="load"):
        parent.replay_fit_parent()
    assert life_state() == before


def test_fit_parent_terminal_aggregate_rehash_is_mandatory(monkeypatch):
    before = life_state()
    original = parent.file_sha256
    calls = {path: 0 for path in (parent.v3.AUTHORITY, parent.v3.BANK, parent.v3.RESULT,
                                  parent.v3.MANIFEST, parent.v3.RECEIPT)}

    def changed_at_terminal(path):
        if path in calls:
            calls[path] += 1
            # The bank is hashed initially, inside semantic load, and at the terminal
            # aggregate barrier; fail only the last parent-wide barrier.
            if path == parent.v3.AUTHORITY and calls[path] >= 4:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(parent, "file_sha256", changed_at_terminal)
    with pytest.raises(RuntimeError, match="aggregate replay"):
        parent.replay_fit_parent()
    assert life_state() == before
