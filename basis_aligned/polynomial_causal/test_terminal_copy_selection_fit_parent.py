import pytest

import terminal_copy_selection_fit_parent as parent


def test_exact_v3_fit_parent_semantically_replays_without_self_authorization():
    binding = parent.replay_fit_parent()
    assert binding["document_count"] == 192
    assert binding["fit_receipt_self_authorizes_selection"] is False
    assert binding["requires_separate_selection_authority"] is True
    assert binding["fit_receipt_file_sha256"] == parent.V3_RECEIPT_SHA256


def test_fit_parent_rejects_any_bound_file_hash_change(monkeypatch):
    monkeypatch.setattr(parent, "V3_RECEIPT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="bytes changed"):
        parent.replay_fit_parent()
