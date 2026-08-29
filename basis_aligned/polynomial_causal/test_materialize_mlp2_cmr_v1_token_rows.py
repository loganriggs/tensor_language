from __future__ import annotations

import json

import pytest
import torch

import materialize_mlp2_cmr_v1_token_rows as rows


def fake_encode(text: str) -> list[int]:
    return list(range(int(text)))


def test_short_documents_are_eot_padded_and_masked_not_replaced(monkeypatch) -> None:
    monkeypatch.setattr(rows, "DOCUMENTS", 3)
    indices = (10, 11, 12)
    value = rows.tokenize_documents(indices, {10: "300", 11: "100", 12: "64"}, fake_encode)
    assert value["rows"].shape == (3, 257)
    assert value["original_token_counts"].tolist() == [300, 100, 64]
    assert value["clipped_token_counts"].tolist() == [257, 100, 64]
    assert int(value["eligible_mask"][0].sum()) == 192
    assert int(value["eligible_mask"][1].sum()) == 35
    assert int(value["eligible_mask"][2].sum()) == 0
    assert torch.equal(value["rows"][1, 100:], torch.full((157,), rows.EOT))
    assert value["rows"][0, 256] == 256


def test_mask_uses_only_targets_inside_original_document(monkeypatch) -> None:
    monkeypatch.setattr(rows, "DOCUMENTS", 1)
    value = rows.tokenize_documents((7,), {7: "66"}, fake_encode)
    positions = value["eligible_mask"][0].nonzero().flatten().tolist()
    assert positions == [64]
    assert value["rows"][0, 65] == 65
    assert value["rows"][0, 66] == rows.EOT


def test_validator_reconstructs_mask_and_support_gates(monkeypatch) -> None:
    monkeypatch.setattr(rows, "DOCUMENTS", 2)
    monkeypatch.setattr(rows, "MIN_SUPPORT_DOCUMENTS", 2)
    monkeypatch.setattr(rows, "MIN_ELIGIBLE_POSITIONS", 2)
    value = rows.tokenize_documents((1, 2), {1: "100", 2: "80"}, fake_encode)
    summary = rows.validate_role(value)
    assert summary["support_documents"] == 2
    assert summary["eligible_positions"] == 50
    value["eligible_mask"][0, 0] = True
    with pytest.raises(RuntimeError, match="eligible mask"):
        rows.validate_role(value)


def test_invalid_tokens_and_missing_frozen_text_fail(monkeypatch) -> None:
    monkeypatch.setattr(rows, "DOCUMENTS", 1)
    with pytest.raises(ValueError, match="missing text"):
        rows.tokenize_documents((1,), {}, fake_encode)
    with pytest.raises(ValueError, match="invalid token"):
        rows.tokenize_documents((1,), {1: "x"}, lambda _: [50_257])


def test_create_only_writer_never_overwrites(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    rows.write_create_only(path, json.dumps({"x": 1}).encode())
    with pytest.raises(FileExistsError):
        rows.write_create_only(path, json.dumps({"x": 2}).encode())
    assert json.loads(path.read_text()) == {"x": 1}
