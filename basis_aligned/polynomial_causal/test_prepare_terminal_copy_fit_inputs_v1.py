import json

import pytest
import torch

import prepare_terminal_copy_fit_inputs_v1 as prep


def _install_transaction(monkeypatch, tmp_path):
    parent_path = tmp_path / "parent.pt"
    rows = torch.arange(192 * 257, dtype=torch.long).reshape(192, 257)
    records = [
        {"document_id": f"doc-{index}", "role": "fit_natural", "role_row_index": index}
        for index in range(192)
    ]
    torch.save({
        "rows": rows, "records": records,
        "copy_cells": {"forbidden": torch.ones(1)},
        "synthetic": {"forbidden": torch.ones(1)},
    }, parent_path)
    parent = {
        "receipt_path": str(tmp_path / "parent_receipt.json"),
        "receipt_sha256": prep.PARENT_RECEIPT_SHA256,
        "container_path": str(parent_path),
        "container_sha256": prep.file_sha256(parent_path),
        "rows_tensor_sha256": prep.tensor_sha256(rows),
        "authorized_use": "fit_per_position_head_write_means_only",
    }
    erratum = tmp_path / "erratum.md"
    erratum.write_text("preserved exposure")
    audit_path = tmp_path / "audit.json"
    audit = {"approved": True, "outcome_access": False}
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(prep, "ERRATUM", erratum)
    monkeypatch.setattr(prep, "AUDIT", audit_path)
    monkeypatch.setattr(prep, "AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(prep, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(prep, "INPUTS", tmp_path / "cache" / "inputs.pt")
    monkeypatch.setattr(prep, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(prep, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(prep, "FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(prep, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(prep, "source_closure", lambda: {"closed": True})
    monkeypatch.setattr(prep, "verify_source", lambda value: None)
    monkeypatch.setattr(prep, "parent_binding", lambda: dict(parent))
    monkeypatch.setattr(prep, "validate_audit", lambda path=None: dict(audit))
    return parent


def test_projection_semantic_round_trip_has_no_forbidden_fields(tmp_path):
    tokens = torch.arange(192 * 256, dtype=torch.long).reshape(192, 256)
    documents = tuple(f"doc-{i}" for i in range(192))
    authority_sha = "a" * 64
    payload = {
        "schema": "terminal_copy_fit_inputs_v1_payload",
        "authority_sha256": authority_sha,
        "tokens": tokens,
        "ordered_document_ids": documents,
        "tokens_sha256": prep.tensor_sha256(tokens),
        "ordered_document_ids_sha256": prep.document_digest(documents),
    }
    path = tmp_path / "inputs.pt"
    torch.save(payload, path)
    replay = prep.validate_projection(path, authority_sha)
    assert set(replay) == {
        "schema", "authority_sha256", "tokens", "ordered_document_ids",
        "tokens_sha256", "ordered_document_ids_sha256",
    }
    assert not ({"copy_cells", "synthetic", "labels", "targets"} & set(replay))


def test_projection_rejects_label_column_and_tampering(tmp_path):
    tokens = torch.zeros(192, 257, dtype=torch.long)
    documents = tuple(f"doc-{i}" for i in range(192))
    payload = {
        "schema": "terminal_copy_fit_inputs_v1_payload",
        "authority_sha256": "a" * 64,
        "tokens": tokens,
        "ordered_document_ids": documents,
        "tokens_sha256": prep.tensor_sha256(tokens),
        "ordered_document_ids_sha256": prep.document_digest(documents),
    }
    path = tmp_path / "inputs.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="tensor semantics"):
        prep.validate_projection(path, "a" * 64)


def test_create_only_never_overwrites(tmp_path):
    path = tmp_path / "x.json"
    prep.create_only_json(path, {"first": 1})
    with pytest.raises(FileExistsError):
        prep.create_only_json(path, {"second": 2})
    assert json.loads(path.read_text()) == {"first": 1}


def test_parent_binding_is_fit_only_metadata():
    binding = prep.parent_binding()
    assert binding["authorized_use"] == "fit_per_position_head_write_means_only"
    assert binding["receipt_sha256"] == prep.PARENT_RECEIPT_SHA256


def test_full_projection_transaction_success_is_input_only(monkeypatch, tmp_path):
    _install_transaction(monkeypatch, tmp_path)
    receipt = prep.execute()
    assert receipt["status"] == "complete_receipt_last_input_only_no_model_access"
    assert receipt["parent_container_fully_deserialized_during_projection"] is True
    assert receipt["parent_container_fields_indexed"] == ["records", "rows"]
    assert receipt["E4_fit_model_forward_calls"] == 0
    payload = prep.validate_projection(prep.INPUTS, receipt["authority_sha256"])
    assert tuple(payload["tokens"].shape) == (192, 256)
    assert not ({"copy_cells", "synthetic", "labels"} & set(payload))
    prep.validate_published_metadata()


def test_full_projection_detects_late_input_reserialization(monkeypatch, tmp_path):
    _install_transaction(monkeypatch, tmp_path)
    original = prep.create_only_json

    def mutate_after_manifest(path, value):
        original(path, value)
        if path == prep.MANIFEST:
            payload = torch.load(prep.INPUTS, map_location="cpu", weights_only=True)
            torch.save(payload, prep.INPUTS)

    monkeypatch.setattr(prep, "create_only_json", mutate_after_manifest)
    with pytest.raises(RuntimeError, match="terminal recheck|adjacent receipt"):
        prep.execute()
    assert prep.FAILURE.exists()
    assert not prep.RECEIPT.exists()


def test_full_projection_detects_late_lock_replacement(monkeypatch, tmp_path):
    _install_transaction(monkeypatch, tmp_path)
    original = prep.create_only_json

    def replace_lock_after_manifest(path, value):
        original(path, value)
        if path == prep.MANIFEST:
            prep.LOCK.unlink()
            prep.LOCK.write_text("replacement\n")

    monkeypatch.setattr(prep, "create_only_json", replace_lock_after_manifest)
    with pytest.raises(RuntimeError, match="lock ownership changed"):
        prep.execute()
    assert not prep.RECEIPT.exists()
