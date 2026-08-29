import json

import pytest
import torch

import prepare_terminal_copy_fit_inputs_v1 as prep


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
