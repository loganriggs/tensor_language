import json
from pathlib import Path

import pytest
import torch

import prepare_mlp0_c512_mlp2_compensation_v1_rows as rows


def test_registry_census_extracts_documents_indices_and_row_tensors(tmp_path, monkeypatch):
    tensor_path = tmp_path / "fineweb_rows.pt"
    tensor = torch.arange(2 * 40, dtype=torch.long).reshape(2, 40)
    torch.save(tensor, tensor_path)
    reference_path = tmp_path / "eval_tokens.pt"
    torch.save(torch.arange(40, dtype=torch.long).reshape(1, 40) + 100, reference_path)
    receipt = tmp_path / "role_rows_receipt.json"
    receipt.write_text(json.dumps({
        "entries": {"eval": {"cache_path": str(tensor_path)}},
        "document_provenance": {"sets": {"eval": [
            {"document_id": "doc-a", "dataset_document_index": 17}
        ]}},
    }))
    monkeypatch.setattr(rows, "REFERENCE_ROWS", reference_path)
    prior, registry_hashes, tensor_hashes = rows.load_registry_exclusions((receipt,))
    assert prior[0] == {"doc-a"} and prior[1] == {17}
    assert tuple(tensor[0].tolist()) in prior[2]
    assert tuple(tensor[0, :32].tolist()) in prior[3]
    assert str(receipt.resolve()) in registry_hashes
    assert {str(tensor_path.resolve()), str(reference_path.resolve())} == set(tensor_hashes)


def test_registry_census_fails_on_missing_registered_row_tensor(tmp_path):
    receipt = tmp_path / "role_manifest.json"
    receipt.write_text(json.dumps({"row_path": str(tmp_path / "missing_fineweb_rows.pt")}))
    with pytest.raises(RuntimeError, match="row-like tensor is missing"):
        rows.load_registry_exclusions((receipt,))


def test_long_row_tensor_filter_rejects_float_programs():
    assert list(rows.long_row_tensors(torch.zeros(3, 40))) == []
    assert len(list(rows.long_row_tensors({"rows": torch.zeros(3, 40, dtype=torch.long)}))) == 1
