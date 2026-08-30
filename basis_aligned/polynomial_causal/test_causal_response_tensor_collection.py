from pathlib import Path

import pytest
import torch

from causal_response_tensor_collection import (
    aggregate_document_responses,
    atomic_create_json,
    document_position_index,
    local_mask_from_global,
    validate_response_tensors,
)


def test_document_index_aggregates_two_rows_from_one_source_document() -> None:
    row_docs = torch.tensor([9, 4, 9, 7])
    documents, position_docs = document_position_index(
        row_docs, torch.tensor([0, 2, 3]), positions_per_row=4
    )
    assert documents.tolist() == [7, 9]
    assert position_docs.shape == (3 * 4,)
    assert int((position_docs == 1).sum()) == 8
    assert int((position_docs == 0).sum()) == 4


def test_local_mask_preserves_selected_row_order() -> None:
    mask = torch.zeros(4, 256, dtype=torch.bool)
    mask[2, 3] = True
    mask[0, 5] = True
    local = local_mask_from_global(mask.reshape(-1), torch.tensor([2, 0]))
    assert local.reshape(2, 256)[0, 3]
    assert local.reshape(2, 256)[1, 5]
    assert int(local.sum()) == 2


def test_document_aggregation_is_signed_additive_and_marks_missing_support() -> None:
    dce = torch.tensor([2.0, -1.0, 3.0, -4.0, 1.0, 2.0])
    position_docs = torch.tensor([0, 0, 0, 1, 1, 1])
    member = {"a": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)}
    off = {"a": torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.bool)}
    result = aggregate_document_responses(
        dce, position_docs, member, off, document_count=2
    )
    assert result["member_signed_sum"].tolist() == [[1.0, 0.0]]
    assert result["member_abs_sum"].tolist() == [[3.0, 0.0]]
    assert result["member_count"].tolist() == [[2, 0]]
    assert result["off_signed_sum"].tolist() == [[3.0, -1.0]]
    assert result["off_abs_sum"].tolist() == [[3.0, 7.0]]
    assert result["off_count"].tolist() == [[1, 3]]


def test_dense_response_validator_enforces_triangle_inequality() -> None:
    stats = {
        "member_signed_sum": torch.ones(2, 3, 1, 2),
        "member_abs_sum": torch.ones(2, 3, 1, 2),
        "off_signed_sum": torch.zeros(2, 3, 1, 2),
        "off_abs_sum": torch.zeros(2, 3, 1, 2),
    }
    counts = torch.tensor([[2, 0]])
    off_counts = torch.tensor([[4, 4]])
    summary = validate_response_tensors(
        stats, counts, off_counts, expected_prefix=(2, 3, 1)
    )
    assert summary["valid_cells"] == 6
    stats["member_abs_sum"][0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="triangle"):
        validate_response_tensors(stats, counts, off_counts, expected_prefix=(2, 3, 1))


def test_create_only_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    atomic_create_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        atomic_create_json(path, {"second": True})
    assert path.read_text().strip().endswith("}")
