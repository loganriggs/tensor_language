import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_mlp0_native_down_hierarchy_v1_rows.py")
SPEC = importlib.util.spec_from_file_location("prepare_mlp0_native_rows", PATH)
ROWS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWS)


def fake_texts(n=12):
    for index in range(n):
        yield f"doc-{index}", "x" * (20 + index)


def fake_encode(text):
    base = len(text)
    return list(range(base * 10, base * 10 + 28))


def test_balanced_harvest_caps_chunks_and_keeps_documents_in_one_wave():
    rows, records = ROWS.harvest_balanced_documents(
        fake_texts(), fake_encode, set(), set(), start_document_index=2,
        n_source_documents=4, wave_documents=2, max_chunks_per_document=2,
        token_length=5,
    )
    assert tuple(rows.shape) == (8, 5)
    assert {record["document_id"] for record in records} == {
        "doc-2", "doc-3", "doc-4", "doc-5"
    }
    assert {record["wave"] for record in records if record["document_id"] in {"doc-2", "doc-3"}} == {"A"}
    assert {record["wave"] for record in records if record["document_id"] in {"doc-4", "doc-5"}} == {"B"}
    assert all(sum(row["document_id"] == document for row in records) == 2
               for document in {row["document_id"] for row in records})


def test_balanced_harvest_skips_excluded_documents_and_seen_prefixes():
    seen = {tuple(fake_encode("x" * 22)[:5])}
    rows, records = ROWS.harvest_balanced_documents(
        fake_texts(), fake_encode, seen, {"doc-3"}, start_document_index=2,
        n_source_documents=2, wave_documents=1, max_chunks_per_document=1,
        token_length=5,
    )
    assert tuple(rows.shape) == (2, 5)
    assert [record["document_id"] for record in records] == ["doc-2", "doc-4"]
    assert int(rows[0, 0]) == 225  # the seen first chunk was skipped within doc-2


def test_harvest_fails_closed_without_enough_eligible_documents():
    with pytest.raises(RuntimeError, match="eligible documents"):
        ROWS.harvest_balanced_documents(
            fake_texts(3), fake_encode, set(), set(), start_document_index=2,
            n_source_documents=2, wave_documents=1, max_chunks_per_document=1,
            token_length=5,
        )


def test_disjointness_checks_documents_indices_rows_prefixes_and_waves():
    rows = torch.arange(2 * ROWS.TOKEN_LENGTH, dtype=torch.long).view(2, ROWS.TOKEN_LENGTH)
    records = [
        {"document_id": "a", "dataset_document_index": 1, "wave": "A"},
        {"document_id": "b", "dataset_document_index": 2, "wave": "B"},
    ]
    gates = ROWS.validate_eval_disjointness(rows, records, (set(), set(), set(), set()))
    assert all(gates.values())
    with pytest.raises(RuntimeError, match="source_documents"):
        ROWS.validate_eval_disjointness(rows, records, ({"a"}, set(), set(), set()))
    with pytest.raises(RuntimeError, match="replication_waves"):
        bad = [dict(records[0]), dict(records[1], document_id="a")]
        ROWS.validate_eval_disjointness(rows, bad, (set(), set(), set(), set()))
