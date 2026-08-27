import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_mlp0_c512_mlp1_interchange_v1_rows.py")
SPEC = importlib.util.spec_from_file_location("prepare_c512_interchange_rows", PATH)
ROWS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROWS)


def test_protocol_constants_and_distinct_authority_paths():
    assert ROWS.START_DOCUMENT_INDEX == 43_000
    assert ROWS.N_SOURCE_DOCUMENTS == 384
    assert ROWS.WAVE_DOCUMENTS == 192
    assert ROWS.CACHE != ROWS.BASE.CACHE
    assert ROWS.RECEIPT != ROWS.BASE.RECEIPT


def test_balanced_helper_keeps_new_documents_in_fixed_waves():
    texts = [(f"doc-{index}", "x" * 80) for index in range(8)]
    encode = lambda text: list(range(len(text) * 10, len(text) * 10 + 40))
    rows, records = ROWS.BASE.harvest_balanced_documents(
        texts, encode, set(), {"doc-1"}, start_document_index=1,
        n_source_documents=4, wave_documents=2, max_chunks_per_document=1,
        token_length=8,
    )
    assert tuple(rows.shape) == (4, 8)
    assert [record["document_id"] for record in records] == ["doc-2", "doc-3", "doc-4", "doc-5"]
    assert [record["wave"] for record in records] == ["A", "A", "B", "B"]


def test_disjointness_fails_closed_on_prior_prefix():
    rows = torch.arange(2 * ROWS.TOKEN_LENGTH, dtype=torch.long).view(2, ROWS.TOKEN_LENGTH)
    records = [
        {"document_id": "a", "dataset_document_index": 43_000, "wave": "A"},
        {"document_id": "b", "dataset_document_index": 43_001, "wave": "B"},
    ]
    prefix = tuple(int(value) for value in rows[0, :32])
    with pytest.raises(RuntimeError, match="prefix32"):
        ROWS.BASE.validate_eval_disjointness(rows, records, (set(), set(), set(), {prefix}))


def test_code_register_is_file_disjoint_and_file_resampled():
    receipt = ROWS.validate_code_register()
    assert receipt["heldout_row_interval"] == [288, 480]
    assert receipt["heldout_rows"] == 192
    assert receipt["heldout_source_files"] == 48
    assert receipt["resampling_unit"] == "source_file"
