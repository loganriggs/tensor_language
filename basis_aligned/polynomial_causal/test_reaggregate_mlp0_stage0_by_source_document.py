import numpy as np
import pytest

from reaggregate_mlp0_stage0_by_source_document import group_ledgers_by_document


def test_grouping_sums_chunks_within_source_document_in_first_seen_order():
    sums = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    counts = np.ones_like(sums)
    grouped_sums, grouped_counts, documents = group_ledgers_by_document(
        sums, counts, ["a", "b", "a"]
    )
    assert documents == ["a", "b"]
    assert np.array_equal(grouped_sums, [[6, 8], [3, 4]])
    assert np.array_equal(grouped_counts, [[2, 2], [1, 1]])


def test_grouping_requires_one_valid_document_id_per_row():
    with pytest.raises(ValueError, match="one source"):
        group_ledgers_by_document(np.ones((2, 1)), np.ones((2, 1)), ["a"])
    with pytest.raises(ValueError, match="nonempty"):
        group_ledgers_by_document(np.ones((1, 1)), np.ones((1, 1)), [""])
