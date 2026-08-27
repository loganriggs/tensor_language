import pytest
import torch

import local_fineweb_harvest as LOCAL


def test_multi_spec_harvest_matches_skip_chunk_and_dedup_semantics():
    texts = [
        ("doc0", "0 1 2 3 4 5 6"),
        ("doc1", "10 11 12 13 14 15 16"),
        ("doc2", "20 21 22 23 24 25 26"),
    ]
    encode = lambda text: [int(value) for value in text.split()]
    tensors, provenance = LOCAL.harvest_texts(
        texts, ((2, 0), (1, 1)), encode, {(0, 1, 2)}, token_length=3
    )
    # range(0, len-3, 3) produces starts 0 and 3 for seven tokens.  doc0's
    # first chunk is deduplicated, so its second chunk is the first output.
    assert torch.equal(tensors[(2, 0)], torch.tensor([[3, 4, 5], [10, 11, 12]]))
    assert torch.equal(tensors[(1, 1)], torch.tensor([[10, 11, 12]]))
    assert provenance[(2, 0)][0]["document_id"] == "doc0"
    assert provenance[(2, 0)][0]["chunk_id"] == 1
    assert provenance[(1, 1)][0]["dataset_document_index"] == 1


def test_harvest_rejects_incomplete_or_invalid_specs():
    with pytest.raises(ValueError, match="unique"):
        LOCAL.harvest_texts([], ((1, 0), (1, 0)), lambda _: [], set(), token_length=3)
    with pytest.raises(RuntimeError, match="ended before"):
        LOCAL.harvest_texts(
            [("doc", "1 2 3")], ((1, 0),),
            lambda text: [int(value) for value in text.split()], set(), token_length=3,
        )


def test_build_requires_exactly_one_source_without_indexing_empty_list():
    with pytest.raises(RuntimeError, match="exactly one"):
        LOCAL.build_shadow([])
    with pytest.raises(RuntimeError, match="exactly one"):
        LOCAL.build_shadow([LOCAL.Path("one"), LOCAL.Path("two")])


def test_shadow_status_cannot_be_mistaken_for_a_license():
    assert "unlicensed" in LOCAL.__doc__.lower()
    assert (8, 40) in LOCAL.SPECS
    assert len(LOCAL.PINNED_SHA256) == 64
