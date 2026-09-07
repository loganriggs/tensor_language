from __future__ import annotations

import pytest

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as v15
from aligned_full_sequence_patch_contract import (
    FullSequenceAlignmentError,
    derive_full_sequence_alignment_contract,
)


def test_v15_target_panels_are_position_aligned():
    contract = derive_full_sequence_alignment_contract(
        v15.build_rows(), required_panels=("A1", "A2")
    )
    assert contract["panel_counts"] == {"A1": 16, "A2": 16}
    assert set(contract["token_lengths"].values()) == {5, 6, 7, 8}


@pytest.mark.parametrize("panel", ("P", "C"))
def test_v15_legacy_controls_fail_closed(panel):
    with pytest.raises(FullSequenceAlignmentError, match="not token-position aligned"):
        derive_full_sequence_alignment_contract(v15.build_rows(), required_panels=(panel,))


def test_semantic_position_mismatch_fails_even_at_equal_length():
    row = dict(next(row for row in v15.build_rows() if row["transform_id"] == "A1"))
    row["donor_semantic_position"] -= 1
    with pytest.raises(FullSequenceAlignmentError, match="semantic positions"):
        derive_full_sequence_alignment_contract([row], required_panels=("A1",))
