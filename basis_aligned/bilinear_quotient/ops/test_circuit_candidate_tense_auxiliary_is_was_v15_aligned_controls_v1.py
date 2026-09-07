from __future__ import annotations

from collections import Counter

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as original
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as aligned
from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract


def test_targets_are_preserved_exactly_and_every_panel_is_aligned():
    rows = aligned.build_rows()
    old_targets = [row for row in original.build_rows() if row["transform_id"] in {"A1", "A2"}]
    new_targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]

    assert new_targets == old_targets
    contract = derive_full_sequence_alignment_contract(
        rows, required_panels=("A1", "A2", "P", "C")
    )
    assert contract["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P", "C")}
    assert len(contract["row_ids"]) == 64


def test_paraphrase_controls_pair_same_tense_cross_cue_targets():
    rows = aligned.build_rows()
    by_key = {(row["group_number"], row["transform_id"]): row for row in rows}

    for group in range(aligned.GROUPS):
        a1, a2, control = (by_key[(group, panel)] for panel in ("A1", "A2", "P"))
        assert control["base_text"] == a1["base_text"]
        assert control["donor_text"] == a2["base_text"]
        assert control["base_answer"] == control["donor_answer"] == a1["base_answer"]
        assert len(control["base_ids"]) == len(control["donor_ids"])


def test_unrelated_controls_are_equal_length_same_answer_and_lexically_rotated():
    controls = [row for row in aligned.build_rows() if row["transform_id"] == "C"]

    assert Counter(len(row["base_ids"]) for row in controls) == {13: 16}
    for row in controls:
        assert row["base_answer"] == row["donor_answer"] == " night"
        assert row["base_text"] != row["donor_text"]
        assert len(row["base_ids"]) == len(row["donor_ids"])
        assert row["base_semantic_position"] == row["donor_semantic_position"]


def test_authority_digest_is_stable():
    assert aligned.authority_sha256() == (
        "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
    )
