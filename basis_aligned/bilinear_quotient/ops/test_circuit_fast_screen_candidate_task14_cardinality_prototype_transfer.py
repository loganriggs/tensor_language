from __future__ import annotations

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority


def test_authority_structure_and_novelty() -> None:
    rows = authority._build_unvalidated()
    digest = authority.validate_rows(rows, verify_hash=False)
    assert len(rows) == 32
    assert len({row["row_id"] for row in rows}) == 32
    assert len({endpoint["text"] for row in rows for endpoint in row["endpoints"].values()}) == 96
    assert len(digest) == 64


def test_counterfactuals_change_only_subject_token() -> None:
    for row in authority._build_unvalidated():
        recipient = row["endpoints"]["recipient"]["ids"]
        for role in ("opposite_same_lemma", "same_number_different_lemma"):
            alternate = row["endpoints"][role]["ids"]
            assert [index for index, pair in enumerate(zip(recipient, alternate)) if pair[0] != pair[1]] == [8]
