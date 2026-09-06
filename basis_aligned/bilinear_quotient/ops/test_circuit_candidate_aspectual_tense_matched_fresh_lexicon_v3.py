import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3 as subject


def test_fresh_matched_banks_are_sealed_disjoint_and_balanced():
    rows = subject.build_rows_by_bank()
    digests = subject.validate_rows_by_bank(rows)
    assert set(digests) == {"has_had", "is_was"}
    assert all(len(bank) == 64 for bank in rows.values())
    assert all(len({row["row_id"] for row in bank}) == 64 for bank in rows.values())
    assert {row["reporter"] for row in rows["has_had"]} == set(subject._AGENTS)
    assert {row["reporter"] for row in rows["is_was"]} == set(subject._AGENTS)
