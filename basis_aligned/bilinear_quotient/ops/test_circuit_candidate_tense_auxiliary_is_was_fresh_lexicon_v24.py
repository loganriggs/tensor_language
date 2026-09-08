import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24 as candidate

def test_v24_is_complete_aligned_and_novel_by_construction():
    rows=candidate.build_rows()
    assert len(rows) == 64 and len({row["row_id"] for row in rows}) == 64
    assert {row["family"] for row in rows} == {"A1","A2","P","C"}
    assert all(row["base_semantic_position"] == row["donor_semantic_position"] for row in rows)
    assert all(all(row["construction_checks"].values()) for row in rows)

def test_v24_has_eight_rows_per_direction_and_unique_reporters():
    rows=candidate.build_rows()
    assert len({row["reporter"] for row in rows}) == 16
    for family in ("A1","A2"):
        panel=[row for row in rows if row["family"] == family]
        assert sum(row["direction_id"] == "present_to_past" for row in panel) == 8
        assert sum(row["direction_id"] == "past_to_present" for row in panel) == 8
