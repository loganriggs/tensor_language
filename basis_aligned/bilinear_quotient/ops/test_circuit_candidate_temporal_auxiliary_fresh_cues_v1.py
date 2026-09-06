import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate


def test_fresh_rows_are_deterministic_and_aligned():
    rows = candidate.build_rows()
    assert candidate.validate_rows(rows) == candidate.validate_rows(candidate.build_rows())
    targets = [row for row in rows if row["transform_id"] in {"A1", "A2"}]
    assert len(targets) == 64
    for row in targets:
        differences = [
            index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
            if pair[0] != pair[1]
        ]
        assert len(row["base_ids"]) == len(row["donor_ids"])
        assert len(differences) == 1


def test_fresh_frames_and_cues_are_present():
    rows = candidate.build_rows()
    texts = {row["base_text"] for row in rows if row["transform_id"] in {"A1", "A2"}}
    assert any(text.startswith("Later ") or text.startswith("Previously ") for text in texts)
    assert any(text.startswith("The note reads: ") for text in texts)
