import circuit_candidate_temporal_iswas_dual_command_v1 as candidate


def test_exact_balanced_inventory_and_hash():
    rows = candidate.build_rows()
    assert len(rows) == 32
    assert candidate.validate_rows(rows) == candidate.EXPECTED_AUTHORITY_SHA256
    assert {row["phase"] for row in rows} == {"FIT", "HOLDOUT"}


def test_every_row_has_exact_joint_factorial_and_two_live_positions():
    for row in candidate.build_rows():
        assert set(row["endpoints"]) == set(candidate.CELLS)
        for endpoint in row["endpoints"].values():
            assert endpoint["ids"][endpoint["temporal_position"]] == endpoint["temporal_answer_id"]
            assert endpoint["ids"][endpoint["iswas_position"]] == endpoint["iswas_answer_id"]
            assert endpoint["temporal_position"] < endpoint["iswas_position"]


def test_each_factor_changes_only_its_registered_answers():
    for row in candidate.build_rows():
        endpoints = row["endpoints"]
        assert endpoints["00"]["temporal_answer_id"] == endpoints["01"]["temporal_answer_id"]
        assert endpoints["10"]["temporal_answer_id"] == endpoints["11"]["temporal_answer_id"]
        assert endpoints["00"]["iswas_answer_id"] == endpoints["10"]["iswas_answer_id"]
        assert endpoints["01"]["iswas_answer_id"] == endpoints["11"]["iswas_answer_id"]
