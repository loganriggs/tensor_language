import run_temporal_auxiliary_will_had_l9h1h4_source_groups_v1 as runner


def test_authority_and_arm_inventory():
    rows, parent = runner.validate_static()
    assert len(rows) == 64
    assert parent["terminal"] == "screen"
    assert runner.ARMS == (
        "full_pair",
        "all_sources",
        "prefix",
        "cue",
        "subject_onset",
        "intervening_suffix",
        "self",
    )


def test_each_row_has_complete_partition():
    rows, _parent = runner.validate_static()
    for row in rows:
        groups = runner.source.aligned_source_partition(
            row["base_ids"], row["donor_ids"], row["base_semantic_position"]
        )
        flattened = [
            position
            for name in runner.source.GROUP_ORDER
            for position in groups[name]
        ]
        assert sorted(flattened) == list(range(row["base_semantic_position"] + 1))
