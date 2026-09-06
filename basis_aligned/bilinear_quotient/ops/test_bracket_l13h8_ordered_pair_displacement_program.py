import circuit_fast_screen_candidate_bracket_l13h8_ordered_pair_displacement_program as authority


def test_export_rows_are_balanced_and_ood_is_closed():
    rows = authority.build_export_rows()
    plan = authority.compile_export_plan()
    assert len(rows) == 72
    assert plan["endpoints"] == 144
    assert plan["stored_scalars"] == 6912
    assert plan["ood_rows_consumed"] == 0
    assert plan["ood_outcomes_consumed"] == 0


def test_ood_authority_is_complete_and_disjoint():
    export = authority.build_export_rows()
    ood = authority.build_ood_rows()
    assert len(ood) == 180
    assert not ({row["row_id"] for row in export} & {row["row_id"] for row in ood})
    assert sum(row["program_role"] == "target" for row in ood) == 72
    assert sum(row["program_role"] == "control" for row in ood) == 108
