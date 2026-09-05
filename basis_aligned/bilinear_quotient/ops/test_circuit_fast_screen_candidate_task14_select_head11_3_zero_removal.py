#!/usr/bin/env python3
# BQLANE: cpu

import pytest

import circuit_fast_screen_candidate_task14_select_head11_3_zero_removal as candidate


def test_rows_cover_all_select_families_and_semantic_cells() -> None:
    rows = candidate.build_rows()
    assert len(rows) == 128
    assert {family: sum(row["family"] == family for row in rows)
            for family in candidate.MIN_NATIVE_ACCURACY} == {
                "A1": 32, "A2": 32, "P": 32, "C": 32,
            }
    cells = {cell: sum(row["cell_id"] == cell for row in rows)
             for cell in {row["cell_id"] for row in rows}}
    assert len(cells) == 14
    assert set(cells.values()) == {8, 16}


def test_plan_is_only_four_conditions_and_sixteen_forwards() -> None:
    plan = candidate.compile_plan()
    assert plan["conditions"] == [
        "native_capture", "zero_head11_3", "zero_attention11", "native_head_replay",
    ]
    assert len(plan["calls"]) == 16
    assert plan["price"]["forward_calls"] == 16
    assert plan["price"]["example_evaluations"] == 512


def test_mutated_row_is_rejected_against_exact_regeneration() -> None:
    rows = candidate.build_rows()
    rows[0] = dict(rows[0], text=rows[0]["text"] + " altered")
    with pytest.raises(candidate.SelectRemovalAuthorityError, match="exact regenerated"):
        candidate.validate_rows(rows)
