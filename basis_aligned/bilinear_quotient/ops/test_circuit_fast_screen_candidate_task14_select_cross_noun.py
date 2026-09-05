#!/usr/bin/env python3
# BQLANE: cpu

import circuit_fast_screen_candidate_task14_select_cross_noun as candidate
import pytest


def test_rows_are_exact_balanced_cross_noun_relations() -> None:
    rows = candidate.build_rows()
    assert len(rows) == 64
    assert all(row["target_group_id"] != row["donor_group_id"] for row in rows)
    assert all(row["base_head_pair"] != row["donor_head_pair"] for row in rows)
    assert all(row["base_attractor_plural"] == row["donor_attractor_plural"] for row in rows)
    assert set(sum(row["cell_id"] == cell for row in rows)
               for cell in {row["cell_id"] for row in rows}) == {16}


def test_pairing_is_a_derangement_within_each_conditioning_stratum() -> None:
    rows = candidate.build_rows()[::2]
    targets = {row["target_group_id"] for row in rows}
    donors = {row["donor_group_id"] for row in rows}
    assert targets == donors
    assert all(row["base_subject_number"] != row["donor_subject_number"] for row in rows)


def test_self_consistent_but_noncanonical_row_is_rejected() -> None:
    rows = candidate.build_rows()
    rows[0] = dict(rows[0], donor_text=rows[0]["donor_text"] + " altered")
    with pytest.raises(candidate.SelectCrossNounAuthorityError, match="exact regenerated"):
        candidate.validate_rows(rows)


def test_plan_reuses_the_eight_forward_targeted_runner_shape() -> None:
    plan = candidate.compile_plan()
    assert plan["price"] == {
        "forward_calls": 8, "example_evaluations": 256,
        "backward_calls": 0, "model_updates": 0,
        "raw_numeric_evidence_bytes": 2048,
    }
