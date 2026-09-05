#!/usr/bin/env python3

from collections import Counter

import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as authority


def test_fresh_authority_is_frozen_balanced_and_single_token():
    rows = authority.build_rows()
    assert authority.validate_rows(rows) == authority.EXPECTED_ROWS_SHA256
    assert set(Counter(row["cell_id"] for row in rows).values()) == {8}
    assert set(Counter(row["diagnostic_cell_id"] for row in rows).values()) == {4}
    assert all(len(authority.old_task14.ENCODING.encode(" " + word)) == 1
               for pair in authority.NOUN_PAIRS for word in pair)


def test_all_96_surfaces_are_unique_and_donors_change_only_subject():
    rows = authority.build_rows()
    prompts = [row[f"{role}_text"] for row in rows for role in ("base", "same", "opposite")]
    tokens = [tuple(row[f"{role}_ids"]) for row in rows for role in ("base", "same", "opposite")]
    assert len(set(prompts)) == len(set(tokens)) == 96
    assert all([i for i, pair in enumerate(zip(row["same_ids"], row["opposite_ids"]))
                if pair[0] != pair[1]] == [8] for row in rows)
    assert all(row["recipient_template"] != row["donor_template"] and
               row["base_ids"][:8] != row["same_ids"][:8] for row in rows)
