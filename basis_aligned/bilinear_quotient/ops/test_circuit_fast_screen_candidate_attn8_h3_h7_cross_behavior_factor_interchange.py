#!/usr/bin/env python3

from collections import Counter

import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange as authority


def test_paired_authority_is_frozen_and_balanced():
    rows = authority.build_rows()
    assert authority.validate_rows(rows) == authority.EXPECTED_ROWS_SHA256
    counts = Counter((row["split"], row["recipient_format"], row["direction"])
                     for row in rows)
    assert len(rows) == 32 and len(counts) == 8 and set(counts.values()) == {4}


def test_formats_have_identical_visible_states_and_next_values():
    rows = authority.build_rows()
    for row in rows:
        recipient_values = (row["visible_base_values"] if row["direction"] == "base_to_donor"
                            else row["visible_donor_values"])
        donor_values = (row["visible_donor_values"] if row["direction"] == "base_to_donor"
                        else row["visible_base_values"])
        assert row["recipient"]["answer_text"].strip() == str(recipient_values[-1]+1)
        assert row["cross_same"]["answer_text"].strip() == str(recipient_values[-1]+1)
        assert row["within_donor"]["answer_text"].strip() == str(donor_values[-1]+1)
        assert row["cross_opposite"]["answer_text"].strip() == str(donor_values[-1]+1)


def test_semantic_positions_and_active_controls_are_explicit():
    for row in authority.build_rows():
        assert set(row["controls"]) == {"repeated_list_copy", "digit_copy", "step_two"}
        for endpoint in (row["recipient"], row["within_donor"], row["cross_same"],
                         row["cross_opposite"], *row["controls"].values()):
            assert len(endpoint["source_positions"]) == 3
            assert endpoint["query_position"] == len(endpoint["ids"])-1
            assert authority.ENC.encode(endpoint["text"] + endpoint["answer_text"]) == \
                endpoint["ids"] + [endpoint["answer_id"]]
        for control in row["controls"].values():
            assert control["answer_id"] != control["preference_foil_id"]
