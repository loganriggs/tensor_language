#!/usr/bin/env python3

import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange as v2
import circuit_fast_screen_candidate_attn8_h3_h7_cross_behavior_factor_interchange_v3 as v3


def test_v3_is_frozen_and_only_step_two_preference_changes():
    old_rows, new_rows = v2.build_rows(), v3.build_rows()
    assert v3.validate_rows(new_rows) == v3.EXPECTED_ROWS_SHA256
    assert len(old_rows) == len(new_rows) == 32
    for old, new in zip(old_rows, new_rows):
        old_step, new_step = old["controls"]["step_two"], new["controls"]["step_two"]
        assert new_step["answer_id"] == old_step["preference_foil_id"]
        assert new_step["preference_foil_id"] == old_step["answer_id"]
        assert new_step["answer_text"].strip() == str(
            int(v3.ENC.decode([new_step["ids"][new_step["source_positions"][-1]]]).strip()) + 1)
        for control in ("repeated_list_copy", "digit_copy"):
            assert old["controls"][control] == new["controls"][control]
