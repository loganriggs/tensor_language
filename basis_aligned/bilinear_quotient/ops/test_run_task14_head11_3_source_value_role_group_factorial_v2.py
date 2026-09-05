#!/usr/bin/env python3

import run_task14_head11_3_source_value_role_group_factorial as v1
import run_task14_head11_3_source_value_role_group_factorial_v2 as runner


def test_v2_is_create_only_and_changes_only_empty_replay_tolerance():
    old = v1.compile_plan()
    new = runner.compile_plan()
    assert runner.OUT != v1.OUT
    assert new["candidate_id"].endswith("numerical_repair_v2")
    assert old["bars"]["maximum_empty_subset_absolute_logit_error"] == 5e-5
    assert new["bars"]["maximum_empty_subset_absolute_logit_error"] == 7e-5
    old_bars = dict(old["bars"])
    new_bars = dict(new["bars"])
    old_bars.pop("maximum_empty_subset_absolute_logit_error")
    new_bars.pop("maximum_empty_subset_absolute_logit_error")
    assert new_bars == old_bars
    assert new["groups"] == old["groups"]
    assert new["conditions"] == old["conditions"]
    assert new["price"] == old["price"]
