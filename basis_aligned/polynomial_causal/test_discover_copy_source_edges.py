from __future__ import annotations

import torch

from discover_copy_source_edges import nearest_repeat_policy


def test_nearest_repeat_policy_uses_only_input_for_source_and_target_for_cell():
    # At p=3 the nearest earlier 7 is p=1, so the predicted successor is input p=2
    # (token 8). The first row's target is 8 and the second row's target is 9.
    rows = torch.tensor([
        [3, 7, 8, 7, 8, 0],
        [3, 7, 8, 7, 9, 0],
    ])
    policy = nearest_repeat_policy(rows, window=4, score_start=0)
    assert policy["source"][:, 3].tolist() == [1, 1]
    assert policy["successor"][:, 3].tolist() == [2, 2]
    assert policy["eligible"][:, 3].tolist() == [True, True]
    assert policy["copy_positive"][:, 3].tolist() == [True, False]
    assert policy["repeat_negative"][:, 3].tolist() == [False, True]


def test_nearest_repeat_policy_prefers_nearest_and_partitions_scored_positions():
    rows = torch.tensor([[4, 5, 4, 4, 6, 7, 8]])
    policy = nearest_repeat_policy(rows, window=5, score_start=2)
    assert int(policy["source"][0, 3]) == 2
    scored = policy["all_scored"]
    partition = (
        policy["copy_positive"]
        | policy["repeat_negative"]
        | policy["nonrepeat"]
    )
    assert torch.equal(scored, partition)
    assert not bool(
        (policy["copy_positive"] & policy["repeat_negative"]).any()
        or (policy["copy_positive"] & policy["nonrepeat"]).any()
        or (policy["repeat_negative"] & policy["nonrepeat"]).any()
    )
