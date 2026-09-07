import math

import run_temporal_iswas_v15_head_factor_dual_greedy_v1 as greedy


def report(a1, direction, p_flips, c_flips, p_kl, c_kl):
    return {
        "targets": {"A1": {"behavior": {
            "signed_projection": a1, "direction_fraction": direction,
        }}},
        "controls": {
            "P": {"top1_flip_count": p_flips, "median_kl": p_kl},
            "C": {"top1_flip_count": c_flips, "median_kl": c_kl},
        },
    }


def test_component_grid_is_exact_four_by_three():
    assert len(greedy.COMPONENTS) == len(set(greedy.COMPONENTS)) == 12
    assert {item[:2] for item in greedy.COMPONENTS} == set(greedy.HEADS)
    assert all(sum(item[:2] == head for item in greedy.COMPONENTS) == 3 for head in greedy.HEADS)


def test_eligibility_distance_is_zero_exactly_inside_bars():
    assert greedy.eligibility_distance(report(.75, .875, 0, 0, .02, .02)) == 0.0
    observed = greedy.eligibility_distance(report(.60, .70, 2, 1, .05, .01))
    expected = .15/.75 + .175/.875 + 3 + 10 * .03
    assert math.isclose(observed, expected)


def test_price_cap_covers_two_full_paths_and_deletions():
    assert greedy.MAX_FORWARDS == 8 + 1 + 1 + 2 * sum(range(1, 13)) + 12
