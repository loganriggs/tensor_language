import math

import run_aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1 as runner


def test_registered_boolean_arm_order():
    assert runner.subsets() == (
        (),
        ("local_l9_value_change",),
        ("carried_l0_v1_change",),
        ("local_l9_value_change", "carried_l0_v1_change"),
    )


def test_two_branch_mobius_shapley_efficiency_and_retention():
    values = {
        (): 1.0,
        ("local_l9_value_change",): 4.0,
        ("carried_l0_v1_change",): 0.0,
        ("local_l9_value_change", "carried_l0_v1_change"): 5.0,
    }
    result = runner.two_branch_factorial(values)
    assert result["interaction"] == 2.0
    assert result["shapley"] == {
        "local_l9_value_change": 4.0,
        "carried_l0_v1_change": 0.0,
    }
    assert math.isclose(result["local_retained_fraction"], 0.8)
    assert result["efficiency_error"] == 0.0
