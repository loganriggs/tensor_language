import math

import run_aspectual_tense_h1h4_local_v9_input_branch_factorial_v1 as runner


def test_registered_boolean_arm_order():
    assert runner.subsets() == ((), ("deep_resid9",), ("direct_x0_reinjection",), ("deep_resid9", "direct_x0_reinjection"))


def test_mobius_accounting_efficiency_and_retention():
    values = {(): 0.0, ("deep_resid9",): 3.0, ("direct_x0_reinjection",): 1.0, ("deep_resid9", "direct_x0_reinjection"): 5.0}
    result = runner.two_branch_factorial(values)
    assert result["interaction"] == 1.0
    assert result["shapley"] == {"deep_resid9": 3.5, "direct_x0_reinjection": 1.5}
    assert math.isclose(result["deep_retained_fraction"], 0.6)
    assert result["efficiency_error"] == 0.0
