import run_aspectual_tense_h1h4_deep_resid9_block8_factorial_v1 as runner


def test_registered_subset_order_and_count():
    assert len(runner.subsets()) == 8
    assert runner.subsets()[0] == ()
    assert runner.subsets()[-1] == runner.BRANCHES


def test_three_factor_shapley_efficiency_for_additive_function():
    weights = dict(zip(runner.BRANCHES, (3.0, -1.0, 2.0)))
    values = {subset: 0.5 + sum(weights[name] for name in subset) for subset in runner.subsets()}
    result = runner.factorial_accounting(values)
    assert result["shapley"] == weights
    assert result["efficiency_error"] == 0.0
