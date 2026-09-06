import run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1 as runner


def test_registered_factorial_shape():
    assert len(runner.subsets()) == 8
    assert runner.subsets()[0] == ()
    assert runner.subsets()[-1] == runner.FACTORS


def test_shapley_efficiency_for_additive_factorial():
    weights = dict(zip(runner.FACTORS, (0.3, 1.2, -0.2)))
    values = {subset: sum(weights[name] for name in subset) for subset in runner.subsets()}
    shapley = runner.factorial_shapley(values)
    assert all(abs(shapley[name] - weights[name]) < 1e-12 for name in runner.FACTORS)
    assert abs(sum(shapley.values()) - values[runner.FACTORS]) < 1e-12
