import run_temporal_iswas_v15_entry12_crossfit_gram_router_v1 as runner


def test_frozen_gram_router_scope_and_price_are_exact():
    assert runner.finite_router.CLASSES == ("A1", "A2", "off")
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 24
    assert runner.PRICE_MAX["fit_parameters"] == 30


def test_all_authorities_are_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_finite_rejects_nested_nonfinite_values():
    assert runner.finite({"x": [1.0]})
    assert not runner.finite({"x": [float("nan")]})
