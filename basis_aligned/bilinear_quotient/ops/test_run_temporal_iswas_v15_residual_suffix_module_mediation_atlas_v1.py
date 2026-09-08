import run_temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1 as runner


def test_frozen_module_scope_and_price_are_exact():
    assert runner.SINGLETONS == tuple(
        f"{kind}{layer}" for layer in range(12, 18) for kind in ("attn", "mlp"))
    assert runner.MEDIATORS == ("joint",) + runner.SINGLETONS
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 120
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0
    assert runner.PRICE_MAX["fit_parameters"] == 0


def test_all_authorities_are_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_finite_rejects_nested_nonfinite_values():
    assert runner.finite({"x": [1.0, 2.0]})
    assert not runner.finite({"x": [float("nan")]})
