import run_temporal_iswas_v15_attention15_head_module_mediation_atlas_v1 as runner


def test_frozen_mediator_scope_and_price_are_exact():
    assert runner.HEADS["module"] == tuple(range(9))
    assert tuple(name for name in runner.HEADS if name != "module") == tuple(
        f"L15H{head}" for head in range(9))
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 120
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0
    assert runner.PRICE_MAX["fit_parameters"] == 0


def test_all_authorities_are_hash_bound():
    assert set(runner.FILES) == set(runner.EXPECTED)
    assert all(len(value) == 64 for value in runner.EXPECTED.values())


def test_finite_rejects_nested_nonfinite_values():
    assert runner.finite({"x": [1.0, 2.0]})
    assert not runner.finite({"x": [float("inf")]})
