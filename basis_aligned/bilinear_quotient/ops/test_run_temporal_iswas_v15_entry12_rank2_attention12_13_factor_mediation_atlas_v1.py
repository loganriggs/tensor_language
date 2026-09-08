import inspect

import run_temporal_iswas_v15_entry12_rank2_attention12_13_factor_mediation_atlas_v1 as runner


def test_exact_mediator_inventory():
    inventory = runner.mediators()
    assert len(inventory) == 35
    assert len(inventory["module12"]) == 45
    assert len(inventory["L12H4"]) == 5
    assert inventory["L12H4:q"] == ("L12H4:q",)


def test_price_counts_reference_state_capture_and_factorials():
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 156
    assert len(runner.mediators()) * 2 * 2 + 16 == 156


def test_gold_routing_and_factorial_are_explicit():
    source = inspect.getsource(runner.main)
    assert "finite_router.routed_absolute" in source
    assert 'cells = {"00": off_factor, "01": rescue, "10": reset, "11": on_factor}' in source
    assert "scorer.singleton_module_composition" in source


def test_nominated_factors_are_frozen_from_weight_atlas():
    assert runner.NOMINATED_FACTORS == ("L12H0:v", "L12H4:q", "L12H4:k", "L13H2:q")
