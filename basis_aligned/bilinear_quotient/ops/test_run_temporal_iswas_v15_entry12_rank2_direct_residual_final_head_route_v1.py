import inspect

import run_temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1 as runner


def test_price_counts_all_reference_capture_replay_and_cells():
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 24
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0


def test_route_freezes_all_twelve_suffix_writes():
    source = inspect.getsource(runner.module_sites)
    assert "range(12, 18)" in source
    assert 'sites[f"attn{layer}"]' in source and 'sites[f"mlp{layer}"]' in source


def test_direct_law_and_exact_head_are_explicit():
    source = inspect.getsource(runner.main)
    assert "range(12, 18)" in source and ".lambdas[0]" in source and ".prod()" in source
    assert "exact_head(backend, output" in source
    assert 'cells = {"00": off, "01": rescue, "10": reset, "11": on}' in source
