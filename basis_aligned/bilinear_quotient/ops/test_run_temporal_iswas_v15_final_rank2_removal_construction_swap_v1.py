import inspect

import run_temporal_iswas_v15_final_rank2_removal_construction_swap_v1 as runner


def test_price_is_exactly_reference_plus_state_captures():
    assert runner.PRICE_MAX["differentiable_transformer_forwards"] == 12
    assert runner.PRICE_MAX["transformer_backward_forwards"] == 0


def test_final_manipulations_use_exact_head_and_whole_payload_swap():
    source = inspect.getsource(runner.main)
    assert "manipulation.removal_and_sufficiency" in source
    assert "manipulation.paired_payload_swap" in source
    assert "decode(backend" in source
    assert '"live_suffix"' in source and '"frozen_suffix"' in source


def test_canonical_report_keys_are_asserted():
    source = inspect.getsource(runner.main)
    for key in ("removal_loss", "sufficiency", "construction_swap"):
        assert key in source
