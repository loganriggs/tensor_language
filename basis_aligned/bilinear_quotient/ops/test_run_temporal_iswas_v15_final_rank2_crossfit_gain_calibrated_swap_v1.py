import inspect

import run_temporal_iswas_v15_final_rank2_crossfit_gain_calibrated_swap_v1 as runner

def test_price_and_fit_parameters_are_frozen():
    assert runner.PRICE_MAX["differentiable_transformer_forwards"]==14
    assert runner.PRICE_MAX["fit_parameters"]==4

def test_fit_uses_state_norms_not_logits():
    source=inspect.getsource(runner.panel_mean_norm)
    assert ".norm(dim=1)" in source
    assert "logit" not in source and "margin" not in source

def test_canonical_crossfit_and_report_keys_exist():
    source=inspect.getsource(runner.main)
    assert "train=1-held" in source
    assert "A1_from_A2" in source and "A2_from_A1" in source
    assert "absolute_error_improvement" in source
