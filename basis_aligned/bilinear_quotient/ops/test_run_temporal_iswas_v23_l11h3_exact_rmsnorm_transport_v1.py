import torch

import run_temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1 as experiment


def test_projected_metrics_exact_and_scaled():
    actual = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) + 1
    selected = torch.tensor([[True, True, False, False], [False, True, True, False]])
    exact = experiment.metrics(torch, actual, actual, selected)
    assert exact["relative_residual"] == 0
    assert abs(exact["cosine"] - 1) < 1e-12
    scaled = experiment.metrics(torch, .5*actual, actual, selected)
    assert abs(scaled["signed_projection"] - .5) < 1e-12
    assert abs(scaled["relative_residual"] - .5) < 1e-12


def test_registered_price_matches_native_self_and_patch_forwards():
    assert experiment.PRICE["model_forwards_exact"] == 4
    assert experiment.PRICE["sequence_evaluations_exact"] == 4 * 64
    assert experiment.PRICE["transformer_backwards"] == 0
