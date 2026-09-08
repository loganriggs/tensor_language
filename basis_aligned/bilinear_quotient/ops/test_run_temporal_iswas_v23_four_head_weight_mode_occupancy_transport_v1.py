import torch

import run_temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1 as experiment


def test_vector_metrics_exact_and_scaled_cases():
    actual = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1
    selected = torch.tensor([[True, True, False], [False, True, True]])
    exact = experiment.vector_metrics(torch, actual, actual, selected)
    assert abs(exact["cosine"] - 1) < 1e-12
    assert abs(exact["signed_projection"] - 1) < 1e-12
    assert exact["relative_residual"] == 0
    scaled = experiment.vector_metrics(torch, 2*actual, actual, selected)
    assert abs(scaled["cosine"] - 1) < 1e-12
    assert abs(scaled["signed_projection"] - 2) < 1e-12
    assert abs(scaled["relative_residual"] - 1) < 1e-12


def test_seven_forward_price_inventory():
    assert experiment.PRICE["model_forwards_exact"] == 2 + 1 + len(experiment.ROUTES)
    assert experiment.PRICE["sequence_evaluations_exact"] == 64 * 7
    assert experiment.PRICE["transformer_backwards"] == 0
