import torch

import run_temporal_iswas_v23_l8h1_layerwise_transport_localization_v1 as experiment


def test_metrics_exact_response():
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 1
    selected = torch.tensor([[True, False, True], [False, True, True]])
    report = experiment.metrics(torch, value, value, selected)
    assert report["relative_residual"] == 0
    assert abs(report["cosine"] - 1) < 1e-12
    assert abs(report["signed_projection"] - 1) < 1e-12


def test_boundary_inventory_and_price_are_frozen():
    assert experiment.BOUNDARIES[:2] == (
        "block9_input_after_block8", "block10_input_after_block9")
    assert experiment.BOUNDARIES[-1] == "M11_normalized_input"
    assert experiment.PRICE["model_forwards_exact"] == 4
    assert experiment.PRICE["sequence_evaluations_exact"] == 4 * 64
