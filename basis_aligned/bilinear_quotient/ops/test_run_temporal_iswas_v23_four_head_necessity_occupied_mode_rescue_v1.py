import torch

import run_temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1 as experiment


def test_metrics_exact_and_opposed_vectors():
    reference = torch.tensor([1., -2., 3., -4.])
    selected = torch.tensor([True, True, True, True])
    exact = experiment.metrics(torch, reference, reference, selected)
    assert exact["relative_residual"] == 0
    assert exact["direction_fraction"] == 1
    opposed = experiment.metrics(torch, -reference, reference, selected)
    assert abs(opposed["cosine"] + 1) < 1e-12
    assert opposed["direction_fraction"] == 0


def test_seven_forward_intervention_inventory():
    # native base/donor, donor self, head removal, full rescue, occupied rescue, mode removal
    assert experiment.PRICE["model_forwards_exact"] == 7
    assert experiment.PRICE["sequence_evaluations_exact"] == 7 * 64
    assert experiment.PRICE["transformer_backwards"] == 0
