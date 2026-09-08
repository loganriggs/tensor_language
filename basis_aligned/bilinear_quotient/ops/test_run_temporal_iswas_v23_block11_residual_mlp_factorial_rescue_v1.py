import torch

import run_temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1 as experiment


def test_replace_first_preserves_block_auxiliary_output():
    x, v1 = torch.tensor([1.]), torch.tensor([2.])
    changed = experiment.replace_first((x, v1), torch.tensor([3.]))
    assert changed[0].item() == 3
    assert changed[1] is v1


def test_metrics_exact_and_opposed_vectors():
    reference = torch.tensor([1., -2., 3., -4.])
    selected = torch.ones(4, dtype=torch.bool)
    assert experiment.metrics(torch, reference, reference, selected)["relative_residual"] == 0
    assert experiment.metrics(torch, -reference, reference, selected)["direction_fraction"] == 0


def test_seven_forward_factorial_inventory():
    # base, donor, donor-self, removal, residual rescue, MLP rescue, joint rescue
    assert experiment.PRICE["model_forwards_exact"] == 7
    assert experiment.PRICE["sequence_evaluations_exact"] == 7 * 64
    assert experiment.PRICE["fit_parameters"] == 0
