import pytest
import torch

import path_conditioned_component_score as subject


def test_directional_effect_matches_central_finite_difference():
    g = torch.Generator().manual_seed(31)
    base = torch.randn(2, 4, 5, generator=g)
    donor = torch.randn(2, 4, 5, generator=g)
    weight = torch.randn(2, 4, 5, generator=g)
    banks = ((1, 2), (0, 3))
    gradient = weight.clone()
    exact = subject.directional_effects(gradient, base, donor, banks)
    direction = donor - base
    epsilon = 1e-3
    finite = []
    for row, bank in enumerate(banks):
        mask = torch.zeros_like(base[row])
        mask[list(bank)] = direction[row, list(bank)]
        objective = lambda alpha: ((base[row] + alpha * mask) * weight[row]).sum()
        finite.append((objective(epsilon) - objective(-epsilon)) / (2 * epsilon))
    assert torch.allclose(exact, torch.stack(finite), atol=2e-3)


def test_directional_effect_rejects_bad_banks():
    value = torch.zeros(1, 2, 3)
    with pytest.raises(subject.PathConditionedScoreError):
        subject.directional_effects(value, value, value, ((1, 1),))
