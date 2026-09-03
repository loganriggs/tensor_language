import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from causal_shortcut_controls import (  # noqa: E402
    centered_token_readout_span,
    deflate_and_orthonormalize,
    pooled_endpoint_tangent_span,
    subspace_overlap_fraction,
)


def test_four_token_readout_has_three_dimensional_contrast_span():
    weight = torch.zeros(10, 6)
    weight[1, 0] = weight[2, 1] = weight[3, 2] = weight[4, 3] = 1
    span = centered_token_readout_span(weight, [1, 2, 3, 4])
    assert span.shape == (6, 3)
    assert torch.allclose(span.T @ span, torch.eye(3), atol=1e-6)


def test_overlap_and_deflation_reject_planted_output_shortcut():
    shortcut = torch.eye(8)[:, :3]
    candidate = torch.stack((shortcut[:, 0], torch.eye(8)[:, 5]), dim=1)
    assert abs(subspace_overlap_fraction(candidate, shortcut) - .5) < 1e-6
    residual = deflate_and_orthonormalize(candidate, shortcut)
    assert residual.shape == (8, 1)
    assert subspace_overlap_fraction(residual, shortcut) < 1e-10


def test_pooled_tangent_span_is_basis_invariant():
    gradients = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [1., 1., 0., 0.]])
    mixed = torch.tensor([[2., 1., 0.], [1., 3., 0.], [0., 0., 1.]]) @ gradients
    left = pooled_endpoint_tangent_span(gradients)
    right = pooled_endpoint_tangent_span(mixed)
    assert torch.allclose(left @ left.T, right @ right.T, atol=1e-6)
