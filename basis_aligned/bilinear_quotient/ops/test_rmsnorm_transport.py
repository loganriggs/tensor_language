import pytest
import torch

import rmsnorm_transport as subject


def test_exact_delta_matches_direct_evaluation_and_split_closes():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    delta = .2 * torch.randn(5, 7, generator=generator, dtype=torch.float64)
    split = subject.transport_split(x, delta, 1e-8)
    direct = subject.rmsnorm_value(x + delta, 1e-8) - subject.rmsnorm_value(x, 1e-8)
    assert torch.allclose(split["exact"], direct, atol=1e-12, rtol=1e-12)
    assert torch.allclose(split["tangent"] + split["remainder"], split["exact"], atol=1e-12, rtol=1e-12)


def test_tangent_matches_centered_finite_difference():
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    direction = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    step = 1e-5
    finite = (subject.rmsnorm_value(x + step*direction, 1e-7)
              - subject.rmsnorm_value(x - step*direction, 1e-7)) / (2*step)
    tangent = subject.tangent_delta(x, direction, 1e-7)
    assert torch.allclose(tangent, finite, atol=2e-9, rtol=2e-9)


def test_radial_direction_is_suppressed_when_epsilon_is_small():
    x = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    tangent = subject.tangent_delta(x, x, 1e-12)
    assert tangent.norm() < 1e-11


@pytest.mark.parametrize("bad_eps", [0, -1])
def test_rejects_nonpositive_epsilon(bad_eps):
    with pytest.raises(subject.RMSNormTransportError):
        subject.exact_delta(torch.ones(2), torch.ones(2), bad_eps)
