import pytest
import torch

from .node import execute


def test_execute_and_bilinear_composition():
    generator = torch.Generator().manual_seed(1701)
    p1 = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    p2 = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    u1 = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    u2 = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
    support = torch.rand(2, 5, 5, generator=generator) > .4
    composed = execute(p1 + p2, u1 + u2, support)
    expanded = sum(execute(p, u, support) for p in (p1, p2) for u in (u1, u2))
    assert torch.allclose(composed, expanded, atol=1e-12, rtol=1e-12)


def test_shape_failures():
    with pytest.raises(ValueError):
        execute(torch.zeros(2, 3), torch.zeros(2, 3, 4), torch.zeros(2, 3))
