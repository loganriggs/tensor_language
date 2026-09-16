import pytest
import torch

from .node import execute


def test_execute_and_bilinear_composition():
    generator = torch.Generator().manual_seed(1702)
    p1 = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    p2 = torch.randn(2, 5, 5, generator=generator, dtype=torch.float64)
    v1 = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
    v2 = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
    weight = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    support = torch.rand(2, 5, 5, generator=generator) > .4
    composed = execute(p1 + p2, v1 + v2, support, weight)
    expanded = sum(execute(p, v, support, weight) for p in (p1, p2) for v in (v1, v2))
    assert torch.allclose(composed, expanded, atol=1e-12, rtol=1e-12)


def test_shape_failures():
    with pytest.raises(ValueError):
        execute(torch.zeros(2, 3), torch.zeros(2, 3, 4), torch.zeros(2, 3), torch.zeros(7, 4))
