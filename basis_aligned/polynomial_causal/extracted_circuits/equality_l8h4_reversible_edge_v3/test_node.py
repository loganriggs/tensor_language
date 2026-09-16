import torch

from .node import execute, merge, split


def test_bf16_roundtrip():
    generator = torch.Generator().manual_seed(1703)
    full = torch.randn(3, 5, 7, generator=generator).to(torch.bfloat16)
    term = torch.randn(3, 5, 7, generator=generator).to(torch.bfloat16)
    removed, correction = split(full, term)
    assert torch.equal(merge(removed, term, correction), full)


def test_semantic_composition():
    generator = torch.Generator().manual_seed(1704)
    p1 = torch.randn(2, 4, 4, generator=generator)
    p2 = torch.randn(2, 4, 4, generator=generator)
    v1 = torch.randn(2, 4, 3, generator=generator)
    v2 = torch.randn(2, 4, 3, generator=generator)
    weight = torch.randn(6, 3, generator=generator)
    support = torch.rand(2, 4, 4, generator=generator) > .3
    composed = execute(p1 + p2, v1 + v2, support, weight)
    expanded = sum(execute(p, v, support, weight) for p in (p1, p2) for v in (v1, v2))
    assert torch.allclose(composed, expanded, atol=2e-5, rtol=2e-5)
