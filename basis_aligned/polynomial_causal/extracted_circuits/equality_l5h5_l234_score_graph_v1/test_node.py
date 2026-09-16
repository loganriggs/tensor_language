import torch

from . import node


def test_three_source_executor():
    generator = torch.Generator().manual_seed(2345)
    sources = [torch.randn(2, 5, 12, generator=generator, dtype=torch.float64) for _ in range(3)]
    weights = [torch.randn(4, 12, generator=generator, dtype=torch.float64) for _ in range(4)]
    positions = torch.arange(5, dtype=torch.float64)[None, :, None]
    frequencies = torch.tensor([1.0, .1], dtype=torch.float64)[None, None, :]
    cos, sin = (positions * frequencies).cos(), (positions * frequencies).sin()
    direct = node.execute(*sources, weights, cos, sin)
    merged_residual = sum((source.float() for source in sources), start=torch.zeros_like(sources[0].float()))
    merged = node.execute_residual(merged_residual.to(sources[0].dtype), weights, cos, sin)
    assert torch.allclose(direct, merged, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(direct.triu(diagonal=1)) == 0
