import torch

from . import node


def test_canonical_five_write_merge():
    generator = torch.Generator().manual_seed(23454)
    writes = [torch.randn(2, 5, 12, generator=generator, dtype=torch.float64) for _ in range(5)]
    weights = [torch.randn(4, 12, generator=generator, dtype=torch.float64) for _ in range(4)]
    positions = torch.arange(5, dtype=torch.float64)[None, :, None]
    frequencies = torch.tensor([1.0, .1], dtype=torch.float64)[None, None, :]
    cos, sin = (positions * frequencies).cos(), (positions * frequencies).sin()
    direct = node.execute(*writes, weights, cos, sin)
    m2, a3, m3, a4, m4 = writes
    residual = torch.zeros_like(m2.float()) + m2.float() + (a3.float() + m3.float()) + (a4.float() + m4.float())
    merged = node.execute_residual(residual.to(m2.dtype), weights, cos, sin)
    assert torch.allclose(direct, merged, atol=1e-12, rtol=1e-12)
    assert torch.count_nonzero(direct.triu(diagonal=1)) == 0
