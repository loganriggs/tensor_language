import torch

from . import node


def test_residual_score_execution_and_source_merge():
    generator = torch.Generator().manual_seed(505)
    residual = torch.randn(2, 5, 12, generator=generator, dtype=torch.float64)
    weights = [torch.randn(4, 12, generator=generator, dtype=torch.float64) for _ in range(4)]
    positions = torch.arange(5, dtype=torch.float64)[None, :, None]
    frequencies = torch.tensor([1.0, .1], dtype=torch.float64)[None, None, :]
    cos, sin = (positions * frequencies).cos(), (positions * frequencies).sin()
    score = node.execute(residual, weights, cos, sin)
    assert score.shape == (2, 5, 5)
    assert torch.count_nonzero(score.triu(diagonal=1)) == 0
    first = residual.float() * .3
    second = residual.float() * .7
    correction = residual.float() - first - second
    assert torch.allclose(node.merge_sources((first, second), correction), residual.float(), atol=1e-7, rtol=1e-7)
