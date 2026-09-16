import torch

from .node import compose_scores, dot_score, execute


def test_execution_and_branch_composition():
    generator = torch.Generator().manual_seed(1705)
    ports = [torch.randn(2, 5, 8, generator=generator, dtype=torch.float64) for _ in range(4)]
    native = execute(*ports)
    assert torch.count_nonzero(native.triu(diagonal=1)) == 0
    q1, k1, q2, k2 = ports
    score1_parts = [dot_score(q1[..., :4], k1[..., :4]) * .5, dot_score(q1[..., 4:], k1[..., 4:]) * .5]
    score2_parts = [dot_score(q2[..., :4], k2[..., :4]) * .5, dot_score(q2[..., 4:], k2[..., 4:]) * .5]
    assert torch.allclose(native, compose_scores(score1_parts, score2_parts), atol=1e-12, rtol=1e-12)
