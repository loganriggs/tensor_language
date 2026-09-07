import torch
import hidden_unit_subspace_contribution as h


def test_exact_compiled_delta_and_ranking():
    g = torch.Generator().manual_seed(7)
    delta = torch.randn(11, 20, generator=g, dtype=torch.float64)
    down = torch.randn(13, 20, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(13, 4, generator=g, dtype=torch.float64))
    a = down.T @ q
    assert torch.allclose(h.compiled_delta(delta, a, q), (delta @ down.T @ q) @ q.T)
    scores = h.contribution_energy(delta, a)
    mask = h.top_fraction_mask(scores, .25)
    assert mask.sum() == 5
    assert scores[mask].min() >= scores[~mask].max()


def test_bad_shapes_and_fraction_fail_closed():
    import pytest
    with pytest.raises(ValueError): h.contribution_energy(torch.ones(2, 3, 4), torch.ones(4, 2))
    with pytest.raises(ValueError): h.contribution_energy(torch.ones(2, 3), torch.ones(4, 2))
    with pytest.raises(ValueError): h.top_fraction_mask(torch.ones(3), 0)
