import torch

from run_setting2_regional_four_source_term_census_v1 import pair_terms


def test_pair_terms_reconstruct_bilinear_reader():
    torch.manual_seed(17)
    n, d, m = 7, 9, 13
    sources = {name: torch.randn(n, d, dtype=torch.float64) for name in "epoa"}
    left = torch.randn(m, d, dtype=torch.float64)
    right = torch.randn(m, d, dtype=torch.float64)
    folded_down = torch.randn(n, m, dtype=torch.float64)
    terms = pair_terms(sources, left, right, folded_down)
    x = sum(sources.values())
    direct = ((x @ left.T) * (x @ right.T) * folded_down).sum(-1)
    assert len(terms) == 10
    assert torch.allclose(sum(terms.values()), direct, rtol=1e-11, atol=1e-11)
