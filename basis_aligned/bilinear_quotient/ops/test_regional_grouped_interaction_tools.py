import torch

from regional_grouped_interaction_tools import (
    apply_rotary,
    ordered_bilinear_scores,
    paired_difference,
    split_named_sources,
)


def test_shape_branches_pair_axis_and_bilinear_closure():
    generator=torch.Generator().manual_seed(19)
    full=torch.randn(2,7,9,8,generator=generator); grouped=torch.randn(2,7,8,generator=generator)
    phase=torch.randn(1,7,1,4,generator=generator); cos=phase.cos(); sin=phase.sin()
    assert apply_rotary(full,cos,sin).shape==(2,7,9,8)
    assert apply_rotary(grouped,cos,sin).shape==(2,7,8)

    arms=torch.arange(6*48*2).reshape(6,48,2)
    assert paired_difference(arms,1).shape==(6,24,2)
    rows=torch.arange(48)
    assert paired_difference(rows,0).shape==(24,)

    sources=[torch.randn(2,7,8,generator=generator) for _ in range(4)]
    selected,remainder=split_named_sources(("a","b","c","d"),sources,("b","d"))
    assert torch.allclose(selected+remainder,sum(sources),rtol=1e-6,atol=1e-6)
    q=[selected,remainder]; k=[2*selected,3*remainder]
    terms=ordered_bilinear_scores(q,k,8)
    total=torch.einsum("bqd,bkd->bqk",sum(q),sum(k))/8
    assert torch.allclose(terms.sum(0),total,rtol=1e-6,atol=1e-6)
