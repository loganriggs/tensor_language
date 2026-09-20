import torch
from quadratic_source_census import carried_sources,pair_table,read_quadratic,greedy_boundary_order


def test_residual_carriage_includes_each_source_once():
    torch.manual_seed(625)
    lambdas=[(.7,.2),(-.4,.1),(1.3,-.2)]
    writes={name:torch.randn(5,dtype=torch.float64) for name in
            ['embedding','attn0','mlp0','attn1','mlp1','attn2']}
    state=writes['embedding']
    for l,(a,b) in enumerate(lambdas):
        state=a*state+b*writes['embedding']+writes[f'attn{l}']
        if l<2:state=state+writes[f'mlp{l}']
    weights=carried_sources(lambdas)
    torch.testing.assert_close(state,sum(writes[k]*v for k,v in weights.items()))


def test_pair_sum_and_fixed_norm_leaveout_are_exact():
    torch.manual_seed(626)
    sources=torch.randn(11,4,5,dtype=torch.float64)
    readers=torch.randn(5,3,dtype=sources.dtype);coeff=torch.tensor([1.,-2.,3.],dtype=sources.dtype)
    h=sources.sum(-2);den=h.square().mean(-1)+torch.finfo(h.dtype).eps
    table=pair_table(sources,readers,coeff,den)
    base=read_quadratic(h,readers,coeff,denominator=den)
    torch.testing.assert_close(table.sum((-1,-2)),base)
    for i in range(4):
        observed=base-read_quadratic(h-sources[:,i],readers,coeff,denominator=den)
        torch.testing.assert_close(observed,2*table[:,i,:].sum(-1)-table[:,i,i])


def test_norm_closed_greedy_matches_explicit_subset_search_each_step():
    torch.manual_seed(626)
    src=torch.randn(13,5,4,dtype=torch.float64)
    readers=torch.randn(4,3,dtype=src.dtype);c=torch.tensor([1.,-2.,3.],dtype=src.dtype)
    numerator=pair_table(src,readers,c,torch.ones(13,dtype=src.dtype))
    gram=src@src.transpose(-1,-2)/4
    eps=torch.full((13,),torch.finfo(src.dtype).eps,dtype=src.dtype)
    truth=read_quadratic(src.sum(1),readers,c)
    order=greedy_boundary_order(numerator,gram,eps,truth)
    retained=set(range(5))
    for i in order[:-1].tolist():
        def loss(j):
            h=src[:,sorted(retained-{j})].sum(1)
            return float((read_quadratic(h,readers,c)-truth).square().sum())
        assert i==min(retained,key=loss)
        retained.remove(i)
