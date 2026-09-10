"""Symmetric conditional routing/value contraction on frozen native factors."""
import torch
from symmetric_factor_interaction_v1 import projectors


def partition(pattern, value, projections):
    """pattern[world,corner,head,source], value[...,source,channel]."""
    p=torch.einsum('kij,wjhs->kwihs',projections,pattern)
    v=torch.einsum('kij,wjhsd->kwihsd',projections,value)
    edges={'routing':p[3][...,None]*v[0], 'value':p[0][...,None]*v[3],
           'cross':p[1][...,None]*v[2]+p[2][...,None]*v[1]}
    full=torch.einsum('ij,wjhd->wihd',projections[3],torch.einsum('wihs,wihsd->wihd',pattern,value))
    return {k:v.sum(-2) for k,v in edges.items()},edges,full


def controls():
    from itertools import product
    c=list(product((-1,1),repeat=5));q=projectors(c)
    gen=torch.Generator().manual_seed(910556)
    p=torch.randn(2,32,3,4,generator=gen,dtype=torch.float64)
    v=torch.randn(2,32,3,4,5,generator=gen,dtype=torch.float64)
    branches,_,full=partition(p,v,q)
    closure=float((sum(branches.values())-full).abs().max());assert closure<1e-12
    o=torch.tensor([x[2] for x in c],dtype=torch.float64)[None,:,None,None]
    h=torch.tensor([x[4] for x in c],dtype=torch.float64)[None,:,None,None]
    for name,a,b in [('routing',o*h,torch.ones_like(o)[...,None]),
                     ('value',torch.ones_like(o),(o*h)[...,None]),('cross',o,h[...,None])]:
        parts,_,f=partition(a,b,q)
        assert torch.equal(parts[name],f)
        assert all(x.count_nonzero()==0 for k,x in parts.items() if k!=name)
    scaled,_,_=partition(2*p,v/2,q)
    gauge=max(float((scaled[k]-branches[k]).abs().max()) for k in branches);assert gauge<1e-12
    return {'passed':True,'closure_max_abs':closure,'reciprocal_factor_scale_error':gauge,
            'routing_value_cross_planted_controls':True,'model_forwards':0}
