"""Exact source-pair accounting at a bilinear reader's input boundary.

Sources are observed native writes carried through residual lambdas. Holding
these writes fixed is a boundary decomposition, not an upstream causal edit.
"""
import torch


def carried_sources(lambdas):
    weights={'embedding':1.}
    for layer,(residual,embedding) in enumerate(lambdas):
        weights={name:value*residual for name,value in weights.items()}
        weights['embedding']+=embedding
        weights[f'attn{layer}']=1.
        if layer<len(lambdas)-1:weights[f'mlp{layer}']=1.
    return weights


def pair_table(sources, readers, coefficients, denominator):
    """sources [...,sources,width], denominator [...] = mean(h²)+native_eps."""
    projections=sources@readers
    return torch.einsum('...sk,k,...tk->...st',projections,coefficients,projections)/denominator[...,None,None]


def read_quadratic(h, readers, coefficients, *, denominator=None):
    if denominator is None:
        denominator=h.square().mean(-1)+torch.finfo(h.dtype).eps
    return ((h@readers).square()*coefficients).sum(-1)/denominator


def greedy_boundary_order(numerator, norm_gram, epsilon, target):
    """Greedy backward selection using true norm-closed prediction error.

    Both matrices are [examples,sources,sources]; their common per-example
    scaling can be arbitrary. epsilon is in that same denominator scaling.
    Returns removal order, so order[-k:] are the k retained sources. This is
    calibration-based support discovery, not a global optimum or causal test.
    """
    numerator,norm_gram,epsilon,target=[t.double() for t in (numerator,norm_gram,epsilon,target)]
    n=target.shape[0];s=numerator.shape[-1]
    nr=numerator.sum(-1);dr=norm_gram.sum(-1)
    current_n=nr.sum(-1);current_d=dr.sum(-1)+epsilon
    nd=numerator.diagonal(dim1=-2,dim2=-1);dd=norm_gram.diagonal(dim1=-2,dim2=-1)
    removed=torch.zeros(s,dtype=torch.bool,device=target.device);order=[]
    for _ in range(s-1):
        trial_n=current_n[:,None]-2*nr+nd
        trial_d=current_d[:,None]-2*dr+dd
        prediction=trial_n/trial_d.clamp_min(epsilon[:,None])
        errors=(prediction-target[:,None]).square().sum(0)
        i=int(errors.masked_fill(removed,torch.inf).argmin());order.append(i);removed[i]=True
        current_n=trial_n[:,i];current_d=trial_d[:,i]
        nr=nr-numerator[:,:,i];dr=dr-norm_gram[:,:,i]
    order.append(int((~removed).nonzero().flatten()[0]))
    return torch.tensor(order,device=target.device)
