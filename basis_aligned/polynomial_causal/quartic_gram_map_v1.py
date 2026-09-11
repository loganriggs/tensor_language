"""Normalized Sym2 Gram matrices to Sym4 coefficients and their isometric adjoint.
Different Gram matrices can encode the same diagonal quartic polynomial.
"""
import torch
from sparse_quartic_core_v1 import indices


def layout(rank,device=None,dtype=torch.float64):
    pairs=torch.triu_indices(rank,rank,device=device)
    pm=torch.where(pairs[0]==pairs[1],1.,2.).to(dtype)
    terms,m=indices(rank,device);lookup={tuple(t):i for i,t in enumerate(terms.T.tolist())}
    target=torch.tensor([lookup[tuple(sorted(a+b))] for a in pairs.T.tolist() for b in pairs.T.tolist()],device=device)
    weights=(pm[:,None]*pm[None,:]).sqrt().flatten()/m[target]
    return dict(pairs=pairs,terms=terms,multiplicity_root=m.to(dtype),target=target,weights=weights,count=terms.shape[1])


def coefficients(matrix,mapping):
    flat=matrix.flatten(-2);idx=mapping['target'].expand(flat.shape)
    result=torch.zeros(*flat.shape[:-1],mapping['count'],dtype=matrix.dtype,device=matrix.device)
    return result.scatter_add(-1,idx,flat*mapping['weights'])


def canonical(coefficient,mapping):
    n=mapping['pairs'].shape[1]
    return (coefficient[...,mapping['target']]*mapping['weights']).reshape(*coefficient.shape[:-1],n,n)
