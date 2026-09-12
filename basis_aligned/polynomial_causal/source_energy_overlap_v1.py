"""Coordinate-invariant overlap of projected source energy operators."""
import torch

def operators(gram,kernels):
    eig,vec=torch.linalg.eigh(gram)
    if float(eig.min())<=0:raise ValueError('Source dictionary must be independent')
    invroot=(vec/eig.sqrt())@vec.T
    return invroot[None]@kernels@invroot[None]

def overlap(gram,kernels):
    m=operators(gram,kernels);flat=m.flatten(1)
    return (flat@flat.T)/(flat.norm(dim=1)[:,None]*flat.norm(dim=1)[None,:]).clamp_min(1e-300)

def literal_shared(gram,kernels):
    inv=torch.linalg.inv(gram)
    e=(inv[None]@kernels@inv[None]).diagonal(dim1=-2,dim2=-1)*gram.diagonal()[None]
    fractions=e/e.sum(0,keepdim=True).clamp_min(1e-300)
    return int(((fractions>=.1).sum(0)>=2).sum())
