"""Sparse symmetric core with each unordered feature product computed once."""
import torch


def compile_core(core):
    n=core.shape[-1]
    p,q=torch.triu_indices(n,n,device=core.device)
    alpha,pair=torch.where(core[:,p,q]!=0)
    ids,inverse=torch.unique(p[pair]*n+q[pair],sorted=True,return_inverse=True)
    coefficients=core[alpha,p[pair],q[pair]]*torch.where(p[pair]==q[pair],1,2)
    return dict(p=ids//n,q=ids%n,alpha=alpha,pair=inverse,coefficients=coefficients,outputs=core.shape[0])


def execute(z,W,P,program):
    s=z@P
    products=s[...,program['p']]*s[...,program['q']]
    terms=products[...,program['pair']]*program['coefficients']
    latent=torch.zeros(*z.shape[:-1],program['outputs'],device=z.device,dtype=z.dtype)
    latent.scatter_add_(-1,program['alpha'].expand_as(terms),terms)
    return latent@W.T
