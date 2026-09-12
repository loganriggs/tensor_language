"""Validated shared-component branch. See README.md for required inputs."""
import itertools
import torch


def execute_head(query,source,rotation,p):
    eps=torch.finfo(torch.float32).eps;dim=p['q1'].shape[-1];width=p['q1'].shape[0]
    qa=query@p['q1'].T;qb=query@p['q2'].T
    ka=source[:,:dim]@p['k1'].T;kb=source[:,:dim]@p['k2'].T
    gate=1/(width**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt())
    z=(source@p['parent'].T).prod(-1)[:,None]*(source@p['children'].T)
    dual=z@p['dual']
    a=torch.einsum('nk,kl,ril->nri',qa,rotation,p['atom_k1'])
    b=torch.einsum('nk,kl,ril->nri',qb,rotation,p['atom_k2'])
    result=query.new_zeros(query.shape[0],p['atom_write'].shape[-1])
    for i,j,k in itertools.permutations(range(3)):
        result+=torch.einsum('nr,nr,nr,ro->no',dual,a[:,:,i],b[:,:,j],p['atom_write'][:,k])
    return gate[:,None]*result
