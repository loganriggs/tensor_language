"""Mixed-precision shared head: source readings and dual mix use FP64."""
import itertools
import torch


def execute_feature_head(query,current,z,rotation,p):
    eps=torch.finfo(torch.float32).eps;width=p['q1'].shape[0]
    qa=query@p['q1'].T;qb=query@p['q2'].T
    ka=current@p['k1'].T;kb=current@p['k2'].T
    gate=1/(width**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt())
    dual=(z@p['dual']).to(query.dtype)
    a=torch.einsum('nk,kl,ril->nri',qa,rotation,p['atom_k1'])
    b=torch.einsum('nk,kl,ril->nri',qb,rotation,p['atom_k2'])
    result=query.new_zeros(query.shape[0],p['atom_write'].shape[-1])
    for i,j,k in itertools.permutations(range(3)):
        result+=torch.einsum('nr,nr,nr,ro->no',dual,a[:,:,i],b[:,:,j],p['atom_write'][:,k])
    return gate[:,None]*result
