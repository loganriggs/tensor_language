"""Shared head2 branch with first-state readings compiled into token lookup."""
import itertools
import torch


def execute_token_head(query,current,token_ids,rotation,p,child_mask=(1.,1.)):
    eps=torch.finfo(torch.float32).eps;width=p['q1'].shape[0]
    qa=query@p['q1'].T;qb=query@p['q2'].T
    ka=current@p['k1'].T;kb=current@p['k2'].T
    gate=1/(width**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt())
    readings=current@p['current_readers'].T+p['token_reads'][token_ids]
    z=readings[:,:2].prod(-1)[:,None]*readings[:,2:]*readings.new_tensor(child_mask)
    dual=z@p['dual']
    a=torch.einsum('nk,kl,ril->nri',qa,rotation,p['atom_k1'])
    b=torch.einsum('nk,kl,ril->nri',qb,rotation,p['atom_k2'])
    result=query.new_zeros(query.shape[0],p['atom_write'].shape[-1])
    for i,j,k in itertools.permutations(range(3)):
        result+=torch.einsum('nr,nr,nr,ro->no',dual,a[:,:,i],b[:,:,j],p['atom_write'][:,k])
    return gate[:,None]*result
