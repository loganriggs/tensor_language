"""Standalone selected shared-component head at a declared q/source interface."""
import itertools
import torch
from shared_cubic_source_projection_v1 import atom_gram


def compile_head(atoms,q1,k1,q2,k2,value,output):
    return dict(parent=atoms[-2,:2].clone(),children=atoms[-2:,2].clone(),
        dual=torch.linalg.inv(atom_gram(atoms))[-2:].clone(),
        q1=q1.clone(),q2=q2.clone(),k1=k1.clone(),k2=k2.clone(),
        atom_k1=torch.einsum('kd,rid->rik',k1,atoms[:,:,:k1.shape[-1]]),
        atom_k2=torch.einsum('kd,rid->rik',k2,atoms[:,:,:k2.shape[-1]]),
        atom_write=torch.einsum('ok,kz,riz->rio',output,value,atoms)/6)


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
