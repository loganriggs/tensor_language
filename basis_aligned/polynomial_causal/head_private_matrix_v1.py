"""Shared low-rank matrix plus block-supported private low-rank matrices."""
import torch

def lowrank(x,rank):
    u,s,v=torch.linalg.svd(x,full_matrices=False)
    left=u[...,:rank]*s[...,:rank].unsqueeze(-2)
    right=v[...,:rank,:]
    return left@right,left,right

def private_fit(x,heads,rank):
    out,inp=x.shape
    blocks=x.reshape(out,heads,inp//heads).permute(1,0,2)
    fit,left,right=lowrank(blocks,rank)
    return fit.permute(1,0,2).reshape_as(x),left,right

def cycle(x,private,heads,shared_rank,private_rank):
    shared,sl,sr=lowrank(x-private,shared_rank)
    private,pl,pr=private_fit(x-shared,heads,private_rank)
    return shared,private,dict(shared_left=sl,shared_right=sr,private_left=pl,private_right=pr)

def reconstruct(factors):
    shared=factors['shared_left']@factors['shared_right']
    private=factors['private_left']@factors['private_right']
    return shared+private.permute(1,0,2).reshape_as(shared)
