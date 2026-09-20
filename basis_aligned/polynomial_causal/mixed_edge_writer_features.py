"""Existing rational edge core as27shared writers and scalar feature amplitudes."""
import torch

def writers(core):
    return torch.cat([core[k] for k in ['value_base','value_linear','cached_value']],dim=-1)

def amplitudes(core,u,v):
    anchor=core['product'];b=len(anchor)
    u=torch.as_tensor(u,device=anchor.device,dtype=anchor.dtype).reshape(-1).expand(b)
    v=torch.as_tensor(v,device=anchor.device,dtype=anchor.dtype).reshape(-1).expand(b)
    def powers(t):return torch.stack([torch.ones_like(t),t,t*t],-1)
    def denom(coeff,t):
        values=torch.einsum('bhki,bi->bhk',coeff,powers(t))
        if not bool((values>0).all()):raise ValueError('Nonpositive head norm')
        return values.prod(-1).sqrt()
    z=torch.zeros_like(u);q=powers(u)[:,None]/denom(core['qnorm'],u)[...,None];q0=powers(z)[:,None]/denom(core['qnorm'],z)[...,None]
    def route(t):return torch.einsum('bhij,bhi,bj->bh',core['product'],q-q0,powers(t))/denom(core['knorm'],t)
    rv,r0=route(v),route(z);s=torch.einsum('bi,bi->b',core['input_norm'],powers(v));s0=core['input_norm'][:,0]
    if not bool((s>0).all() and (s0>0).all()):raise ValueError('Nonpositive input norm')
    return torch.cat([rv/s.sqrt()[:,None]-r0/s0.sqrt()[:,None],v[:,None]*rv/s.sqrt()[:,None],rv-r0],dim=-1)

def execute(core,u,v):return torch.einsum('bdr,br->bd',writers(core),amplitudes(core,u,v))
