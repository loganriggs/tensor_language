"""Exact token-numerator and RMS-moment state for one quadratic response path."""
import math
import torch
from calibration_two_readers_v1 import EPS32


def compile_state(h,b,a,readers):
    # h(delta)=h+delta*b+delta^2*a; readers shape [batch,tokens,width].
    a=a.expand_as(h);width=h.shape[-1]
    numerator=torch.stack([torch.einsum('btd,bd->bt',readers,x) for x in [h,b,a]],-1)
    rho=torch.stack([h.square().sum(-1),2*(h*b).sum(-1),2*(h*a).sum(-1)+b.square().sum(-1),2*(b*a).sum(-1),a.square().sum(-1)],-1)/width
    rho[:,0]+=EPS32
    return {'numerator':numerator,'rho':rho}


def evaluate(state,delta):
    delta=torch.as_tensor(delta,dtype=state['rho'].dtype,device=state['rho'].device)
    if delta.ndim==0:delta=delta.expand(state['rho'].shape[0])
    powers=torch.stack([delta**k for k in range(5)],-1)
    numerator=(state['numerator']*powers[:,None,:3]).sum(-1)
    rho=(state['rho']*powers).sum(-1);assert bool((rho>0).all())
    return 30*torch.tanh(numerator/(30*rho.sqrt()[:,None]))


def advance(state,delta):
    delta=torch.as_tensor(delta,dtype=state['rho'].dtype,device=state['rho'].device)
    if delta.ndim==0:delta=delta.expand(state['rho'].shape[0])
    output={}
    for key,coeff in state.items():
        shift=delta[:,None] if key=='numerator' else delta
        output[key]=torch.stack([sum(math.comb(j,k)*coeff[...,j]*shift**(j-k) for j in range(k,coeff.shape[-1])) for k in range(coeff.shape[-1])],-1)
    return output


def controls():
    rng=torch.Generator().manual_seed(9111480)
    rand=lambda *s:torch.randn(*s,generator=rng,dtype=torch.float64)
    h,b,a,v,delta=rand(7,5),rand(7,5),rand(5),rand(7,2,5),rand(7)
    state=compile_state(h,b,a,v);changed=h+delta[:,None]*b+delta[:,None].square()*a
    direct=30*torch.tanh(torch.einsum('btd,bd->bt',v,changed)/(30*(changed.square().mean(-1)+EPS32).sqrt()[:,None]))
    err=float((evaluate(state,delta)-direct).abs().max())
    composition=float((evaluate(advance(state,delta*.4),delta*.6)-direct).abs().max())
    assert err<1e-10 and composition<1e-10
    return {'direct_max_abs':err,'composition_max_abs':composition,'gpu_accessed':False}
