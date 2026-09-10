"""Exact scalar residual accumulation, retaining learned re-entry coefficients."""
import torch

def coefficients(lambdas):
    lam=lambdas.double();n=len(lam);weights=torch.ones(n,device=lam.device,dtype=torch.float64)
    product=torch.ones((),device=lam.device,dtype=torch.float64)
    for j in reversed(range(n)):
        weights[j]=product;product=product*lam[j,0]
    embedding=product+(weights*lam[:,1]).sum()
    return weights,embedding

def controls():
    gen=torch.Generator().manual_seed(9111380);rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    lam=rand(5,2);x0=rand(4,7);writes=rand(5,2,4,7);x=x0.clone()
    for j in range(5):x=lam[j,0]*x+lam[j,1]*x0+writes[j,0]+writes[j,1]
    w,b=coefficients(lam);fold=b*x0+(w[:,None,None]*writes.sum(1)).sum(0)
    e=rand(7);e/=e.norm();donor=rand(5,2,4)
    edited=writes+((donor-writes@e)[...,None]*e)
    z=x0.clone()
    for j in range(5):z=lam[j,0]*z+lam[j,1]*x0+edited[j].sum(0)
    expected=b*(x0@e)+(w[:,None]*donor.sum(1)).sum(0)
    errors={'vector_accumulation':float((fold-x).abs().max()),'all_port_scalar_control':float((z@e-expected).abs().max())}
    assert max(errors.values())<1e-10
    return errors
