"""Exact quadratic split of direct embedding re-entry and its complement."""
import torch
from calibration_two_readers_v1 import EPS32,scalar


def embedding_coefficient(lambdas):
    coefficient=1.
    for keep,inject in lambdas:coefficient=float(keep)*coefficient+float(inject)
    return coefficient


def terms(a,token_route,Q):
    a=a.double();t=token_route.double();c=a-t;Q=Q.double()
    rho2=a.square().mean(-1)+EPS32
    tt=((t@Q)*t).sum(-1)/rho2
    tc=2*((t@Q)*c).sum(-1)/rho2
    cc=((c@Q)*c).sum(-1)/rho2
    return dict(token=tt,interaction=tc,context=cc)


def controls():
    gen=torch.Generator().manual_seed(9111213)
    rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    a=rand(12,7);t=rand(12,7);raw=rand(7,7);Q=(raw+raw.T)/2;beta=torch.tensor(.7)
    u=a/(a.square().mean(-1,keepdim=True)+EPS32).sqrt()
    split=terms(a,t,Q);error=float((sum(split.values())+beta-scalar(u,Q,beta)).abs().max())
    assert error<1e-10
    assert float(split['interaction'].norm())>.1
    lambdas=[(2.,3.),(.5,-1.),(-2.,4.)];x=rand(7);y=x.clone()
    for keep,inject in lambdas:y=keep*y+inject*x
    coef=embedding_coefficient(lambdas);assert torch.allclose(y,coef*x,atol=1e-12,rtol=1e-12)
    return dict(passed=True,quadratic_sum_max_abs=error,coefficient=coef,interaction_live=True,scope='C=a-T retains other token-dependent computation; native normalization is explicit')
