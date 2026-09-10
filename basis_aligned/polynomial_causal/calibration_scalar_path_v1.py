"""Exact conditional scalar replacement and response derivative, no fitted reader."""
import torch
from calibration_two_readers_v1 import EPS32


def path(background, s, w, U):
    b=background.double();s=s.double();w=w.double();U=U.double()
    A=b@U.T;B=U@w
    c=b.square().mean(-1)+EPS32;d=(b*w).mean(-1);e=w.square().mean()
    rho=(c+2*d*s+e*s.square()).sqrt()
    t=(A+s[...,None]*B)/rho[...,None]
    z=30*torch.tanh(t/30)
    derivative=(B*c[...,None]-A*d[...,None]+s[...,None]*(B*d[...,None]-A*e))/rho[...,None].pow(3)
    derivative=derivative*(1-torch.tanh(t/30).square())
    return z,derivative


def controls():
    g=torch.Generator().manual_seed(9111150)
    rand=lambda *shape:torch.randn(*shape,generator=g,dtype=torch.float64)
    b=rand(7,5);w=rand(5);U=rand(13,5);s=rand(7)
    z,d=path(b,s,w,U);h=b+s[:,None]*w
    expected=30*torch.tanh((h@U.T)/(30*(h.square().mean(-1,keepdim=True)+EPS32).sqrt()))
    step=1e-5;fd=(path(b,s+step,w,U)[0]-path(b,s-step,w,U)[0])/(2*step)
    identity=float((z-expected).abs().max());derivative=float((d-fd).abs().max())
    assert identity<1e-10 and derivative<1e-7
    # Even one pre-softcap vocabulary score need not increase monotonically in s.
    # b=(1,0), w=(0,1), U=(1,0): t(s)=1/sqrt((1+s^2)/2+eps).
    signs=[]
    for value in (-1.,1.):
        signs.append(float(path(torch.tensor([[1.,0.]]),torch.tensor([value]),torch.tensor([0.,1.]),torch.tensor([[1.,0.]]))[1]))
    assert signs[0]>0 and signs[1]<0
    return dict(passed=True,path_max_abs=identity,derivative_max_abs=derivative,nonmonotonic_counterexample_derivatives=signs)
