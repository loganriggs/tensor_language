"""Gaussian source moment for a shared two-key product and linear value read."""
import torch

def moment(value,a,b):
    """value[h,d], a/b[...,d]; returns [...,h,h], no source tensor materialization.

    Computes E[((a.y)*(b.y))^2 (F.y)(F.y)^T] for y~N(0,I).
    No key or value normalization is absorbed in this identity.
    """
    aa=a.square().sum(-1);bb=b.square().sum(-1);ab=(a*b).sum(-1)
    fa=a@value.T;fb=b@value.T
    outer=lambda x,y:x[..., :,None]*y[...,None,:]
    return ((aa*bb+2*ab.square())[...,None,None]*(value@value.T)
            +2*bb[...,None,None]*outer(fa,fa)+2*aa[...,None,None]*outer(fb,fb)
            +4*ab[...,None,None]*(outer(fa,fb)+outer(fb,fa)))
