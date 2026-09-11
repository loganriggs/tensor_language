"""Fully symmetric four-linear contraction of two bias-free bilinear layers.
Input x has [...,4,d]. Keeps shared previous-layer products; no dense quartic tensor.
Down1 may already include a full-output metric transform. Native RMS nodes remain external.
"""
import torch


def contract(x,left0,right0,down0,left1,right1,down1,scale=1.):
    assert x.shape[-2]==4 and left0.shape==right0.shape and left1.shape==right1.shape
    l=x@left0.T;r=x@right0.T
    pair=torch.triu_indices(4,4,offset=1,device=x.device)
    products=(l[...,pair[0],:]*r[...,pair[1],:]+r[...,pair[0],:]*l[...,pair[1],:])/2
    previous=(products@down0.T)*scale
    a=previous@left1.T;b=previous@right1.T
    # Pair order01,02,03,12,13,23; three partitions of four slots.
    i=torch.tensor([0,1,2],device=x.device);j=torch.tensor([5,4,3],device=x.device)
    aggregate=(a[...,i,:]*b[...,j,:]+b[...,i,:]*a[...,j,:]).sum(-2)/6
    return aggregate@down1.T
