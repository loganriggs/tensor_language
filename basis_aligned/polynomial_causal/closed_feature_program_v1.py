"""Closed bilinear residual program in a common orthonormal reader/writer frame.

Assumptions: every learned input reader and output write/bias lies in span(B).
Residual reentry may use the original input. Arbitrary full-output readers keep
an explicitly charged initial-complement readout; no intermediate native state
is supplied to execute(). This is not a claim about the native checkpoint.
"""
import torch


def encode(x0,b,readout):
    z0=x0@b;perp=x0-z0@b.T
    return dict(z0=z0,perp_norm2=perp.square().sum(-1),perp_readout=perp@readout.T,
                feature_readout=readout@b,ambient=x0.shape[-1])


def execute(encoded,blocks,edits=(),omit_perpendicular_norm=False):
    z0=encoded['z0'];z=z0.clone();alpha=z.new_tensor(1.)
    d=encoded['ambient'];eps=torch.finfo(torch.float32).eps
    complement=encoded['perp_norm2']*(0 if omit_perpendicular_norm else 1)
    edit_set=set(edits)
    for j,block in enumerate(blocks):
        lam,mu=block['reentry'];z=lam*z+mu*z0;alpha=lam*alpha+mu
        rho=((z.square().sum(-1)+alpha.square()*complement)/d+eps).sqrt()
        inp=z/rho[:,None];products=(inp@block['left'].T)*(inp@block['right'].T)
        for layer,factor in edit_set:
            if layer==j:products[:,factor]=0
        z=z+products@block['down'].T+block['bias']
    norm2=z.square().sum(-1)+alpha.square()*complement
    logits=(z@encoded['feature_readout'].T+alpha*encoded['perp_readout'])/(norm2/d+eps).sqrt()[:,None]
    return dict(features=z,norm2=norm2,logits=30*torch.tanh(logits/30),alpha=alpha)
