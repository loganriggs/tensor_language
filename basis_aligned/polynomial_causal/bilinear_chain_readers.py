"""Exact scalar-observable pullback through a conditional bilinear chain."""
import torch


def backward_readers(blocks, mlp_inputs, final_state, readout, eps):
    """Gradient of first softcapped logit minus second, at every chain boundary.

    Uses explicit native weights and states; no parameter learning or autograd.
    These are local readers for choosing a basis, not finite-effect predictions.
    """
    x=final_state;d=x.shape[-1]
    s=x.square().mean(-1,keepdim=True)+eps
    raw=(x@readout.T)/s.sqrt()
    slopes=1-torch.tanh(raw/30).square()
    u=slopes[...,0,None]*readout[0]-slopes[...,1,None]*readout[1]
    q=(u-x*(u*x).sum(-1,keepdim=True)/(d*s))/s.sqrt()
    readers=[q]
    for b,h in reversed(list(zip(blocks,mlp_inputs))):
        lh=h@b['left'].T;rh=h@b['right'].T
        p0=(lh*rh)@b['down'].T
        sh=h.square().mean(-1,keepdim=True)+eps
        w=q@b['down']
        cross=((w*rh)@b['left']+(w*lh)@b['right'])/sh
        radial=2*h*(q*p0).sum(-1,keepdim=True)/(d*sh.square())
        q=b['lambdas'][0]*(q+cross-radial)
        readers.append(q)
    return readers[::-1]


def secant_readers(blocks, base_inputs, edited_inputs, base_final, edited_final,
                   readout, eps):
    """Exact two-endpoint reader: q_l dot delta_state_l equals final margin change.

    Requires both executions, so this is calibration/attribution, not a predictor.
    Symmetric divided differences retain every quadratic and normalization term.
    """
    d=base_final.shape[-1]
    s0=base_final.square().mean(-1,keepdim=True)+eps
    s1=edited_final.square().mean(-1,keepdim=True)+eps
    r0,r1=s0.sqrt(),s1.sqrt()
    raw0=(base_final@readout.T)/r0;raw1=(edited_final@readout.T)/r1
    diff=raw1-raw0
    safe=torch.where(diff.abs()>1e-12,diff,torch.ones_like(diff))
    slopes=torch.where(diff.abs()>1e-12,
        30*(torch.tanh(raw1/30)-torch.tanh(raw0/30))/safe,
        1-torch.tanh(raw0/30).square())
    u=slopes[...,0,None]*readout[0]-slopes[...,1,None]*readout[1]
    mid=(base_final+edited_final)/2
    q=.5*(1/r0+1/r1)*u-2*mid*(mid*u).sum(-1,keepdim=True)/(d*r0*r1*(r0+r1))
    readers=[q]
    for b,h0,h1 in reversed(list(zip(blocks,base_inputs,edited_inputs))):
        mid=(h0+h1)/2
        lm=mid@b['left'].T;rm=mid@b['right'].T
        p0=((h0@b['left'].T)*(h0@b['right'].T))@b['down'].T
        p1=((h1@b['left'].T)*(h1@b['right'].T))@b['down'].T
        s0=h0.square().mean(-1,keepdim=True)+eps
        s1=h1.square().mean(-1,keepdim=True)+eps
        w=q@b['down']
        cross=((w*rm)@b['left']+(w*lm)@b['right'])*.5*(1/s0+1/s1)
        radial=mid*(q*(p0+p1)).sum(-1,keepdim=True)/(d*s0*s1)
        q=b['lambdas'][0]*(q+cross-radial)
        readers.append(q)
    return readers[::-1]
