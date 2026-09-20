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
