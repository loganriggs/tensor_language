"""Exact finite-observable pullback through a dynamic attention/MLP suffix."""
import torch
from finite_attention_readers import rms,rms_pullback,pullback as attention_pullback


def readout_reader(base,edited,readout_pairs,positions,eps):
    """Return [readers,batch,tokens,width] readers of softcapped logit contrasts."""
    batch=torch.arange(len(base),device=base.device)
    x0=base[batch,positions];x1=edited[batch,positions]
    raw0=torch.einsum('bd,rcd->rbc',rms(x0,eps),readout_pairs)
    raw1=torch.einsum('bd,rcd->rbc',rms(x1,eps),readout_pairs)
    difference=raw1-raw0;safe=torch.where(difference.abs()>1e-12,difference,torch.ones_like(difference))
    slopes=torch.where(difference.abs()>1e-12,30*(torch.tanh(raw1/30)-torch.tanh(raw0/30))/safe,
                       1-torch.tanh(raw0/30).square())
    g=slopes[...,0,None]*readout_pairs[:,None,0]-slopes[...,1,None]*readout_pairs[:,None,1]
    selected=rms_pullback(x0,x1,g,eps)
    q=torch.zeros((len(readout_pairs),)+base.shape,device=base.device,dtype=base.dtype)
    q[:,batch,positions]=selected
    return q


def mlp_pullback(block,h0,h1,q,eps):
    """Includes residual skip; output bias cancels in the paired response."""
    mid=(h0+h1)/2;d=mid.shape[-1]
    lm=mid@block['left'].T;rm=mid@block['right'].T
    p0=((h0@block['left'].T)*(h0@block['right'].T))@block['down'].T
    p1=((h1@block['left'].T)*(h1@block['right'].T))@block['down'].T
    s0=h0.square().mean(-1,keepdim=True)+eps;s1=h1.square().mean(-1,keepdim=True)+eps
    w=q@block['down']
    cross=((w*rm)@block['left']+(w*lm)@block['right'])*.5*(1/s0+1/s1)
    radial=mid*(q*(p0+p1)).sum(-1,keepdim=True)/(d*s0*s1)
    return q+cross-radial


def secant_readers(blocks,base,edited,first_value,readout_pairs,positions,eps):
    """Trace dictionaries hold states, raw_attention_inputs, and mlp_inputs.

    states has one extra entry: the incoming state before the first block.
    Both full executions are required. This is not a low-rank or causal guarantee.
    """
    q=readout_reader(base['states'][-1],edited['states'][-1],readout_pairs,positions,eps)
    readers=[q]
    for l in reversed(range(len(blocks))):
        b=blocks[l]
        qh=mlp_pullback(b,base['mlp_inputs'][l],edited['mlp_inputs'][l],q,eps)
        raw0=base['raw_attention_inputs'][l];raw1=edited['raw_attention_inputs'][l]
        qa=attention_pullback(b['attention'],raw0,raw1,first_value,b['mixture'],b['heads'],
                              b['cos'],b['sin'],eps,b['head_eps'],qh)
        q=b['lambdas'][0]*(qh+qa)
        readers.append(q)
    return readers[::-1]
