"""Closed state in a fixed writer span; input readers need not lie in that span.

h=alpha*x0+W*a. Immutable input readings and Gram terms are explicitly retained.
"""
import torch


def encode(x0,w,readers,readout):
    return dict(initial_reads=[x0@f.T for f in readers],cross=[f@w for f in readers],
                writer_inner=x0@w,gram=w.T@w,norm0=x0.square().sum(-1),
                output0=x0@readout.T,output_w=readout@w,ambient=x0.shape[-1])


def execute(encoded,blocks,edits=()):
    k=encoded['gram'].shape[0];a=encoded['writer_inner'].new_zeros(encoded['writer_inner'].shape[0],k)
    alpha=a.new_tensor(1.);eps=torch.finfo(torch.float32).eps
    def norm2():
        return alpha.square()*encoded['norm0']+2*alpha*(a*encoded['writer_inner']).sum(-1)+torch.einsum('ni,ij,nj->n',a,encoded['gram'],a)
    for j,block in enumerate(blocks):
        lam,mu=block['reentry'];a=lam*a;alpha=lam*alpha+mu
        reads=alpha*encoded['initial_reads'][j]+a@encoded['cross'][j].T
        reads=reads/(norm2()/encoded['ambient']+eps).sqrt()[:,None]
        pairs=block['products'];products=reads[:,pairs[:,0]]*reads[:,pairs[:,1]]
        for layer,factor in edits:
            if layer==j:products[:,factor]=0
        a=a+products@block['mixing'].T+block['bias']
    norms=norm2();linear=alpha*encoded['output0']+a@encoded['output_w'].T
    logits=30*torch.tanh(linear/(norms/encoded['ambient']+eps).sqrt()[:,None]/30)
    return dict(coefficients=a,alpha=alpha,norm2=norms,logits=logits)
