"""Isotropic quartic metric using orthonormal quadratic readers without dense feature traces."""
import torch

def metric(k,c,b,n,target_traces):
    eye=torch.eye(b.shape[-1],dtype=b.dtype,device=b.device)
    assert float((b.transpose(-1,-2)@b-eye).abs().max())<=1e-8
    beta=(n*n.sum(-1,keepdim=True)+2*n.square())/3
    h=torch.einsum('kdi,ldj->klij',b,b)
    tracegram=(h.square()*beta[:,None,:,None]*beta[None,:,None,:]).sum((-1,-2))
    projected=torch.einsum('kdi,mde->kmie',b,target_traces)
    diagonal=(projected*b.transpose(-1,-2)[:,None,:,:]).sum(-1)
    tracecross=(diagonal*beta[:,None,:]).sum(-1)
    trace=beta.sum(-1);targettrace=target_traces.diagonal(dim1=-2,dim2=-1).sum(-1)
    return 24*k+72*tracegram+9*trace[:,None]*trace[None,:],24*c+72*tracecross+9*trace[:,None]*targettrace[None,:]
