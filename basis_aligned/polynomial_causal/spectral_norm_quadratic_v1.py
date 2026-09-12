"""Weight-only PSD norm quadratic with isotropic orthogonal remainder."""
import torch


def compile_norm(a):
    values,vectors=torch.linalg.eigh(a@a.transpose(-1,-2))
    values=values.flip(-1);vectors=vectors.flip(-1)
    assert bool((values[..., -1]>1e-12*values[...,0]).all())
    readers=(vectors.transpose(-1,-2)@a)/values.sqrt()[...,None]
    return dict(values=values,readers=readers,dimension=a.shape[-1],rows=a.shape[-2])


def norm_mean(x,program,k):
    values=program['values'];readers=program['readers'][:,:k]
    z=torch.einsum('nd,hkd->nhk',x,readers)
    top=(z.square()*values[None,:,:k]).sum(-1)
    beta=values[:,k:].sum(-1)/(program['dimension']-k)
    remainder=x.square().sum(-1)[:,None]-z.square().sum(-1)
    return (top+remainder*beta[None])/program['rows']+torch.finfo(torch.float32).eps
