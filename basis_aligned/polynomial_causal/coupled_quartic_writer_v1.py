"""Exact coefficient Gram and target contraction for signed quadratic squares."""
import torch

def gram(b,nu):
    h=torch.einsum('kdi,ldj->klij',b,b)
    tr=(h.square()*nu[:,None,:,None]*nu[None,:,None,:]).sum((-1,-2))
    a=(nu[:,None,:,None]*h*nu[None,:,None,:])@h.transpose(-1,-2)
    tr2=(a*a.transpose(-1,-2)).sum((-1,-2))
    return (tr.square()+2*tr2)/3

def target_cross(b,nu,oracle):
    rank=b.shape[-1];i,j=torch.meshgrid(torch.arange(rank,device=b.device),torch.arange(rank,device=b.device),indexing='ij');i=i.flatten();j=j.flatten()
    out=[]
    for bk,nk in zip(b,nu):
        slots=torch.stack([bk[:,i].T,bk[:,i].T,bk[:,j].T,bk[:,j].T],1)
        values=oracle(slots)
        out.append((values*(nk[i]*nk[j])[:,None]).sum(0))
    return torch.stack(out)

def solve(k,c):
    scale=k.diagonal().sqrt();kn=k/scale[:,None]/scale[None,:];cn=c/scale[:,None]
    eig,vec=torch.linalg.eigh(kn);keep=eig>eig.max()*1e-12
    normalized=vec[:,keep]@((vec[:,keep].T@cn)/eig[keep,None])
    weights=normalized/scale[:,None]
    residual=(kn@normalized-cn).norm()/cn.norm()
    return weights,dict(rank=int(keep.sum()),normal_residual=float(residual),minimum_scaled_eigenvalue=float(eig.min()))

def features(b,nu,x):
    reads=torch.einsum('nd,kdr->nkr',x,b)
    return ((reads.square()*nu).sum(-1)).square()
