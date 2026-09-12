"""Differentiable coefficient contractions for a fixed graph of quadratic products."""
import torch

def gram(b,nu,edges):
    h=torch.einsum('kdi,ldj->klij',b,b)
    trace=(h.square()*nu[:,None,:,None]*nu[None,:,None,:]).sum((-1,-2))
    weighted=h*nu[None,:,None,:]
    i,j=edges
    first=weighted[i[:,None],i[None,:]]@weighted[i[None,:],j[:,None]]
    second=weighted[j[:,None],j[None,:]]@weighted[j[None,:],i[:,None]]
    cycle=(first*second.transpose(-1,-2)).sum((-1,-2))
    return (trace[i[:,None],i[None,:]]*trace[j[:,None],j[None,:]]+
            trace[i[:,None],j[None,:]]*trace[j[:,None],i[None,:]]+4*cycle)/6

def target_cross(b,nu,edges,oracle):
    rank=b.shape[-1]
    a,c=torch.meshgrid(torch.arange(rank,device=b.device),torch.arange(rank,device=b.device),indexing='ij')
    a=a.flatten();c=c.flatten();out=[]
    for i,j in edges.T:
        slots=torch.stack([b[i,:,a].T,b[i,:,a].T,b[j,:,c].T,b[j,:,c].T],1)
        out.append((oracle(slots)*(nu[i,a]*nu[j,c])[:,None]).sum(0))
    return torch.stack(out)

def features(b,nu,edges,x):
    q=(torch.einsum('nd,kdr->nkr',x,b).square()*nu).sum(-1)
    return q[:,edges[0]]*q[:,edges[1]]
