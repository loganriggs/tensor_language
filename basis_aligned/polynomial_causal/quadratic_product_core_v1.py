"""Exact fully symmetric quartic coefficient core over shared quadratic intermediates."""
import torch

def pairs(count,device=None):return torch.triu_indices(count,count,device=device)

def gram(b,nu):
 h=torch.einsum('kdi,ldj->klij',b,b)
 tr=(h.square()*nu[:,None,:,None]*nu[None,:,None,:]).sum((-1,-2))
 weighted=h*nu[None,:,None,:]
 ij=pairs(len(b),b.device);rows=[]
 for i,j in ij.T:
  first=weighted[i]@weighted[:,j]
  second=weighted[j]@weighted[:,i]
  cycle=torch.einsum('kab,lba->kl',first,second)
  block=(tr[i,:,None]*tr[j,None,:]+tr[j,:,None]*tr[i,None,:]+4*cycle)/6
  rows.append(block[ij[0],ij[1]])
 return torch.stack(rows)

def target_cross(b,nu,oracle):
 rank=b.shape[-1];a,c=torch.meshgrid(torch.arange(rank,device=b.device),torch.arange(rank,device=b.device),indexing='ij');a=a.flatten();c=c.flatten();out=[]
 for i,j in pairs(len(b),b.device).T:
  slots=torch.stack([b[i,:,a].T,b[i,:,a].T,b[j,:,c].T,b[j,:,c].T],1)
  out.append((oracle(slots)*(nu[i,a]*nu[j,c])[:,None]).sum(0))
 return torch.stack(out)

def quadratic_values(b,nu,x):return (torch.einsum('nd,kdr->nkr',x,b).square()*nu).sum(-1)
def features(b,nu,x):
 q=quadratic_values(b,nu,x);ij=pairs(len(b),b.device)
 return q[:,ij[0]]*q[:,ij[1]]
