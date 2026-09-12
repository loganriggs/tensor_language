"""Exact small-cluster reader chart, retaining one feature per original cubic.
Only a coordinate change: K aligned reader triples become a mean and K-1
independent reader-space directions. Product moments avoid subtraction of
large nearly identical cubics. Singular reader clusters are rejected.
"""
import itertools
import torch
from cubic_secant_block_v1 import align_pair

def encode(atoms,indices):
 indices=list(indices);k=len(indices);assert k>=2
 aligned=torch.stack([atoms[indices[0]]]+[align_pair(atoms[indices[0]],atoms[i]) for i in indices[1:]])
 base=aligned.mean(0);centered=(aligned-base).flatten(1);u,s,v=torch.linalg.svd(centered,full_matrices=False)
 u=u[:,:k-1];s=s[:k-1];direction=v[:k-1].reshape(k-1,3,-1)
 if float(s.min())<=1e-12:raise ValueError('Singular reader cluster requires a different limiting chart')
 others=[i for i in range(len(atoms)) if i not in indices]
 theta=torch.cat((atoms[others],base[None],direction));loading=torch.cat((torch.ones(k,1,dtype=atoms.dtype,device=atoms.device),u*s[None,:]),1)
 return theta,dict(cluster_size=k,loading=loading,singular_values=s,indices=indices)

def components(theta,chart):
 k=chart['cluster_size'];n=len(theta)-k;L=chart['loading'];W=torch.linalg.inv(L)
 tuples=list(itertools.product(range(k),repeat=3));bank=theta[-k:]
 atoms=torch.cat((theta[:n],torch.stack([torch.stack([bank[r,j] for j,r in enumerate(term)]) for term in tuples])))
 mix=theta.new_zeros(n+k,n+k**3);mix[:n,:n]=torch.eye(n,dtype=theta.dtype,device=theta.device)
 for t,term in enumerate(tuples):
  degree=sum(r!=0 for r in term)
  if degree==0:mix[n,n+t]=1
  elif degree==1:
   r=next(r for r in term if r!=0);mix[n+r,n+t]=1
  else:
   coefficient=L[:,term[0]]*L[:,term[1]]*L[:,term[2]];mix[n:,n+t]=W@coefficient
 return atoms,mix

def reconstruct(theta,chart):
 k=chart['cluster_size'];return torch.cat((theta[:-k],torch.einsum('ir,rjd->ijd',chart['loading'],theta[-k:])))

def normalized_reader_gradient(theta,gradient,chart):
 k=chart['cluster_size'];g=torch.linalg.solve(chart['loading'].T,gradient[-k:].flatten(1)).reshape(gradient[-k:].shape)
 g=torch.cat((gradient[:-k],g));raw=reconstruct(theta,chart);norm=raw.norm(dim=-1,keepdim=True);unit=raw/norm
 return norm*(g-(g*unit).sum(-1,keepdim=True)*unit)
