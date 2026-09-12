"""Exact quartic trace contractions and isotropic repeated-input metric."""
import torch

def dense_trace(t):return torch.einsum('...abcc->...ab',t)
def repeated_inner(k,a,b):
 return 24*k+72*torch.einsum('iab,jab->ij',a,b)+9*a.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None]*b.diagonal(dim1=-2,dim2=-1).sum(-1)[None,:]
def square_traces(b,nu):
 q=torch.einsum('kdi,ki,kei->kde',b,nu,b)
 return (q*q.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]+2*(q@q))/3

def native_traces(l0,r0,d0,l1,r1,d1,scale=1.):
 ll=l0@l0.T;lr=l0@r0.T;rr=r0@r0.T;tr=(l0*r0).sum(-1);outputs=[]
 for coefficient in d1:
  matrix=l1.T@(coefficient[:,None]*r1);matrix=(matrix+matrix.T)/2
  h=(d0.T@matrix@d0)*scale**2
  weight=h@tr;first=l0.T@(weight[:,None]*r0);first=(first+first.T)/2
  product=(l0.T@((h*lr.T)@r0)+l0.T@((h*rr)@l0)+r0.T@((h*ll)@r0)+r0.T@((h*lr)@l0))/4
  outputs.append((first+2*product)/3)
 return torch.stack(outputs)
