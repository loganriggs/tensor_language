"""Exact coefficient norm of fixed-value sums of joint-QK products."""
import torch

def energy(pairs,v):
 raw=v.new_zeros(());contract=None
 for a,b in pairs:
  av=a@v;bv=b@v
  z=(av[:,None,None]*b[None,:,:]+a[:,None,:]*bv[None,:,None]+bv[:,None,None]*a[None,:,:]+b[:,None,:]*av[None,:,None])/4
  contract=z if contract is None else contract+z
  for c,d in pairs:
   raw+=((a*c).sum()*(b*d).sum()+(a*d).sum()*(b*c).sum()+((a.T@d)*(c.T@b)).sum()+((a@d.T)*(c@b.T)).sum())/4
 return v.square().sum()*raw/3+2*contract.square().sum()/3
