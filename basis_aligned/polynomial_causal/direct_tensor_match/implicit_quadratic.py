"""Differentiable exact CP quadratic metric, without output-by-input-by-input tensor."""
import torch

def inner(c,a,b,d,l,r,gaussian=True,block=256):
 total=c.new_zeros(())
 for i in range(0,len(a),block):
  aa,bb=a[i:i+block],b[i:i+block]
  gram=(aa@l.T)*(bb@r.T)+(aa@r.T)*(bb@l.T)
  total=total+((c[:,i:i+block].T@d)*gram).sum()
 if gaussian:
  ta=c@(a*b).sum(-1);tb=d@(l*r).sum(-1)
  return total+(ta*tb).sum()
 return total*.5

def compare(c,a,b,d,l,r):
 t=inner(c,a,b,c,a,b);s=inner(d,l,r,d,l,r);x=inner(c,a,b,d,l,r)
 return (t+s-2*x)/t
