"""Exact zero-mean Gaussian quartic feature Gram, including covariance M."""
import torch

def pairings(indices):
 if not indices:yield ();return
 a=indices[0]
 for j,b in enumerate(indices[1:],1):
  for tail in pairings(indices[1:j]+indices[j+1:]):yield ((a,b),)+tail

PAIRINGS=tuple(pairings(tuple(range(8))))

def gaussian_cp_gram(factors,other,covariance=None):
 # Pairing eight scalar linear forms is Wick's eighth-moment formula.
 vectors=list(factors)+list(other);products={}
 for a in range(8):
  for b in range(a+1,8):
   left=vectors[a] if covariance is None else vectors[a]@covariance
   if a<4 and b>=4:value=left@vectors[b].T
   else:
    value=(left*vectors[b]).sum(1);value=value[:,None] if b<4 else value[None,:]
   products[(a,b)]=value
 result=factors[0].new_zeros((len(factors[0]),len(other[0])))
 for matching in PAIRINGS:
  value=products[matching[0]]
  for key in matching[1:]:value=value*products[key]
  result=result+value
 return result
