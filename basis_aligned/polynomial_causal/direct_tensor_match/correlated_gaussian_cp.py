"""Exact affine quartic CP cross Gram for correlated standard Gaussian inputs."""
from functools import lru_cache
import torch

def gram(factors,biases,other,other_biases,rho=1.):
 vectors=list(factors)+list(other);means=[v[:,None] for v in biases]+[v[None,:] for v in other_biases];dots={}
 for i in range(8):
  for j in range(i+1,8):
   if i<4<=j:value=rho*(vectors[i]@vectors[j].T)
   else:
    value=(vectors[i]*vectors[j]).sum(1);value=value[:,None] if j<4 else value[None,:]
   dots[i,j]=value
 unit=factors[0].new_ones(1,1)
 @lru_cache(None)
 def moment(mask):
  if mask==0:return unit
  i=(mask&-mask).bit_length()-1;rest=mask^(1<<i);result=means[i]*moment(rest)
  for j in range(i+1,8):
   if rest&(1<<j):result=result+dots[i,j]*moment(rest^(1<<j))
  return result
 result=moment(255);moment.cache_clear();dots.clear();means.clear();return result

def metric(f,b,g,c,rho=None):
 value=gram(f,b,g,c)
 return value if rho is None else (value-gram(f,b,g,c,rho))/(1-rho)
