"""Hermite-degree-resolved quartic CP Gram and native cross moments."""
import itertools
import torch
from noncentral_gaussian_cp import PARTITIONS,affine_moment
from quartic_cp import directional

def gram_degrees(factors,biases):
 vectors=list(factors)*2;means=[v[:,None] for v in biases]+[v[None,:] for v in biases];dots={}
 for i in range(8):
  for j in range(i+1,8):
   if i<4<=j:value=vectors[i]@vectors[j].T
   else:
    value=(vectors[i]*vectors[j]).sum(1);value=value[:,None] if j<4 else value[None,:]
   dots[i,j]=value
 out=[factors[0].new_zeros((len(factors[0]),len(factors[0]))) for _ in range(5)]
 for singles,pairs in PARTITIONS[8]:
  term=factors[0].new_ones((1,1));degree=sum(i<4<=j for i,j in pairs)
  for i in singles:term=term*means[i]
  for pair in pairs:term=term*dots[pair]
  out[degree]=out[degree]+term
 return torch.stack(out)

def cross_degrees(teacher,location,projection,factors,biases):
 mean,linear,quadratic=projection;out=[mean[:,None]*affine_moment(factors,biases)]
 for size in range(1,5):
  result=torch.zeros_like(out[0])
  for selected in itertools.combinations(range(4),size):
   rest=[i for i in range(4) if i not in selected];moment=affine_moment([factors[i] for i in rest],[biases[i] for i in rest]);chosen=[factors[i] for i in selected]
   if size==1:derivative=linear@chosen[0].T
   elif size==2:derivative=2*torch.einsum('ki,vij,kj->vk',chosen[0],quadratic,chosen[1])
   elif size==3:derivative=24*directional(*teacher,[location.expand_as(chosen[0])]+chosen).T
   else:derivative=24*directional(*teacher,chosen).T
   result=result+derivative*moment
  out.append(result)
 return torch.stack(out)

def combine(parts,rho):
 if rho is None:weights=[1.]*5
 elif rho==1:weights=list(range(5))
 else:weights=[sum(rho**j for j in range(k)) for k in range(5)]
 return sum(w*p for w,p in zip(weights,parts))
