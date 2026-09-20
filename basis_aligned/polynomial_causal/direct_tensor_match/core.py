"""Exact data-free polynomial matching. Coefficients are canonical monomials.

Small-degree reference: no sampled activations in optimization. Gaussian moments
and symmetrized Frobenius metrics are different, both implemented exactly.
"""
import itertools,math
from functools import lru_cache
import torch
from torch import nn

@lru_cache(None)
def terms(d,n):
 return tuple(itertools.combinations_with_replacement(range(d),n))

def counts(t,d):return [t.count(i) for i in range(d)]

def metric(d,n,kind='gaussian',device='cpu'):
 ts=terms(d,n);cs=[counts(t,d) for t in ts]
 if kind=='frobenius':
  return torch.diag(torch.tensor([math.prod(math.factorial(v) for v in c)/math.factorial(n) for c in cs],dtype=torch.float64,device=device))
 def moment(k):return 0 if k%2 else math.prod(range(1,k,2))
 return torch.tensor([[math.prod(moment(x+y) for x,y in zip(a,b)) for b in cs] for a in cs],dtype=torch.float64,device=device)

def multiply(a,b,d,n,m):
 lookup={t:i for i,t in enumerate(terms(d,n+m))}
 index=torch.tensor([lookup[tuple(sorted(x+y))] for x in terms(d,n) for y in terms(d,m)],device=a.device)
 value=(a[..., :,None]*b[...,None,:]).flatten(-2)
 out=value.new_zeros(value.shape[:-1]+(len(lookup),))
 return out.scatter_add(-1,index.expand_as(value),value)

def packed_quadratic(left,right):
 d=left.shape[-1];i,j=torch.triu_indices(d,d,device=left.device)
 return left[...,i]*right[...,j]+left[...,j]*right[...,i]*(i!=j)

def normrows(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)

class Model(nn.Module):
 def __init__(self,d,o,kind,width,degree=2,output_rank=2):
  super().__init__();self.d=d;self.o=o;self.kind=kind;self.width=width;self.degree=degree
  def param(name,*shape):setattr(self,name,nn.Parameter(torch.randn(*shape,dtype=torch.float64)*.3))
  q=len(terms(d,2))
  if kind=='monomial':param('weight',o,len(terms(d,degree)))
  elif kind=='cp':param('left',width,d);param('right',width,d);param('weight',o,width)
  elif kind=='tucker':
   param('leaf',width,d);param('core',output_rank,width*(width+1)//2);param('weight',o,output_rank)
  elif kind=='tree':param('left',width,q);param('right',width,q);param('weight',o,width*width)
  elif kind=='dag':param('bank',width,q);param('weight',o,width*(width+1)//2)
  else:raise ValueError(kind)
 def atoms(self):
  d=self.d;k=self.kind
  if k=='cp':return packed_quadratic(normrows(self.left),normrows(self.right))
  if k=='tucker':
   leaf=normrows(self.leaf);i,j=torch.triu_indices(self.width,self.width,device=leaf.device)
   return self.core@packed_quadratic(leaf[i],leaf[j])
  if k=='tree':
   l,r=normrows(self.left),normrows(self.right)
   return multiply(l[:,None,:],r[None,:,:],d,2,2).reshape(self.width**2,-1)
  if k=='dag':
   bank=normrows(self.bank);i,j=torch.triu_indices(self.width,self.width,device=bank.device)
   return multiply(bank[i],bank[j],d,2,2)
 def forward(self):return self.weight if self.kind=='monomial' else self.weight@self.atoms()
 def penalty(self):
  # Dictionary atoms fixed in row scale. Joint scale of Tucker core is penalized too.
  return self.weight.abs().mean()+(self.core.abs().mean() if self.kind=='tucker' else 0.)

def inner(a,b,M):return ((a@M)*b).sum()
def coefficients_from_dense(tensor):
 d=tensor.shape[1];n=tensor.ndim-1;lookup={t:i for i,t in enumerate(terms(d,n))}
 idx=torch.tensor([lookup[tuple(sorted(t))] for t in itertools.product(range(d),repeat=n)],device=tensor.device)
 return tensor.new_zeros(tensor.shape[0],len(lookup)).scatter_add(1,idx[None].expand(tensor.shape[0],-1),tensor.flatten(1))

def evaluate(c,x,n):
 mon=torch.stack([x[:,list(t)].prod(-1) for t in terms(x.shape[-1],n)],-1)
 return mon@c.T

def covariance_metric(d,n,covariance):
 """Exact correlated-Gaussian monomial Gram via Wick recursion; no samples."""
 from functools import lru_cache
 sigma=covariance.detach().cpu().tolist()
 @lru_cache(None)
 def moment(indices):
  if not indices:return 1.
  if len(indices)%2:return 0.
  i=indices[0];rest=indices[1:]
  return sum(sigma[i][j]*moment(rest[:k]+rest[k+1:]) for k,j in enumerate(rest))
 ts=terms(d,n)
 return torch.tensor([[moment(tuple(sorted(a+b))) for b in ts] for a in ts],dtype=covariance.dtype,device=covariance.device)
