"""Differentiable exact secant coordinates for one pair of source cubics.
For fixed t>0 this is an invertible linear reparameterization of the two
aligned reader triples; the even/odd feature basis is evaluated without
subtracting nearly equal products. Output coefficients remain freely solved.
"""
import itertools
import torch
from cubic_secant_block_v1 import align_pair

def encode(atoms,pair):
 i,j=pair
 other=[k for k in range(len(atoms)) if k not in pair]
 second=align_pair(atoms[i],atoms[j]);base=(atoms[i]+second)/2
 delta=(atoms[i]-second)/2;t=float(delta.norm())
 if t<=1e-12:raise ValueError('Coincident pair needs a separate tangent-limit representation')
 return torch.cat((atoms[other],base[None],(delta/t)[None])),t

def components(theta,t):
 n=len(theta)-2;products=[]
 for mask in range(8):
  products.append(torch.stack([theta[-1,j] if mask&(1<<j) else theta[-2,j] for j in range(3)]))
 atoms=torch.cat((theta[:n],torch.stack(products)))
 mix=theta.new_zeros(n+2,n+8);mix[:n,:n]=torch.eye(n,device=theta.device,dtype=theta.dtype)
 for mask in range(8):
  degree=mask.bit_count();mix[n+degree%2,n+mask]=1 if degree<2 else t*t
 return atoms,mix

def dense_features(atoms):
 raw=torch.einsum('ri,rj,rk->rijk',atoms[:,0],atoms[:,1],atoms[:,2])
 return sum(raw.permute(0,*(i+1 for i in p)) for p in itertools.permutations(range(3))).flatten(1)/6

def features(theta,t=None):
 if t is None:return dense_features(theta)
 a,m=components(theta,t);return m@dense_features(a)

def raw_atoms(theta,t=None):
 if t is None:return theta
 return torch.cat((theta[:-2],(theta[-2]+t*theta[-1])[None],(theta[-2]-t*theta[-1])[None]))

def normalized_reader_gradient(theta,gradient,t=None):
 """Gradient in unit-reader coordinates, independent of raw/secant scaling."""
 a=raw_atoms(theta,t)
 g=gradient if t is None else torch.cat((gradient[:-2],((gradient[-2]+gradient[-1]/t)/2)[None],((gradient[-2]-gradient[-1]/t)/2)[None]))
 norm=a.norm(dim=-1,keepdim=True);unit=a/norm
 return norm*(g-(g*unit).sum(-1,keepdim=True)*unit)
