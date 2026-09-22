"""Quadratic Gaussian Gram and optimal scalar quartic substitution controls."""
import itertools
import numpy as np
import torch
from noncentral_gaussian_cp import PARTITIONS,affine_moment

def gram(a,b,alpha,beta):
 vectors=[a,b,a,b];means=[alpha[:,None],beta[:,None],alpha[None,:],beta[None,:]];dots={}
 for i in range(4):
  for j in range(i+1,4):
   if i<2<=j:v=vectors[i]@vectors[j].T
   else:
    v=(vectors[i]*vectors[j]).sum(1);v=v[:,None] if j<2 else v[None,:]
   dots[i,j]=v
 out=a.new_zeros(len(a),len(a))
 for singles,pairs in PARTITIONS[4]:
  term=a.new_ones(1,1)
  for i in singles:term=term*means[i]
  for ij in pairs:term=term*dots[ij]
  out+=term
 return out

def controls():
 nodes,w=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=2)));z=torch.tensor(nodes*2**.5,dtype=torch.float64)[ix];weight=torch.tensor(w/np.pi**.5,dtype=torch.float64)[ix].prod(1);rows=[]
 for seed in range(5):
  torch.manual_seed(36000+seed);f=[torch.randn(3,2,dtype=torch.float64) for _ in range(4)];bias=[torch.randn(3,dtype=torch.float64) for _ in range(4)]
  if seed==1:f[1]=f[0].clone()
  if seed==2:f[2]=-f[0]
  if seed==3:bias=[b*0 for b in bias]
  if seed==4:f[3]=f[2].clone()
  new=[f[2],f[3],f[2],f[3]];nb=[bias[2],bias[3],bias[2],bias[3]];oldv=torch.stack([z@a.T+b for a,b in zip(f,bias)]).prod(0);newv=torch.stack([z@a.T+b for a,b in zip(new,nb)]).prod(0);oo=affine_moment(f+f,bias+bias);on=affine_moment(f+new,bias+nb);nn=affine_moment(new+new,nb+nb);scale=on/nn;err=oo-on.square()/nn;ref=(weight[:,None]*(oldv-newv*scale).square()).sum(0);q=(z@f[0].T+bias[0])*(z@f[1].T+bias[1]);G=gram(f[0],f[1],bias[0],bias[1]);greplay=float((G-q.T@(weight[:,None]*q)).norm()/G.norm());ereplay=float((err-ref).norm()/oo.norm());assert max(greplay,ereplay)<1e-10;rows.append(dict(seed=seed,gram_replay=greplay,edit_energy_replay=ereplay))
 return rows
