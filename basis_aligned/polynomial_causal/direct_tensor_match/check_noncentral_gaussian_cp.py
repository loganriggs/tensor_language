"""Full covariance and nonzero mean checked by exact eighth-degree quadrature."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from noncentral_gaussian_cp import project_shifted,cross,gram,PARTITIONS
from quartic_cp_profile import normalize_factors,profile
P=Path(__file__).resolve().parent


def controls():
 torch.set_num_threads(2);dtype=torch.float64;rows=[];d=3
 nodes,weights=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=d)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ix];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ix].prod(1)
 for seed in range(5):
  torch.manual_seed(10900+seed);teacher=[torch.randn(*s,dtype=dtype) for s in [(2,4),(4,3),(4,3),(3,4),(4,d),(4,d)]]
  if seed==1:teacher[4][1]=teacher[4][0]
  if seed==2:teacher[0][1]=teacher[0][0]
  if seed==3:teacher[5]=teacher[4].clone()
  if seed==4:teacher[4][1]=teacher[4][0];teacher[5][1]=-teacher[5][0]
  raw=torch.randn(d,d,dtype=dtype);M=raw@raw.T+.2*torch.eye(d,dtype=dtype);S=torch.linalg.cholesky(M);mu=torch.randn(d,dtype=dtype);x=z@S.T+mu
  transformed=teacher[:4]+[teacher[4]@S,teacher[5]@S];location=torch.linalg.solve(S,mu);projection=project_shifted(transformed,location)
  params=[torch.randn(3,d,dtype=dtype,requires_grad=True) for _ in range(4)];f=normalize_factors(params);fw=[a@S for a in f];bias=[a@mu for a in f]
  X=cross(transformed,location,projection,fw,bias);G=gram(fw,bias,fw,bias)
  C,l,r,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;y=((h@l.T)*(h@r.T))@C.T;phi=torch.ones(len(x),3,dtype=dtype)
  for a in f:phi=phi*(x@a.T)
  xref=y.T@(w[:,None]*phi);gref=phi.T@(w[:,None]*phi);loss,c=profile(G,X,ridge=1e-6);direct=(w[:,None]*(phi@c.T-y).square()).sum()-(w[:,None]*y.square()).sum()+1e-6*c.square().sum()
  ga=torch.autograd.grad(loss,params,retain_graph=True);gb=torch.autograd.grad(direct,params)
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
  row=dict(seed=seed,mean=rel(projection[0],(w[:,None]*y).sum(0)),cross=rel(X,xref),gram=rel(G,gref),gradient=rel(torch.cat([a.flatten() for a in ga]),torch.cat([a.flatten() for a in gb])))
  assert max(row[k] for k in ['mean','cross','gram','gradient'])<1e-8,row
  rows.append(row)
 assert len(PARTITIONS[8])==764
 return rows
if __name__=='__main__':
 rows=controls();(P/'NONCENTRAL_GAUSSIAN_CP_CONTROLS_V1.json').write_text(json.dumps(dict(partial_matchings=764,controls=rows,scope='Exact Gaussian N(mu,Sigma) native teacher/CP Gram and cross, independent tensor-product quadrature and gradients; not actual empirical eighth moments.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
