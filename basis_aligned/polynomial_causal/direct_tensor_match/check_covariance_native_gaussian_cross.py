"""Covariance whitening of native Gaussian CP cross and a support redteam."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from native_quartic_gaussian_projection import project,cp_cross
from quartic_cp_profile import normalize_factors,profile
from gaussian_cp import gaussian_cp_gram
P=Path(__file__).resolve().parent


def main():
 torch.set_num_threads(2);dtype=torch.float64;rows=[];d=3
 nodes,weights=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=d)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ix];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ix].prod(1)
 for seed in range(5):
  torch.manual_seed(10800+seed);teacher=[torch.randn(*s,dtype=dtype) for s in [(2,4),(4,3),(4,3),(3,4),(4,d),(4,d)]]
  if seed==1:teacher[4][1]=teacher[4][0]
  if seed==2:teacher[0][1]=teacher[0][0]
  if seed==3:teacher[5]=teacher[4].clone()
  if seed==4:teacher[4][1]=teacher[4][0];teacher[5][1]=-teacher[5][0]
  raw=torch.randn(d,d,dtype=dtype);M=raw@raw.T+.2*torch.eye(d,dtype=dtype);S=torch.linalg.cholesky(M);x=z@S.T
  transformed=teacher[:4]+[teacher[4]@S,teacher[5]@S];m,Q=project(transformed)
  params=[torch.randn(5,d,dtype=dtype,requires_grad=True) for _ in range(4)];f=normalize_factors(params);fw=[a@S for a in f]
  cross=cp_cross(transformed,m,Q,fw);gram=gaussian_cp_gram(f,f,M);whitegram=gaussian_cp_gram(fw,fw)
  C,l,r,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;y=((h@l.T)*(h@r.T))@C.T;phi=torch.ones(len(x),5,dtype=dtype)
  for a in f:phi=phi*(x@a.T)
  reference=y.T@(w[:,None]*phi);loss,c=profile(gram,cross,ridge=1e-6);direct=(w[:,None]*(phi@c.T-y).square()).sum()-(w[:,None]*y.square()).sum()+1e-6*c.square().sum()
  g1=torch.autograd.grad(loss,params,retain_graph=True);g2=torch.autograd.grad(direct,params)
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
  row=dict(seed=seed,cross=rel(cross,reference),gram_whitening=rel(gram,whitegram),gradient=rel(torch.cat([a.flatten() for a in g1]),torch.cat([a.flatten() for a in g2])))
  assert max(row[k] for k in ['cross','gram_whitening','gradient'])<1e-8,row
  rows.append(row)
 # f=x0^4+x1^4, approximation=x0^4. Full-support but anisotropic laws.
 factors=[torch.eye(2,dtype=dtype) for _ in range(4)];teacher=torch.tensor([[1.,1.]],dtype=dtype);delta=torch.tensor([[0.,1.]],dtype=dtype);redteam=[]
 for epsilon in [1.,.1,.01,.001,0.]:
  covariance=torch.diag(torch.tensor([1.,epsilon],dtype=dtype));G=gaussian_cp_gram(factors,factors,covariance)
  error=float(((delta@G@delta.T)/(teacher@G@teacher.T)).sqrt())
  expected=(105*epsilon**4/(105+18*epsilon**2+105*epsilon**4))**.5
  assert abs(error-expected)<1e-12
  redteam.append(dict(second_input_variance=epsilon,weighted_gaussian_relative_error=error,isotropic_relative_error=(105/228)**.5))
 result=dict(controls=rows,anisotropy_redteam=redteam,scope='Exact N(0,M) cross via whitening, not general empirical lifted eighth moment or noncentral Gaussian. Singular M loses global equivalence; positive-definite but ill-conditioned M can hide large errors outside emphasized directions.')
 (P/'COVARIANCE_NATIVE_GAUSSIAN_CROSS_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
