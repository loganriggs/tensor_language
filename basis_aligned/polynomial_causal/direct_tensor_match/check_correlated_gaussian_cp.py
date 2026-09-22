import itertools,json
from pathlib import Path
import numpy as np
import torch
from correlated_gaussian_cp import gram

def controls():
 torch.set_num_threads(2);a,w=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=4)));x=torch.tensor(a*2**.5,dtype=torch.float64)[ix];weights=torch.tensor(w/np.pi**.5,dtype=torch.float64)[ix].prod(1);rows=[]
 for seed in range(5):
  torch.manual_seed(35000+seed);f=[torch.randn(3,2,dtype=torch.float64,requires_grad=True) for _ in range(4)];g=[torch.randn(2,2,dtype=torch.float64) for _ in range(4)];b=[torch.randn(3,dtype=torch.float64) for _ in range(4)];c=[torch.randn(2,dtype=torch.float64) for _ in range(4)]
  if seed==1:b=[v*0 for v in b];c=[v*0 for v in c]
  if seed==2:g=[g[0]]*4
  if seed==3:b=[v*3 for v in b]
  if seed==4:g[1]=-g[0]
  for rho in [0.,.5,1.]:
   z=x[:,:2];zz=rho*z+(1-rho*rho)**.5*x[:,2:];p=torch.stack([z@v.T+bb for v,bb in zip(f,b)]).prod(0);q=torch.stack([zz@v.T+bb for v,bb in zip(g,c)]).prod(0);ref=p.T@(weights[:,None]*q);actual=gram(f,b,g,c,rho);ga=torch.autograd.grad(actual.sum(),f,retain_graph=True);gb=torch.autograd.grad(ref.sum(),f);rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-30)).detach());err=rel(actual,ref);ge=rel(torch.cat([v.flatten() for v in ga]),torch.cat([v.flatten() for v in gb]));assert max(err,ge)<1e-9;rows.append(dict(seed=seed,rho=rho,gram=err,gradient=ge))
 return rows
if __name__=='__main__':
 rows=controls();Path(__file__).with_name('CORRELATED_GAUSSIAN_CP_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print('15 rectangular paired Gram/gradient controls passed')
