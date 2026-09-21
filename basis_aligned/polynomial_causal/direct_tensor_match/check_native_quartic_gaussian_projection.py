"""Independent dense and quadrature checks for native Gaussian CP cross."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from native_quartic_gaussian_projection import project,cp_cross
from quartic_cp import directional
from gaussian_cp import gaussian_cp_gram
from quartic_cp_profile import profile,normalize_factors
P=Path(__file__).resolve().parent


def controls():
 torch.set_num_threads(2);rows=[];dtype=torch.float64;d=3
 nodes,weights=np.polynomial.hermite.hermgauss(5);ix=torch.tensor(list(itertools.product(range(5),repeat=d)));x=torch.tensor(nodes*2**.5,dtype=dtype)[ix];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ix].prod(1)
 for seed in range(5):
  torch.manual_seed(10600+seed);teacher=[torch.randn(*s,dtype=dtype) for s in [(2,4),(4,3),(4,3),(3,4),(4,d),(4,d)]]
  if seed==1:teacher[4][1]=teacher[4][0]
  if seed==2:teacher[0][1]=teacher[0][0]
  if seed==3:teacher[5]=teacher[4].clone()
  if seed==4:teacher[4][1]=teacher[4][0];teacher[5][1]=-teacher[5][0]
  mean,Q=project(teacher);idx=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)]);eye=torch.eye(d,dtype=dtype);H=directional(*teacher,[eye[idx[:,s]] for s in range(4)]).T.reshape(2,d,d,d,d)
  qref=6*torch.einsum('vijkk->vij',H);mref=3*torch.einsum('viijj->v',H)
  # Independent repeated-input teacher execution at exact polynomial quadrature.
  C,l,r,D,A,B=teacher;h=((x@A.T)*(x@B.T))@D.T;y=((h@l.T)*(h@r.T))@C.T
  p=[torch.randn(5,d,dtype=dtype,requires_grad=True) for _ in range(4)];f=normalize_factors(p);phi=torch.ones(len(x),5,dtype=dtype)
  for v in f:phi=phi*(x@v.T)
  cross=cp_cross(teacher,mean,Q,f);reference=y.T@(w[:,None]*phi)
  gram=gaussian_cp_gram(f,f);loss,c=profile(gram,cross,ridge=1e-6)
  dense=(w[:,None]*(phi@c.T-y).square()).sum()-(w[:,None]*y.square()).sum()+1e-6*c.square().sum()
  g1=torch.autograd.grad(loss,p,retain_graph=True);g2=torch.autograd.grad(dense,p)
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
  row=dict(seed=seed,mean=rel(mean,mref),quadratic=rel(Q,qref),cross=rel(cross,reference),gradient=rel(torch.cat([a.flatten() for a in g1]),torch.cat([a.flatten() for a in g2])))
  assert max(row[k] for k in ['mean','quadratic','cross','gradient'])<1e-8,row
  rows.append(row)
 return rows
if __name__=='__main__':
 rows=controls();(P/'NATIVE_GAUSSIAN_CP_CROSS_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
