"""Check normalized-metric profiled CP compression against explicit coefficients/quadrature."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from cp_parent_refit import prepare,objective
from quartic_cp import cp_entries as entries
from quartic_cp_profile import normalize_factors


def controls():
 torch.set_num_threads(2);dtype=torch.float64;nodes,weights=np.polynomial.hermite.hermgauss(5);ids=torch.tensor(list(itertools.product(range(5),repeat=3)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ids];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ids].prod(1);indices=torch.tensor(list(itertools.product(range(3),repeat=4)));rows=[]
 for seed,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(12900+seed);teacher=normalize_factors([torch.randn(3,3,dtype=dtype) for _ in range(4)]);C=torch.randn(2,3,dtype=dtype)
  if family=='shared_input':teacher[1]=teacher[0].clone()
  if family=='shared_output':C[1]=.5*C[0]
  if family=='squares':teacher[2:]=[a.clone() for a in teacher[:2]]
  if family=='cancellation':
   for a in teacher:a[1]=a[0]
   C[:,1]=-C[:,0]
  raw=[torch.randn(4,3,dtype=dtype,requires_grad=True) for _ in range(4)];student=normalize_factors(raw);S=torch.diag(torch.tensor([.7,1.1,1.3],dtype=dtype));mu=torch.randn(3,dtype=dtype)*.3;lam=13.;parent=prepare(teacher,C,S,mu,lam)
  loss,c,info=objective(parent,student);grad=torch.autograd.grad(loss,raw,retain_graph=True)
  x=z@S.T+mu
  def vals(f):
   v=torch.ones(len(x),f[0].shape[0],dtype=dtype)
   for a in f:v=v*(x@a.T)
   return v
  # Identity coefficients expose each atom in entry-by-atom layout.
  phi0=entries(torch.eye(4,dtype=dtype),student,indices);target0=entries(C,teacher,indices)
  phi1=vals(student);target1=vals(teacher)@C.T
  g=(phi1.T@(w[:,None]*phi1)+lam*phi0.T@phi0)/(1+lam);cross=(target1.T@(w[:,None]*phi1)+lam*target0.T@phi0)/(1+lam);scale=g.diag().sqrt();unitg=g/(scale[:,None]*scale[None,:]);unitx=cross/scale
  cc=torch.linalg.solve(unitg+1e-10*torch.eye(4,dtype=dtype),unitx.T).T.detach();ref=(cc*(cc@unitg)).sum()-2*(cc*unitx).sum()+1e-10*cc.square().sum();refgrad=torch.autograd.grad(ref,raw)
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-30)).detach())
  ge=rel(torch.cat([a.flatten() for a in grad]),torch.cat([a.flatten() for a in refgrad]));ve=abs(float((loss-ref).detach()))/(1+abs(float(ref.detach())));ce=rel(c,cc/scale);assert ge<1e-9 and ve<1e-10 and ce<1e-9
  explicit_energy=((target1.square()*w[:,None]).sum()+lam*target0.square().sum())/(1+lam);ee=rel(info['energy'],explicit_energy);assert ee<1e-10
  rows.append(dict(family=family,gradient_error=ge,loss_error=ve,readout_error=ce,teacher_energy_error=ee))
 return rows
if __name__=='__main__':
 rows=controls();Path(__file__).with_name('CP_PARENT_REFIT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
