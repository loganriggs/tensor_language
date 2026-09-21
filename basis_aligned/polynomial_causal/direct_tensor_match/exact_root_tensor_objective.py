"""Exact symmetric-quartic coefficient objective for shared quadratic roots.
Teacher constant omitted from optimization; no input probes in self/cross terms.
"""
import torch
from shared_quadratic_bank import bank_gram,native_bank_cross,bank_entries,normalize_bank
from quartic_cp import directional

def objective(teacher,u,v,ridge=1e-6,profiled=True):
 gram=bank_gram(u,v);cross=native_bank_cross(teacher,u,v);eye=torch.eye(len(gram),dtype=gram.dtype,device=gram.device)
 if profiled:
  with torch.no_grad():coeff=torch.linalg.solve(gram+ridge*eye,cross.T).T
 else:coeff=torch.linalg.solve(gram+ridge*eye,cross.T).T
 value=(coeff*(coeff@gram)).sum()-2*(coeff*cross).sum()+ridge*coeff.square().sum()
 return value,coeff,gram,cross

def controls():
 torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for seed,family in enumerate(['independent','shared_inputs','shared_outputs','squares','cancellation']):
  torch.manual_seed(660+seed);d=4
  teacher=[torch.randn(3,5,dtype=dtype)/3,torch.randn(5,4,dtype=dtype)/3,torch.randn(5,4,dtype=dtype)/3,torch.randn(4,6,dtype=dtype)/3,torch.randn(6,d,dtype=dtype)/3,torch.randn(6,d,dtype=dtype)/3]
  u=torch.randn(3,2,d,dtype=dtype);v=torch.randn_like(u)
  if family=='shared_inputs':u[1]=u[0];v[1]=v[0]
  if family=='shared_outputs':teacher[0][1]=teacher[0][0]
  if family=='squares':v=u.clone()
  if family=='cancellation':u[1]=u[0];v[1]=-v[0]
  u.requires_grad_();v.requires_grad_();uu,vv=normalize_bank(u,v);loss,c,g,cross=objective(teacher,uu,vv);full,_,_,_=objective(teacher,uu,vv,profiled=False)
  indices=torch.cartesian_prod(*[torch.arange(d) for _ in range(4)]);eye=torch.eye(d,dtype=dtype);h=directional(*teacher,[eye[indices[:,i]] for i in range(4)]);phi=bank_entries(uu,vv,indices);gd=phi.T@phi;xd=h.T@phi
  dense=(phi@c.T-h).square().sum()+1e-6*c.square().sum()-h.square().sum();gram_error=float((g-gd).norm()/gd.norm());cross_error=float((cross-xd).norm()/xd.norm());loss_error=float(abs(loss-dense)/h.square().sum());ga=torch.autograd.grad(loss,(u,v),retain_graph=True);gb=torch.autograd.grad(full,(u,v),retain_graph=True);gc=torch.autograd.grad(dense,(u,v));gradient_error=max(float((a-b).norm()/b.norm().clamp_min(1e-30)) for a,b in zip(ga,gb));dense_gradient_error=max(float((a-b).norm()/b.norm().clamp_min(1e-30)) for a,b in zip(ga,gc));assert max(gram_error,cross_error,loss_error,gradient_error,dense_gradient_error)<1e-8
  rows.append(dict(family=family,gram_error=gram_error,cross_error=cross_error,loss_error=loss_error,profile_gradient_error=gradient_error,dense_gradient_error=dense_gradient_error))
 return rows
if __name__=='__main__':
 import json
 from pathlib import Path
 rows=controls();Path(__file__).with_name('EXACT_ROOT_TENSOR_OBJECTIVE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
