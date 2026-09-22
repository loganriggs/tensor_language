"""Independent quadrature and dense symmetric coefficients check residual gradients."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from local_quartic_residual import objective
from noncentral_gaussian_cp import project_shifted
from quartic_cp import directional
from quartic_cp_profile import normalize_factors
P=Path(__file__).resolve().parent

def dense_values(teacher,x):
 C,l,r,D,L,R=teacher;h=((x@L.T)*(x@R.T))@D.T;return ((h@l.T)*(h@r.T))@C.T

def controls():
 torch.set_num_threads(2);dtype=torch.float64;a,w=np.polynomial.hermite.hermgauss(5);z=torch.tensor(list(itertools.product(a*2**.5,repeat=3)),dtype=dtype);qw=torch.tensor([a*b*c for a,b,c in itertools.product(w/np.pi**.5,repeat=3)],dtype=dtype);rows=[]
 for seed,name in enumerate(['generic','square','shared_first_layer','signed_outputs','zero_parent']):
  torch.manual_seed(23000+seed);t=[torch.randn(*s,dtype=dtype) for s in [(2,3),(3,2),(3,2),(2,4),(4,3),(4,3)]]
  if seed==1:t[2]=t[1]
  if seed==2:t[5]=t[4]
  if seed==3:t[0][1]=-t[0][0]
  pf=normalize_factors([torch.randn(3,3,dtype=dtype) for _ in range(4)]);pc=torch.randn(2,3,dtype=dtype)
  if seed==4:pc.zero_()
  M=torch.randn(3,3,dtype=dtype);S=M@M.T/3+torch.eye(3,dtype=dtype)*.5;mu=torch.randn(3,dtype=dtype);loc=torch.linalg.solve(S,mu);tr=t[:4]+[t[4]@S,t[5]@S];projection=project_shifted(tr,loc);x=z@S.T+mu
  residual=dense_values(t,x)-torch.stack([x@f.T for f in pf]).prod(0)@pc.T
  pars=[torch.randn(4,3,dtype=dtype,requires_grad=True) for _ in range(4)];fs=normalize_factors(pars);weights=torch.tensor([.3,1.7],dtype=dtype)
  for lam in [0.,.1]:
   loss,C,info=objective(t,tr,loc,projection,S,mu,pf,pc,fs,2,weights,lam)
   pred=torch.stack([x@f.T for f in fs]).prod(0)@C.T
   independent=(qw[:,None]*weights*(pred.square()-2*pred*residual)).sum()
   if lam:
    # Enumerate all ordered coefficient slots via polarization and symmetrization.
    ids=list(itertools.product(range(3),repeat=4));vectors=[torch.eye(3,dtype=dtype)[[i[j] for i in ids]] for j in range(4)]
    teachercoef=directional(*t,vectors)
    def cpcoeff(fs,C):
     terms=[]
     for idx in ids:
      terms.append(sum(torch.prod(torch.stack([fs[j][:,idx[p[j]]] for j in range(4)]),0) for p in itertools.permutations(range(4)))/24)
     return torch.stack(terms)@C.T
    basecoef=cpcoeff(pf,pc);candidatecoef=cpcoeff(fs,C);independent+=lam*(weights*(candidatecoef.square()-2*candidatecoef*(teachercoef-basecoef))).sum()
   independent=independent/(1+lam)+1e-6*(weights[:,None]*C.square()).sum()
   a=torch.autograd.grad(loss,pars,retain_graph=True);b=torch.autograd.grad(independent,pars,retain_graph=True)
   le=float((loss-independent).abs()/independent.abs().clamp_min(1));ge=max(float((u-v).norm()/v.norm().clamp_min(1e-20)) for u,v in zip(a,b));assert le<1e-9 and ge<1e-8,(name,lam,le,ge)
   rows.append(dict(family=name,coefficient_weight=lam,loss_error=le,gradient_error=ge,**info))
 return rows
if __name__=='__main__':
 rows=controls();(P/'LOCAL_QUARTIC_RESIDUAL_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
