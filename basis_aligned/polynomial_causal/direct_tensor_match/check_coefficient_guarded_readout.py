"""Compare spectral constrained readout with independent SLSQP and KKT checks."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from coefficient_guarded_readout import GuardedReadout
P=Path(__file__).resolve().parent


def controls():
 torch.set_num_threads(2);dtype=torch.float64;rows=[]
 for seed in range(5):
  torch.manual_seed(11100+seed);a=torch.randn(12,5,dtype=dtype);b=torch.randn(14,5,dtype=dtype)
  if seed==1:a[:,1]=a[:,0]
  if seed==2:b[:,1]=b[:,0]
  if seed==3:a=a@torch.diag(torch.tensor([1.,.1,.01,.001,.0001],dtype=dtype))
  if seed==4:b[:,1]=-b[:,0]
  y=torch.randn(12,3,dtype=dtype);z=torch.randn(14,3,dtype=dtype)
  G0=a.T@a+.1*torch.eye(5,dtype=dtype);X0=y.T@a;G=b.T@b+.1*torch.eye(5,dtype=dtype);X=z.T@b
  solver=GuardedReadout(G0,X0,G,X);previous=None
  for ratio in [0.,.01,.1,1.,10.]:
   budget=ratio*solver.capture;C,info=solver.solve(budget);objective=float((C*(C@G)).sum()-2*(C*X).sum())
   assert info['displacement']<=budget+1e-9*(1+budget)
   if previous is not None:assert objective<=previous+1e-9
   previous=objective
   if ratio==0:
    rows.append(dict(seed=seed,ratio=ratio,endpoint_error=float((C-solver.C0).norm())));continue
   g0,x0,g,x,c0=[v.numpy() for v in [G0,X0,G,X,solver.C0]]
   def fun(flat):
    c=flat.reshape(3,5);return np.sum(c*(c@g))-2*np.sum(c*x)
   def jac(flat):return (2*(flat.reshape(3,5)@g-x)).ravel()
   def con(flat):
    delta=flat.reshape(3,5)-c0;return budget-np.sum(delta*(delta@g0))
   def cjac(flat):return (-2*((flat.reshape(3,5)-c0)@g0)).ravel()
   result=minimize(fun,c0.ravel(),jac=jac,constraints={'type':'ineq','fun':con,'jac':cjac},method='SLSQP',options={'ftol':1e-11,'maxiter':1000})
   cerror=float(np.linalg.norm(C.numpy()-result.x.reshape(3,5))/(1+np.linalg.norm(C.numpy())))
   gap=abs(objective-result.fun)/(1+abs(objective))
   assert gap<1e-7 and cerror<1e-5 and info['stationarity']<1e-10,(seed,ratio,gap,cerror,info)
   rows.append(dict(seed=seed,ratio=ratio,objective_gap=gap,coefficient_error=cerror,slsqp_success=bool(result.success),**info))
 return rows
if __name__=='__main__':
 rows=controls();(P/'COEFFICIENT_GUARDED_READOUT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),max_objective_gap=max(r.get('objective_gap',0) for r in rows),max_coefficient_error=max(r.get('coefficient_error',0) for r in rows))))
