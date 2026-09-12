"""Matched ten-start dense planted solver comparison, no model/data fit.
Compares TRF nonlinear least squares and L-BFGS on the SAME variable-projected
coefficient residual, raw versus invertibly equivalent secant coordinates.
No global-recovery or native-scalability conclusion follows from this toy.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares,minimize
from cubic_secant_coordinates_v1 import encode,features
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
 out=P/'CUBIC_SECANT_SOLVER_V1_CONTROL.json';assert not out.exists()
 torch.manual_seed(9122011);base=torch.randn(3,4);base=base/base.norm(dim=-1,keepdim=True)
 direction=torch.randn(3,4);direction=direction/direction.norm();sep=.03
 planted=torch.stack((base+sep*direction,base-sep*direction));theta,t=encode(planted,(0,1));target=features(theta,t).T
 target=target/target.norm(dim=0,keepdim=True);scale=target.norm()
 # Dense polynomial coefficient objective includes all repeated-input permutations.
 def residual(x,mode,t):
  bank=features(x.reshape(2,3,4),t if mode=='secant' else None).T
  q,r=torch.linalg.qr(bank,mode='reduced')
  return ((q@(q.T@target)-target)/scale).flatten()
 controls=[];reports=[];tic=time.perf_counter()
 for seed in range(10):
  torch.manual_seed(9122020+seed);start=torch.randn(2,3,4);start=start/start.norm(dim=-1,keepdim=True)
  sec,step=encode(start,(0,1))
  r0=residual(start,'raw',step);r1=residual(sec,'secant',step)
  controls.append(float((r0-r1).abs().max()))
  for mode,x0 in [('raw',start),('secant',sec)]:
   def fun(x):return residual(torch.from_numpy(x),'raw' if mode=='raw' else 'secant',step).detach().numpy()
   def jac(x):
    z=torch.from_numpy(x).requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda y:residual(y,mode,step),z,vectorize=True).detach().numpy()
   for method in ['lbfgs','trf']:
    started=time.perf_counter()
    if method=='trf':
     fit=least_squares(fun,x0.numpy().ravel(),jac=jac,method='trf',max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-9)
     grad=float(fit.optimality);evals=fit.nfev
    else:
     def objective(x):
      z=torch.from_numpy(x).requires_grad_(True);r=residual(z,mode,step);v=(r*r).sum()/2;g=torch.autograd.grad(v,z)[0]
      return float(v.detach()),g.detach().numpy()
     fit=minimize(objective,x0.numpy().ravel(),jac=True,method='L-BFGS-B',options={'maxiter':250,'maxls':30,'gtol':1e-9,'ftol':1e-14})
     grad=float(np.max(np.abs(fit.jac)));evals=fit.nfev
    rr=fun(fit.x);error=float(np.linalg.norm(rr))
    row=dict(seed=seed,coordinates=mode,solver=method,relative_coefficient_error=error,recovered=error<=1e-5,stationary=grad<=1e-8,gradient_inf=grad,evaluations=evals,seconds=time.perf_counter()-started,termination=str(fit.message))
    reports.append(row);print(json.dumps(row),flush=True)
 summary=[]
 for mode in ['raw','secant']:
  for solver in ['lbfgs','trf']:
   rows=[r for r in reports if r['coordinates']==mode and r['solver']==solver]
   summary.append(dict(coordinates=mode,solver=solver,recovered=sum(r['recovered'] for r in rows),stationary=sum(r['stationary'] for r in rows),median_seconds=float(np.median([r['seconds'] for r in rows])),best_error=min(r['relative_coefficient_error'] for r in rows)))
 result=dict(instrument_pass=max(controls)<=1e-10,max_initial_residual_difference=max(controls),reports=reports,summary=summary,seconds=time.perf_counter()-tic,scope='Ten identical random source banks per arm; same rank2 symmetric cubic source span and exact output elimination. Dense 4D synthetic coefficient target only, not native circuit recovery or global optimality. Fixed budgets; solver termination and stationarity recorded separately.')
 out.write_text(json.dumps(result,indent=2)+'\n');assert result['instrument_pass']
if __name__=='__main__':main()
