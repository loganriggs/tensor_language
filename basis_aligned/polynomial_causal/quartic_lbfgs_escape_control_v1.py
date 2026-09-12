"""Four independent starts of a planted two-output quadratic-square program.

A constraints/monotonicity <=1e-10; B all relative errors <=1e-4;
C all projected gradients <=1e-6. 1000 steps /30 seconds each, CPU only.
"""
from pathlib import Path
import json
import torch
from coupled_quartic_writer_v1 import gram
from quartic_manifold_lbfgs_v1 import fit, tangent, retract

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
torch.manual_seed(91801)
tb=torch.linalg.qr(torch.randn(2,8,2),mode='reduced')[0]
tn=torch.tensor([[.8,-.6],[.6,.8]])
ta=torch.tensor([[1.,.3],[-.2,.9]])
target_energy=float((ta*(gram(tb,tn)@ta)).sum())
def evaluate(b,n,divisor,gradient=False):
    if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
    with torch.set_grad_enabled(gradient):
        allk=gram(torch.cat([b,tb]),torch.cat([n,tn]));k=allk[:2,:2];c=allk[:2,2:]@ta
        with torch.no_grad():a=torch.linalg.solve(k,c)
        loss=(target_energy+(a*(k@a)).sum()-2*(a*c).sum())/divisor
        if gradient:
            gb,gn=torch.autograd.grad(loss,(b,n));gb,gn=tangent(b,n,gb,gn)
            return float(loss),a,gb.detach(),gn.detach()
        return float(loss),a

torch.manual_seed(91814)
b=torch.linalg.qr(torch.randn_like(tb),mode='reduced')[0]
n=torch.randn_like(tn);n=n/n.norm(dim=-1,keepdim=True)
b,n,a,h,reason=fit(b,n,evaluate,target_energy,max_steps=1000,max_seconds=30)
initial_error=max(0.,h[-1]['objective'])**.5
prior=json.loads((P/'QUARTIC_LBFGS_INDEPENDENT_V1_CONTROL.json').read_text())['reports'][-1]
assert abs(initial_error-prior['relative_coefficient_error'])<1e-10
reports=[]
for scale in (.01,.1,.5):
 for seed in range(91821,91825):
  torch.manual_seed(seed)
  db,dn=tangent(b,n,torch.randn_like(b),torch.randn_like(n))
  size=(db.square().sum()+dn.square().sum()).sqrt();db/=size;dn/=size
  bn,nn=retract(b,n,db,dn,scale)
  bn,nn,an,hh,reason=fit(bn,nn,evaluate,target_energy,max_steps=1000,max_seconds=30)
  r=dict(scale=scale,seed=seed,error=max(0.,hh[-1]['objective'])**.5,gradient=hh[-1]['projected_gradient_norm'],termination=reason,iterations=len(hh))
  reports.append(r);print(json.dumps(r),flush=True)
result=dict(initial_error=initial_error,reports=reports,pred_repair=any(r['error']<=1e-4 for r in reports),scope='Perturbation recovery on one failed planted start; no classification of its stationary point and no native/global guarantee.')
out=P/'QUARTIC_LBFGS_ESCAPE_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
