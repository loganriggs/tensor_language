"""Four independent starts of a planted two-output quadratic-square program.

A constraints/monotonicity <=1e-10; B all relative errors <=1e-4;
C all projected gradients <=1e-6. 1000 steps /30 seconds each, CPU only.
"""
from pathlib import Path
import json
import torch
from coupled_quartic_writer_v1 import gram
from quartic_manifold_lbfgs_v1 import fit, tangent

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
reports=[]
for seed in range(91811,91815):
    torch.manual_seed(seed)
    b=torch.linalg.qr(torch.randn_like(tb),mode='reduced')[0]
    n=torch.randn_like(tn);n=n/n.norm(dim=-1,keepdim=True)
    b,n,a,h,reason=fit(b,n,evaluate,target_energy,max_steps=1000,max_seconds=30)
    error=max(0.,h[-1]['objective'])**.5
    constraint=max(float((b.transpose(-1,-2)@b-torch.eye(2)).abs().max()),float((n.norm(dim=-1)-1).abs().max()))
    monotonic=all(z['objective']<=h[i]['objective']+1e-10 for i,z in enumerate(h[1:]))
    r=dict(seed=seed,relative_coefficient_error=error,constraint_error=constraint,monotonic=monotonic,
           termination=reason,iterations=len(h),seconds=h[-1]['seconds'],gradient=h[-1]['projected_gradient_norm'])
    reports.append(r);print(json.dumps(r),flush=True)
result=dict(pred_a=all(r['constraint_error']<=1e-10 and r['monotonic'] for r in reports),
            pred_b=all(r['relative_coefficient_error']<=1e-4 for r in reports),
            pred_c=all(r['gradient']<=1e-6 for r in reports),reports=reports,
            scope='Four independent starts on one realizable two-node planted target; no native/global recovery guarantee.')
out=P/'QUARTIC_LBFGS_INDEPENDENT_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');assert result['pred_a']
