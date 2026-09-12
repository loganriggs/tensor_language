from pathlib import Path
import torch,json
from quartic_manifold_lbfgs_v1 import fit,tangent,retract
from coupled_quartic_writer_v1 import gram
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91701)
trueb=torch.linalg.qr(torch.randn(1,5,2),mode='reduced')[0];truen=torch.tensor([[.8,-.6]])
b,n=retract(trueb,truen,torch.randn_like(trueb),torch.randn_like(truen),.05)
def evaluate(b,n,divisor,gradient=False):
 if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
 with torch.set_grad_enabled(gradient):
  k=gram(torch.cat([b,trueb]),torch.cat([n,truen]));a=(k[0,1]/k[0,0]).detach();loss=(k[1,1]-2*a*k[0,1]+a*a*k[0,0])/divisor
  if gradient:
   gb,gn=torch.autograd.grad(loss,(b,n));gb,gn=tangent(b,n,gb,gn)
   return float(loss),a,gb.detach(),gn.detach()
  return float(loss),a
b,n,a,h,reason=fit(b,n,evaluate,1.,max_steps=300,max_seconds=30)
err=max(0.,h[-1]['objective'])**.5
orth=float((b.transpose(-1,-2)@b-torch.eye(2)).abs().max());sphere=float((n.norm(dim=-1)-1).abs().max())
r=dict(pred_a=orth<=1e-10 and sphere<=1e-10 and all(z['objective']<=h[i]['objective']+1e-12 for i,z in enumerate(h[1:])),pred_b=err<=1e-4,pred_c=len(h)<195,coefficient_error=err,orthogonality_error=orth,sphere_error=sphere,termination=reason,iterations=len(h),final_gradient=h[-1]['projected_gradient_norm'])
p=P/'QUARTIC_MANIFOLD_LBFGS_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a'] and r['pred_b'] and r['pred_c']
