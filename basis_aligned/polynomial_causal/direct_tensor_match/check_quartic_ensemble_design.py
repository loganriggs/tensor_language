"""Check single-family equivalence and duplication invariance, including gradients."""
import json
from pathlib import Path
import torch
from quartic_finite_response import joint_design,ensemble_design,fit

def main():
 torch.set_num_threads(2);rows=[]
 for seed in range(5):
  torch.manual_seed(7260+seed);x=torch.randn(23,seed+3,dtype=torch.float64);delta=torch.randn_like(x);y=torch.randn(23,4,dtype=torch.float64);r=torch.randn_like(y);u=torch.randn(3,2,x.shape[1],dtype=x.dtype,requires_grad=True);v=torch.randn_like(u,requires_grad=True);c=torch.randn(6,4,dtype=x.dtype)
  a,b=joint_design(x,y,delta,r,u,v,.7);loss=(a@c-b).square().sum();ag=torch.autograd.grad(loss,(u,v),retain_graph=True)
  for repeats in [1,4]:
   aa,bb=ensemble_design(x,y,delta[None].repeat(repeats,1,1),r[None].repeat(repeats,1,1),u,v,.7);other=(aa@c-bb).square().sum();bg=torch.autograd.grad(other,(u,v),retain_graph=True)
   row=dict(seed=seed,repeats=repeats,objective_error=float(abs((other-loss)/loss).detach()),gradient_error=max(float((a-b).norm()/a.norm()) for a,b in zip(ag,bg)));rows.append(row)
 assert max(max(r['objective_error'],r['gradient_error']) for r in rows)<1e-12
 # Executes the same optional callback interface as the future native fit.
 info,p=fit(x,y,delta[None].repeat(4,1,1),r[None].repeat(4,1,1),(u.detach(),v.detach()),steps=2,design_builder=ensemble_design);assert p['C'].shape==(4,6)
 result=dict(controls=rows,callback_objective=info['objective'],scope='Algebra/objective/gradient smoke; not toy recovery or native fit.')
 Path(__file__).with_name('QUARTIC_ENSEMBLE_DESIGN_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
