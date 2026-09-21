from pathlib import Path
import json,torch
from empirical_pair_metric import prepare,solve
P=Path(__file__).parent;torch.set_num_threads(2);checks=[]
for seed in range(5):
 g=torch.Generator().manual_seed(9320+seed);dtype=torch.float64;rand=lambda *shape:torch.randn(*shape,dtype=dtype,generator=g)
 T=rand(2,8,8);T=(T+T.transpose(-1,-2))/2;La,Ra,Lb,Rb=rand(8,4),rand(8,4),rand(8,5),rand(8,5);x=rand(64,8);K=torch.eye(8,dtype=dtype);ca,cb=rand(64),rand(64)
 for family in ['source','downstream']:
  a=prepare(T[0],T[1],La,Ra,Lb,Rb,x,K,ca,cb,family);w,ridge,res=solve(a,.1);w=w.detach().requires_grad_();rawa=(La*w[:4])@Ra.T;rawb=(Lb*w[4:])@Rb.T;hat=torch.stack([(rawa+rawa.T)/2,(rawb+rawb.T)/2]);coef=(hat-T).square().sum()/T.square().sum();emp=(a['design']@w-a['target']).square().sum()/a['divisor'];loss=coef+.1*emp+ridge*w.square().sum();grad=torch.autograd.grad(loss,w)[0];replay=float(grad.abs().max());assert max(res,replay)<1e-10
  # Dense source residual agrees with the centered feature design.
  E=hat-T;centered=torch.einsum('ni,oij,nj->no',x,E,x)-torch.einsum('ij,oji->o',K,E)
  explicit=(centered.square().sum(1).mean()/((torch.einsum('ni,oij,nj->no',x,T,x)-torch.einsum('ij,oji->o',K,T)).square().sum(1).mean())) if family=='source' else (ca*centered[:,0]+cb*centered[:,1]).square().mean()
  mismatch=abs(float(emp-explicit));assert mismatch<1e-10
  checks.append(dict(seed=seed,family=family,normal_equation_replay=res,dense_gradient_replay=replay,empirical_design_replay=mismatch))
(P/'EMPIRICAL_PAIR_METRIC_PREFLIGHT_V1.json').write_text(json.dumps(checks,indent=2)+'\n');print('PASS',len(checks),'checks')
