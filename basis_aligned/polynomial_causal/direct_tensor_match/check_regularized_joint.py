from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from regularized_joint_private import RegularizedJointMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 T,bases,private,_,_=fixture(case);oracle=list(bases)+list(private);rng=torch.Generator().manual_seed(46000+case)
 for rho in (1e-6,1e-4,.01):
  metric=RegularizedJointMetric(T,[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS],rho)
  params=[(v+.1*torch.randn(v.shape,dtype=v.dtype,generator=rng)).requires_grad_() for v in oracle]
  losses=[metric.loss(params)[0],metric.loss(params,dense=True)[0],metric.loss(params,solve_through=True)[0]];grads=[torch.autograd.grad(v,params) for v in losses]
  lr=max(abs(float((losses[0]-v).detach())) for v in losses[1:]);gr=max(float((a-b).norm()/(1+a.norm())) for gs in grads[1:] for a,b in zip(grads[0],gs));assert lr<1e-8 and gr<1e-8
  direction=[torch.randn(v.shape,dtype=v.dtype,generator=rng) for v in params];eps=1e-6
  fd=float((metric.loss([v+eps*u for v,u in zip(params,direction)])[0]-metric.loss([v-eps*u for v,u in zip(params,direction)])[0]).detach()/(2*eps));ad=sum(float((g*u).sum()) for g,u in zip(grads[0],direction));assert abs(fd-ad)<1e-6
  with torch.no_grad():oracle_bias=float(metric.reconstruction(oracle).sqrt())
  rows.append(dict(case=case,rho=rho,loss_replay=lr,gradient_replay=gr,finite_difference=abs(fd-ad),regularized_oracle_reconstruction_error=oracle_bias))
# Exact overlap is well-defined with positive rho, even though the old chart
# properly rejected it without regularization.
T,bases,private,_,_=fixture(0);ps=list(bases)+list(private)
for j,(a,b) in enumerate(GROUPS):ps[3+j]=ps[3+j].clone();ps[3+j][:,0]=bases[a][:,0]
metric=RegularizedJointMetric(T,[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS],1e-4);a=metric.loss(ps,dense=True)[0];b=metric.loss(ps,solve_through=True)[0];assert torch.isfinite(a) and abs(float(a-b))<1e-8
(P/'REGULARIZED_JOINT_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,exact_overlap_control=True,scope='Explicit penalized objective, independent full loss and differentiable normal solve; unpenalized reconstruction recorded separately.'),indent=2)+'\n');print('PASS fifteen penalized envelope/solve controls and exact-overlap control')
