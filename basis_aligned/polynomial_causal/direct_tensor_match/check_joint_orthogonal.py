from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from joint_orthogonal_private import JointOrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 T,bases,private,_,_=fixture(case);shared=[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS];metric=JointOrthogonalPrivateMetric(T,shared);oracle=list(bases)+list(private)
 oracle_error=float(metric.loss(oracle,dense=True)[0]);assert oracle_error<1e-20
 rng=torch.Generator().manual_seed(44000+case);params=[(v+.1*torch.randn(v.shape,dtype=v.dtype,generator=rng)).requires_grad_() for v in oracle]
 losses=[metric.loss(params)[0],metric.loss(params,dense=True)[0],metric.loss(params,solve_through=True)[0]];grads=[torch.autograd.grad(v,params) for v in losses]
 loss_replay=max(abs(float((losses[0]-v).detach())) for v in losses[1:]);grad_replay=max(float((a-b).norm()/(1+a.norm())) for gs in grads[1:] for a,b in zip(grads[0],gs));assert loss_replay<1e-8 and grad_replay<1e-8
 direction=[torch.randn(v.shape,dtype=v.dtype,generator=rng) for v in params];eps=1e-6
 fd=float((metric.loss([v+eps*u for v,u in zip(params,direction)])[0]-metric.loss([v-eps*u for v,u in zip(params,direction)])[0]).detach()/(2*eps));ad=sum(float((g*u).sum()) for g,u in zip(grads[0],direction));assert abs(fd-ad)<1e-6
 rows.append(dict(case=case,oracle_error=oracle_error,loss_replay=loss_replay,gradient_replay=grad_replay,finite_difference=abs(fd-ad)))
(P/'JOINT_ORTHOGONAL_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Both shared and private directions variable; exact core envelope differentiated through shared projectors. Five unchanged fixtures.'),indent=2)+'\n');print('PASS five joint-shared/private oracle and gradient controls')
