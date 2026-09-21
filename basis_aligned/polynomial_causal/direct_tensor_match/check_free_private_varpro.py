from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from free_private_varpro import FreePrivateMetric,parameters_from_private
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 T,bases,private,_,_=fixture(case);shared=[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS];metric=FreePrivateMetric(T,shared);oracle=parameters_from_private(shared,private);oracle_error=float(metric.loss(oracle,dense=True)[0]);assert oracle_error<1e-20
 rng=torch.Generator().manual_seed(41000+case);params=[(p+.1*torch.randn(p.shape,dtype=p.dtype,generator=rng)).requires_grad_() for p in oracle]
 losses=[metric.loss(params)[0],metric.loss(params,dense=True)[0],metric.loss(params,solve_through=True)[0]];grads=[torch.autograd.grad(loss,params) for loss in losses];replay=max(float((a-b).norm()/(1+a.norm())) for other in grads[1:] for a,b in zip(grads[0],other));loss_replay=max(abs(float((losses[0]-x).detach())) for x in losses[1:]);assert replay<1e-8 and loss_replay<1e-8
 directions=[torch.randn(p.shape,dtype=p.dtype,generator=rng) for p in params];eps=1e-6;fd=float((metric.loss([p+eps*v for p,v in zip(params,directions)])[0]-metric.loss([p-eps*v for p,v in zip(params,directions)])[0]).detach()/(2*eps));ad=sum(float((a*v).sum()) for a,v in zip(grads[0],directions));assert abs(fd-ad)<1e-6
 rows.append(dict(case=case,oracle_squared_error=oracle_error,loss_replay=loss_replay,envelope_dense_and_solved_gradient_replay=replay,finite_difference_replay=abs(fd-ad)))
(P/'FREE_PRIVATE_VARPRO_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Five unchanged planted product targets; freeprivate spans, exactly eliminated symmetriccores, envelopegradients checkedagainst independent differentiableKronecker solve.'),indent=2)+'\n');print('PASS five oracle, full loss, envelope/through-solve gradients and finite differences')
