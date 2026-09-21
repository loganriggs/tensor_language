from pathlib import Path
import json,torch
from check_overlap_varpro import fixture
from shared_subspace_loss import SharedSubspaceLoss
P=Path(__file__).parent
torch.set_num_threads(2)
rows=[]
for case in range(5):
 T,_,truth=fixture(case,mixed=True)
 metric=SharedSubspaceLoss(T)
 oracle=[truth[i] for i in (0,2,4,6)]
 oracle_error=float(metric.loss(oracle,explicit=True));assert oracle_error<1e-20
 rng=torch.Generator().manual_seed(29000+case)
 params=[torch.randn(p.shape,dtype=p.dtype,generator=rng).requires_grad_() for p in oracle]
 implicit=metric.loss(params);explicit=metric.loss(params,explicit=True)
 gi=torch.autograd.grad(implicit,params);ge=torch.autograd.grad(explicit,params)
 delta=max(float((a-b).norm()/(1+a.norm())) for a,b in zip(gi,ge))
 assert abs(float(implicit-explicit))<1e-10 and delta<1e-8
 # One independent directional finite difference through all QR bases.
 ds=[torch.randn(p.shape,dtype=p.dtype,generator=rng) for p in params]
 eps=1e-6
 fd=float((metric.loss([p+eps*v for p,v in zip(params,ds)],explicit=True)-metric.loss([p-eps*v for p,v in zip(params,ds)],explicit=True))/(2*eps))
 ad=sum(float((a*v).sum()) for a,v in zip(gi,ds));assert abs(fd-ad)<1e-6
 rows.append(dict(case=case,oracle_squared_error=oracle_error,loss_replay=abs(float(implicit-explicit)),gradient_replay=delta,directional_derivative_discrepancy=abs(fd-ad)))
(P/'SHARED_SUBSPACE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Five unchanged planted mixed-product targets; exact dense-core elimination and differentiable QR checked. No optimizer recovery or native success claim.'),indent=2)+'\n')
print('PASS five oracle, dense-loss, gradient and finite-difference controls')
