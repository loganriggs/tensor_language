from pathlib import Path
import json,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from pairwise_exact_supports import supports
from joint_orthogonal_private import JointOrthogonalPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for case in range(5):
 T,bases,private,_,_=fixture(case);known=list(bases)+list(private);U=supports(T);latent=[u.T@v for u,v in zip(U,known)]
 parameter_replay=max(float((u@v-t).norm()/t.norm()) for u,v,t in zip(U,latent,known));assert parameter_replay<1e-8
 metric=JointOrthogonalPrivateMetric(T,[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS]);oracle=float(metric.loss([u@v for u,v in zip(U,latent)],dense=True)[0]);assert oracle<1e-20
 rng=torch.Generator().manual_seed(45000+case);params=[(v+.1*torch.randn(v.shape,dtype=v.dtype,generator=rng)).requires_grad_() for v in latent]
 expand=lambda ps:[u@v for u,v in zip(U,ps)]
 loss=metric.loss(expand(params))[0];dense=metric.loss(expand(params),dense=True)[0];g=torch.autograd.grad(loss,params);gd=torch.autograd.grad(dense,params)
 discrepancy=max(float((a-b).norm()/(1+a.norm())) for a,b in zip(g,gd));assert discrepancy<1e-8
 rows.append(dict(case=case,support_dimensions=[u.shape[1] for u in U],planted_widths=[v.shape[1] for v in known],parameter_replay=parameter_replay,oracle_loss=oracle,gradient_replay=discrepancy))
(P/'JOINT_SUPPORTS_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows,scope='Weight-derived supports and pairwise intersections contain each planted direction. Only dimension choices use planted architecture. Native full-rank limitation explicit.'),indent=2)+'\n');print(json.dumps(rows,indent=2))
