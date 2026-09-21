from pathlib import Path
import torch,json
from dual_geometry_source_metric import DualGeometrySourceMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for seed,alpha in enumerate([0.,.1,.5,.9,1.]):
 g=torch.Generator().manual_seed(10100+seed);r=lambda *s:torch.randn(*s,dtype=torch.float64,generator=g)
 T=r(6,8,8);T=(T+T.transpose(-1,-2))/2;B=r(8,8);S=B@B.T/8+torch.eye(8,dtype=T.dtype)*.1
 L,R,V=[a.requires_grad_() for a in (r(8,4),r(8,4),r(8,2))];m=DualGeometrySourceMetric(T,S,alpha)
 loss,W,v=m.loss(L,R,V);dense=m.explicit(L,R,V,W,v);err=abs(float((loss-dense).detach()));a=torch.autograd.grad(loss,(L,R,V),retain_graph=True);b=torch.autograd.grad(dense,(L,R,V),retain_graph=True);grad=max(float((x-y).abs().max()) for x,y in zip(a,b));full,_,_=m.loss(L,R,V,detach=False);c=torch.autograd.grad(full,(L,R,V));envelope=max(float((x-y).abs().max()) for x,y in zip(a,c))
 assert max(err,grad,envelope)<1e-10;rows.append(dict(seed=seed,alpha=alpha,dense_replay=err,gradient_replay=grad,envelope_replay=envelope))
(P/'DUAL_GEOMETRY_SOURCE_PREFLIGHT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
