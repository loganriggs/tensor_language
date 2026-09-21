from pathlib import Path
import json,torch
from empirical_source_varpro import EmpiricalSourceVarpro
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for seed in range(5):
 g=torch.Generator().manual_seed(9410+seed);r=lambda *s:torch.randn(*s,dtype=torch.float64,generator=g)
 T=r(6,8,8);T=(T+T.transpose(-1,-2))/2;x=r(64,8);K=torch.eye(8,dtype=T.dtype);Y=torch.einsum('ni,oij,nj->no',x,T,x)-torch.einsum('ij,oji->o',K,T)
 L,R,V=[a.requires_grad_() for a in [r(8,4),r(8,4),r(8,3)]];metric=EmpiricalSourceVarpro(T,x,K,Y,[0,.01,.1,1,10][seed]);loss,W,v=metric.loss(L,R,V);grad=torch.autograd.grad(loss,(L,R,V),retain_graph=True);dense=metric.explicit(L,R,V,W,v);check=torch.autograd.grad(dense,(L,R,V));error=abs(float(loss.detach()-dense.detach()));ge=max(float((a-b).abs().max()) for a,b in zip(grad,check));full,_,_=metric.loss(L,R,V,detach=False);third=torch.autograd.grad(full,(L,R,V));envelope=max(float((a-b).abs().max()) for a,b in zip(grad,third));assert max(error,ge,envelope)<1e-9
 rows.append(dict(seed=seed,lam=metric.lam,loss_replay=error,gradient_replay=ge,envelope_replay=envelope))
(P/'EMPIRICAL_SOURCE_VARPRO_PREFLIGHT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
