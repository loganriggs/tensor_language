from pathlib import Path
import json,torch
from source_sobolev import SourceSobolev
from shared_quadratic_products import materialize_mixed
P=Path(__file__).resolve().parent;torch.set_num_threads(2);records=[]
for seed in range(5):
 torch.manual_seed(758+seed);n=8;r=4;L=torch.randn(n,r,dtype=torch.double,requires_grad=True);R=torch.randn(n,r,dtype=torch.double,requires_grad=True)
 T=torch.randn(3,n,n,dtype=torch.double);T=(T+T.transpose(-1,-2))/2;B=torch.randn(n,n,dtype=torch.double);H=B@B.T+torch.eye(n,dtype=torch.double)
 metric=SourceSobolev(T,H,[0,.1,1,10,100][seed]);loss,W=metric.loss(L,R);explicit=metric.explicit(materialize_mixed(L,R,W),W)
 g=torch.autograd.grad(loss,(L,R),retain_graph=True);exact,_=metric.loss(L,R,detach=False);gg=torch.autograd.grad(exact,(L,R))
 replay=float(abs(loss-explicit));grad=max(float((a-b).abs().max()) for a,b in zip(g,gg));assert replay<1e-10 and grad<1e-9
 records.append(dict(seed=seed,lambda_value=metric.lam,loss_replay=replay,envelope_gradient_error=grad))
(P/'SOURCE_SOBOLEV_CHECK_V1.json').write_text(json.dumps(dict(records=records,all_pass=True),indent=2)+'\n');print(records)
