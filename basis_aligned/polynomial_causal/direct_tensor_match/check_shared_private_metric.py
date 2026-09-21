from pathlib import Path
import json,torch
from shared_private_metric import SharedPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2);rows=[]
for seed in range(5):
 g=torch.Generator().manual_seed(8595+seed);dtype=torch.float64
 T=torch.randn(6,10,10,dtype=dtype,generator=g);T=(T+T.transpose(-1,-2))/2;B=torch.randn(6,6,dtype=dtype,generator=g);M=B.T@B+torch.eye(6,dtype=dtype)
 L=torch.randn(10,4,dtype=dtype,generator=g,requires_grad=True);R=torch.randn(10,4,dtype=dtype,generator=g,requires_grad=True);V=torch.randn(10,0 if seed==0 else 3,dtype=dtype,generator=g,requires_grad=True)
 metric=SharedPrivateMetric(T,M);loss,W,v=metric.loss(L,R,V);a=torch.autograd.grad(loss,(L,R,V),retain_graph=True)
 explicit=metric.explicit(metric.dense(L,R,V,W,v),W,v);b=torch.autograd.grad(explicit,(L,R,V));gradient=max(float((aa-bb).abs().max()) if aa.numel() else 0 for aa,bb in zip(a,b));error=abs(float(loss.detach()-explicit.detach()));assert max(error,gradient)<1e-10
 full,_,_=metric.loss(L,R,V,detach=False);c=torch.autograd.grad(full,(L,R,V));envelope=max(float((aa-cc).abs().max()) if aa.numel() else 0 for aa,cc in zip(a,c));assert envelope<1e-10
 rows.append(dict(seed=seed,private_width=V.shape[1],loss_replay=error,gradient_replay=gradient,envelope_replay=envelope))
(P/'SHARED_PRIVATE_METRIC_PREFLIGHT_V1.json').write_text(json.dumps(dict(checks=rows,scope='Five synthetic dense-loss, gradient and analytic-readout envelope checks; includes empty-private baseline. These are instrument checks, not structural recovery experiments.'),indent=2)+'\n');print(rows)
