from pathlib import Path
import json,torch
from shared_private_metric import SharedPrivateMetric
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'GLOBAL_PRIVATE_RESIDUAL_PROGRAMS_V1.pt',weights_only=True);T=d['teacher'];F=T.flatten(1);e,U=torch.linalg.eigh(F@F.T);M=(U*e.pow(-.5))@U.T;root=torch.linalg.inv(d['inverse_root']);metric=SharedPrivateMetric(T,M);rows=[]
for k in [0,16]:
 p=programs[str(k)];L,R,V=[(root@p[n]).requires_grad_() for n in ['left_reader','right_reader','square_reader']]
 loss,W,v=metric.loss(L,R,V);loss.backward();dense=metric.explicit(metric.dense(L,R,V,W,v),W,v);error=abs(float(loss.detach()-dense.detach()));assert error<1e-8 and all(torch.isfinite(a.grad).all() for a in [L,R,V])
 rows.append(dict(removed=k,initial_objective=float(loss.detach()),dense_replay=error,gradient_norms=[float(a.grad.norm()) for a in [L,R,V]]))
(P/'SHARED_PRIVATE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
