from pathlib import Path
import torch,json
from empirical_source_varpro import EmpiricalSourceVarpro
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);p=torch.load(P/'COMPACT_GROUP_FROZEN_V1.pt',weights_only=True);root=torch.linalg.inv(d['inverse_root']);L,R,V=[(root@p[n]).requires_grad_() for n in ['left_reader','right_reader','square_reader']];T=d['teacher'];x=(d['z'][:128]-d['mu'])@d['inverse_root'];K=d['inverse_root']@d['old_covariance']@d['inverse_root'];Y=torch.einsum('ni,oij,nj->no',x,T,x)-torch.einsum('ij,oji->o',K,T)
metric=EmpiricalSourceVarpro(T,x,K,Y,1);loss,W,v=metric.loss(L,R,V);loss.backward();check=metric.explicit(L,R,V,W,v);replay=abs(float(loss.detach()-check.detach()));assert replay<1e-8 and all(torch.isfinite(a.grad).all() for a in [L,R,V])
out=dict(native_dimension=1152,shared_products=367,private_squares=32,preflight_rows=128,objective=float(loss.detach()),dense_replay=replay,gradient_norms=[float(a.grad.norm()) for a in [L,R,V]])
(P/'EMPIRICAL_SOURCE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
