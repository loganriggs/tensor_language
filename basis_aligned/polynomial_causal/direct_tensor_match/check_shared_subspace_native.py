from pathlib import Path
import json,torch
from shared_subspace_loss import SharedSubspaceLoss
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True)
Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);rows=[]
for key in ('calibration_shaped_0','native_isotropic_26301'):
 A=S if key.startswith('calibration') else torch.eye(1152,dtype=Q.dtype);program=programs[key]
 params=[(A@x).detach().requires_grad_() for x in [program['input_basis']]+[program['pairs'][str(j)]['private_reader'] for j in range(3)]]
 metric=SharedSubspaceLoss(A@Q@A);implicit=metric.loss(params);explicit=metric.loss(params,True)
 grad=torch.autograd.grad(implicit,params);assert all(torch.isfinite(g).all() for g in grad)
 replay=abs(float((implicit-explicit).detach()));assert replay<1e-8
 rows.append(dict(key=key,loss_replay=replay,root_error=float(explicit.detach().sqrt()),shapes=[list(p.shape) for p in params],finite_gradients=True))
(P/'SHARED_SUBSPACE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print(json.dumps(rows))
