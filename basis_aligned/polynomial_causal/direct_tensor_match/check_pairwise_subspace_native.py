from pathlib import Path
import torch,json
from pairwise_subspace_loss import PairwiseSubspaceLoss,from_common
from shared_subspace_loss import SharedSubspaceLoss
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);states=torch.load(P/'SHARED_SUBSPACE_NATIVE_STATES_V1.pt',weights_only=True);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(d['inverse_root']);rows=[]
for key in ('calibration_shaped_0','native_isotropic_26301'):
 A=S if key.startswith('calibration') else torch.eye(1152,dtype=Q.dtype);T=A@Q@A;old=states[key];new=[a.requires_grad_() for a in from_common(old)];metric=PairwiseSubspaceLoss(T)
 implicit=metric.loss(new);explicit=metric.loss(new,True);previous=SharedSubspaceLoss(T).loss(old,True);grad=torch.autograd.grad(implicit,new)
 replay=max(abs(float((implicit-explicit).detach())),abs(float((explicit-previous).detach())));assert replay<1e-8 and all(torch.isfinite(g).all() for g in grad)
 rows.append(dict(key=key,shapes=[list(p.shape) for p in new],initialization_and_dense_replay=replay,finite_gradients=True))
(P/'PAIRWISE_SUBSPACE_NATIVE_PREFLIGHT_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print(json.dumps(rows))
