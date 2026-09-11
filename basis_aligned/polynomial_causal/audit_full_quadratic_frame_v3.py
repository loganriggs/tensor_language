"""V3 uses exact-Hessian trust regions and preserves every outcome.

Independent-start recovery correction: V1 seed544 repeated the planted span.

Preserve V1 receipt. Require every new initial span to be far from the truth.
Recovery tolerance is separate from finite precision optimizer stopping.
"""
import json
from pathlib import Path
import torch
from full_quadratic_frame_v2 import fit
from multioutput_quadratic_blocks_v1 import MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross

torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(1881)
dim,rank=7,2
truth=torch.linalg.qr(torch.randn(dim,rank)).Q
l=torch.stack([truth[:,0],truth[:,1],truth[:,0]])
r=torch.stack([truth[:,0],truth[:,1],truth[:,1]])
down=torch.randn(dim,3)
total=((down.T@down)*product_cross(l,r,l,r)).sum()
obj=MultioutputWeightObjective(torch.eye(dim),total,l=l,r=r,d=down,penalty=1e-8)
rows=[]
for seed in (1901,1902,1903):
    torch.manual_seed(seed)
    initial=torch.linalg.qr(torch.randn(1,dim,rank)).Q
    initial_error=float((initial[0]@initial[0].T-truth@truth.T).norm()/rank**.5)
    assert initial_error>.5
    final,report=fit(obj,initial,seconds=20,tolerance=1e-7)
    error=float((final[0]@final[0].T-truth@truth.T).norm()/rank**.5)
    rows.append(dict(seed=seed,initial_projector_error=initial_error,
                     final_projector_error=error,**report))
result=dict(correction='Exact-Hessian trust regions on the same independent starts after V1 CG seed1902 failed; original failure preserved.',
            starts=rows,all_recovered=all(r['final_projector_error']<1e-5 for r in rows),
            all_gradient_stops=all(r['gradient_norm']<=1e-7 for r in rows))
with Path(__file__).with_name('FULL_QUADRATIC_FRAME_V3_RESTART_AUDIT.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
