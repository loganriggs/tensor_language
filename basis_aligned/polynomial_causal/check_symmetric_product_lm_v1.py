import json
from pathlib import Path
import torch
from fit_symmetric_product_gauss_newton_v1 import fit,canonical_project
from symmetric_product_gauss_newton_v1 import loss_gradient
from joint_quadratic_fit_v1 import product_cross

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(104)
l,r,d=torch.randn(2,5),torch.randn(2,5),torch.randn(4,2)
target=(l,r,d);total=((d.T@d)*product_cross(l,r,l,r)).sum()
p=(l+.1*torch.randn_like(l),r+.1*torch.randn_like(r),d)
p,record=fit(p,target,total,seconds=10,max_steps=40)
values=[record['initial']]+[v['objective'] for v in record['history']]
result=dict(initial=record['initial'],final=record['final'],accepted=record['accepted'],rejected=record['rejected'],max_objective_increase=max(b-a for a,b in zip(values,values[1:])),projection_replay=float(abs(loss_gradient(canonical_project(p,target,.01),target,total,.01)[0]-record['final'])))
result['passed']=result['final']<result['initial']-1e-4 and result['max_objective_increase']<=1e-12 and result['projection_replay']<=1e-12
assert result['passed'],result
Path(__file__).with_name('SYMMETRIC_PRODUCT_LM_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
