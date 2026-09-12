"""Compare Schur scores against direct augmented least-squares solves."""
from pathlib import Path
import json
import torch
from quartic_residual_addition_v1 import scores

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(91840)
x=torch.randn(20,7);target=torch.randn(20,2);k=x.T@x;c=x.T@target
ids=torch.tensor([0,2,4]);gains,var,base=scores(k,c,ids)
errors=[]
for j in (1,3,5,6):
    active=torch.cat([ids,torch.tensor([j])]);a=torch.linalg.solve(k[active][:,active],c[active])
    direct=float((a*c[active]).sum())-base
    errors.append(abs(direct-float(gains[j]))/max(1.,abs(direct)))
result=dict(pred_a=max(errors)<=1e-10,maximum_relative_error=max(errors),retained_excluded=bool(torch.isneginf(gains[ids]).all()))
assert result['pred_a'] and result['retained_excluded']
p=Path(__file__).resolve().parent/'QUARTIC_RESIDUAL_ADDITION_V1_CONTROL.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(result)
