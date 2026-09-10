from pathlib import Path
import torch,json
from fit_sparse_core_rcg_v1 import fit
from joint_quadratic_fit_v1 import product_cross

torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(716)
basis,_=torch.linalg.qr(torch.randn(8,3));l=basis[:,[0,1,1,0]].T;r=basis[:,[0,1,2,2]].T;d=torch.randn(5,4);total=((d.T@d)*product_cross(l,r,l,r)).sum();q,_=torch.linalg.qr(torch.randn(8,3))
a=fit(q,l,r,d,total,count=4,seconds=30,max_steps=40)
b=fit(q,l,r,d,total,count=4,seconds=30,max_steps=20);b=fit(q,l,r,d,total,count=4,seconds=30,max_steps=20,state=b)
result=dict(initial=a['initial_capture'],final=a['history'][-1]['captured_energy'],maximum_score_decrease=a['maximum_score_decrease'],orthogonality_error=a['history'][-1]['orthogonality_error'],resume_frame_error=float((a['q']-b['q']).abs().max()),resume_capture_error=abs(a['history'][-1]['captured_energy']-b['history'][-1]['captured_energy']))
result['passed']=result['final']>result['initial']+.1 and max(result[k] for k in ['maximum_score_decrease','orthogonality_error','resume_frame_error','resume_capture_error'])<=1e-10;assert result['passed'],result
Path(__file__).with_name('SPARSE_CORE_RCG_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
