"""Fixed-width output sharing: paired sum versus separate first-order input roles."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);out=p/'MIDPOINT_LINEAR_ROLE_METRIC_V1.json';assert not out.exists()
e=torch.load(p/'MIDPOINT_ADAPTIVE_MIXED_GRAPHS_V1.pt',weights_only=True)['adaptive_mixed512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();n-=n.mean(0);m-=m.mean(0);S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double();yn=n@e['linear_n']@S;ym=m@e['linear_m']@S;records=[]
for metric,z in [('paired',yn+ym),('separate',torch.cat([yn,ym])),('source',ym)]:
 _,s,Vh=torch.linalg.svd(z,full_matrices=False);Q=Vh[:256].T
 err=lambda x:float((x-(x@Q)@Q.T).norm()/x.norm())
 records.append(dict(metric=metric,rank=256,joint_error=err(yn+ym),context_linear_error=err(yn),source_linear_error=err(ym)))
result=dict(records=records,scope='Same256-dimensional shared output basis budget. Calibrated original linear maps only. Paired mean-zero sum, separate mean-zero roles, or source-only objective. Relative errors of each linear target; no held/native confirmation. Centered bilinear interaction unaffected by all variants.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
