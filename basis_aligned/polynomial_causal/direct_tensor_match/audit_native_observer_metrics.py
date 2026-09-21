"""Keep isotropic and covariance-weighted scalar-operator baselines explicit."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
e=torch.load(p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True);M=e['native_matrix'];U,s,Vh=torch.linalg.svd(M,full_matrices=False);isotropic=s[0]*torch.outer(U[:,0],Vh[0])
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);roots=[]
for key in ['n','m']:
 x=rows[key].flatten(0,1).double();x-=x.mean(0);ev,V=torch.linalg.eigh(x.T@x/len(x));roots.append((V*ev.clamp_min(0).sqrt())@V.T)
N,P=roots;records=[]
for name,q in [('frozen',torch.outer(e['frozen_a'],e['frozen_b'])),('covariance_optimal',torch.outer(e['native_rank1_a'],e['native_rank1_b'])),('isotropic_optimal',isotropic)]:
 records.append(dict(candidate=name,isotropic_coefficient_error=float((q-M).norm()/M.norm()),covariance_coefficient_error=float((N@(q-M)@P).norm()/(N@M@P).norm())))
result=dict(records=records,isotropic_leading_energy_fraction=float(s[0].square()/s.square().sum()),scope='Same fixed native scalar observer, two explicit input metrics. All objectives use original weights; covariance is data-informed. Isotropic best rank1 is a control, not newly validated behavioral candidate.')
out=p/'MIDPOINT_NATIVE_OBSERVER_METRIC_COMPARISON_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
