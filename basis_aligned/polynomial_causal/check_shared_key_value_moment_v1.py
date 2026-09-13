from pathlib import Path
import itertools,json,numpy as np,torch
from shared_key_value_moment_v1 import moment
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.manual_seed(7131203)
nodes,weights=np.polynomial.hermite.hermgauss(4);nodes=nodes*2**.5;weights=weights/np.pi**.5
indices=torch.tensor(list(itertools.product(range(4),repeat=5)));y=torch.tensor(nodes,dtype=torch.float64)[indices];prob=torch.tensor(weights,dtype=torch.float64)[indices].prod(-1)
f=torch.randn(3,5,dtype=torch.float64);a=torch.randn(5,dtype=torch.float64);b=torch.randn(5,dtype=torch.float64)
v=y@f.T;amplitude=(y@a)*(y@b);direct=v.T@((prob*amplitude.square())[:,None]*v);pred=moment(f,a,b);error=float((pred-direct).norm()/direct.norm())
base=(a.square().sum()*b.square().sum()+2*(a@b).square())*(f@f.T);correction=pred-base;s=torch.linalg.svdvals(correction)
out={'pred_a':error<1e-12,'pred_b':float(s[-1]/s[0])<1e-12,'relative_error':error,'quadrature_points':len(y),'correction_singular_values':s.tolist(),'scope':'Exact degree6 Gaussian moment checked by four-point product Gauss-Hermite quadrature. Fixed query/key readers and linear values; no normalized attention or behavioral claim.'}
(P/'SHARED_KEY_VALUE_MOMENT_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
