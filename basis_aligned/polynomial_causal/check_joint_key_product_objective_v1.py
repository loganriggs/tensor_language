from pathlib import Path
import json,torch
from joint_key_product_objective_v1 import captured,gradient
P=Path(__file__).resolve().parent
torch.manual_seed(7131120);torch.set_num_threads(2)
a,b=[torch.randn(4,7,dtype=torch.float64) for _ in range(2)]
t=(torch.einsum('ia,jb->ijab',a,b)+torch.einsum('ib,ja->ijab',a,b))/2
g1=a.T@a;g2=b.T@b;u=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q;p=u@u.T
projected=torch.einsum('ijab,ac,bd->ijcd',t,p,p)
full=captured(torch.eye(7,dtype=torch.float64),g1,g2)
errors=[float(abs(full-t.square().sum())/full),float(abs(captured(p,g1,g2)-projected.square().sum())/full),float(abs((t-projected).square().sum()-(full-captured(p,g1,g2)))/full)]
e=torch.randn_like(p);e=(e+e.T)/2;step=1e-5
fd=(captured(p+step*e,g1,g2)-captured(p-step*e,g1,g2))/(2*step)
graderror=float(abs(fd-(gradient(p,g1,g2)*e).sum())/full)
out={'pred_a':max(errors)<1e-12,'pred_b':graderror<1e-8,'identity_errors':errors,'gradient_error':graderror,'scope':'Explicit tiny symmetric tensor check; unnormalized inside-key product only, no attention denominator or query-state restriction.'}
(P/'JOINT_KEY_PRODUCT_OBJECTIVE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
