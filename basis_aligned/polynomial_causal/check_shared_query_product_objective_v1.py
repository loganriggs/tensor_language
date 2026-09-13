from pathlib import Path
import json,torch
from shared_query_product_objective_v1 import captured,gradient
P=Path(__file__).resolve().parent
torch.manual_seed(7131125);torch.set_num_threads(2)
a,b=[torch.randn(5,7,dtype=torch.float64) for _ in range(2)]
t=torch.einsum('ia,jb->ijab',a,b);t=(t+t.transpose(0,1))/2;t=(t+t.transpose(2,3))/2
u=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q;p=u@u.T
g1=(a.T@a)[None];g2=(b.T@b)[None];c=(a.T@b)[None]
full=captured(torch.eye(7,dtype=torch.float64),g1,g2,c)
projected=torch.einsum('ijab,ac,bd->ijcd',t,p,p)
errs=[float(abs(full-t.square().sum())/full),float(abs(captured(p,g1,g2,c)-projected.square().sum())/full),float(abs((t-projected).square().sum()-(full-captured(p,g1,g2,c)))/full)]
e=torch.randn_like(p);e=(e+e.T)/2;h=1e-5
fd=(captured(p+h*e,g1,g2,c)-captured(p-h*e,g1,g2,c))/(2*h)
ge=float(abs(fd-(gradient(p,g1,g2,c)*e).sum())/full)
out={'pred_a':max(errs)<1e-12,'pred_b':ge<1e-8,'identity_errors':errs,'gradient_error':ge,'scope':'Double-symmetric polynomial coefficient tensor; no normalization or native behavior.'}
(P/'SHARED_QUERY_PRODUCT_OBJECTIVE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
