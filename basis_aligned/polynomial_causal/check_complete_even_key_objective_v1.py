from pathlib import Path
import json,torch
from complete_even_key_objective_v1 import grams,loss
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.manual_seed(7131134)
def tensor(a,b):
 t=torch.einsum('ia,jb->ijab',a,b);t=(t+t.transpose(0,1))/2;return (t+t.transpose(2,3))/2
rows=[]
for seed in range(4):
 m1,m2=[torch.randn(5,8,dtype=torch.float64) for _ in range(2)];basis=torch.linalg.qr(torch.randn(8,4,dtype=torch.float64)).Q
 g=grams(m1,m2,basis);inside=[m@basis@basis.T for m in [m1,m2]];outside=[m-i for m,i in zip([m1,m2],inside)]
 original=tensor(*inside)+tensor(*outside)
 for rank in [0,1,3,4]:
  u=torch.linalg.qr(torch.randn(4,rank,dtype=torch.float64),mode='reduced').Q;p=u@u.T
  retained=[m@basis@p@basis.T for m in [m1,m2]];other=[m-i for m,i in zip([m1,m2],retained)]
  actual=(tensor(*retained)+tensor(*other)-original).square().sum();pred=loss(p,*g)
  err=float(abs(actual-pred)/original.square().sum());rows.append(dict(seed=seed,rank=rank,error=err))
 # Differentiate the polynomial extension and verify a symmetric direction.
 pp=p.clone().requires_grad_(True);value=loss(pp,*g);grad=torch.autograd.grad(value,pp)[0]
 e=torch.randn_like(p);e=(e+e.T)/2;step=1e-5
 fd=(loss(p+step*e,*g)-loss(p-step*e,*g))/(2*step)
 rows.append(dict(seed=seed,gradient_error=float(abs(fd-(grad*e).sum())/original.square().sum())))
out={'pred_a':max(r.get('error',0) for r in rows)<1e-12,'pred_b':max(r.get('gradient_error',0) for r in rows)<1e-8,'rows':rows,'scope':'Complete unnormalized even-numerator replacement coefficient error, shared query/key sources. Explicit tiny tensor test, including empty/full projectors. Denominator-weighted behavior untested.'}
(P/'COMPLETE_EVEN_KEY_OBJECTIVE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
