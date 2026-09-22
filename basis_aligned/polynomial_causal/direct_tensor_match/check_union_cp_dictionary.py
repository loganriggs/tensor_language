"""Independent quadrature controls for fixed-bank Gaussian union fitting."""
import itertools,json
import numpy as np
import torch
from mixed_gaussian_cp import gram_dynamic
from quartic_cp_profile import profile

def check():
 torch.set_num_threads(2);a,w=np.polynomial.hermite.hermgauss(5);z=torch.tensor(list(itertools.product(a*2**.5,repeat=3)),dtype=torch.float64);weights=torch.tensor([a*b*c for a,b,c in itertools.product(w/np.pi**.5,repeat=3)],dtype=z.dtype);rows=[]
 for seed,name in enumerate(['complementary','shared_factors','repeated_powers','signed_cancellation','duplicate_banks']):
  torch.manual_seed(21000+seed);fs=[torch.randn(6,3,dtype=z.dtype) for _ in range(4)];bs=[torch.randn(6,dtype=z.dtype) for _ in range(4)]
  if seed==1:fs[1]=fs[0]
  if seed==2:fs=[fs[0]]*4;bs=[bs[0]]*4
  if seed==4:
   fs=[torch.cat([f[:3],f[:3]]) for f in fs];bs=[torch.cat([b[:3],b[:3]]) for b in bs]
  c=torch.randn(2,6,dtype=z.dtype)
  if seed==3:c[:,1]=-c[:,0]
  phi=torch.stack([z@f.T+b for f,b in zip(fs,bs)]).prod(0);y=phi@c.T;g=gram_dynamic(fs,bs,fs,bs);cross=c@g;oracle=phi.T@(weights[:,None]*phi);err=float((g-oracle).norm()/oracle.norm());assert err<1e-12
  ce=float((cross-y.T@(weights[:,None]*phi)).norm()/cross.norm());assert ce<1e-12
  _,fit=profile(g,cross,ridge=1e-6);normal=float((fit@(g+1e-6*torch.eye(6))-cross).norm()/cross.norm());assert normal<1e-10
  rows.append(dict(family=name,gram_error=err,cross_error=ce,normal_residual=normal))
 return rows
if __name__=='__main__':print(json.dumps(check(),indent=2))
