"""Independent degree-six Gaussian quadrature checks values and parameter gradients."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from gaussian_cp_derivative_gram import gram

def controls():
 torch.set_num_threads(2);dtype=torch.float64;nodes,weights=np.polynomial.hermite.hermgauss(4);ids=torch.tensor(list(itertools.product(range(4),repeat=3)));z=torch.tensor(nodes*2**.5,dtype=dtype)[ids];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ids].prod(1);rows=[]
 for seed,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(12800+seed);f=[torch.randn(7,3,dtype=dtype) for _ in range(4)];b=[torch.randn(7,dtype=dtype) for _ in range(4)]
  if family=='shared_input':f[1]=f[0].clone();b[1]=b[0].clone()
  if family=='squares':f[2:]=[a.clone() for a in f[:2]];b[2:]=[a.clone() for a in b[:2]]
  if family=='cancellation':
   for a,c in zip(f,b):a[1]=a[0];c[1]=c[0]
   f[0][1]*=-1;b[0][1]*=-1
  f=[a.requires_grad_() for a in f];b=[a.requires_grad_() for a in b];G=gram(f,b)
  linear=[z@a.T+c for a,c in zip(f,b)];jac=0.
  for i in range(4):
   factor=torch.ones_like(linear[0])
   for j in range(4):
    if i!=j:factor=factor*linear[j]
   jac=jac+factor[:,:,None]*f[i][None,:,:]
  reference=torch.einsum('nad,nbd,n->ab',jac,jac,w)
  C=torch.randn(2,7,dtype=dtype)
  if family=='shared_output':C[1]=2*C[0]
  probe=C.T@C;ga=torch.autograd.grad((G*probe).sum(),f+b,retain_graph=True);gb=torch.autograd.grad((reference*probe).sum(),f+b)
  rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-30)).detach())
  value=rel(G,reference);gradient=rel(torch.cat([a.flatten() for a in ga]),torch.cat([a.flatten() for a in gb]));assert value<1e-10 and gradient<1e-10
  eig=torch.linalg.eigvalsh(G.detach());assert float(eig[0])>=-1e-10*float(eig[-1]);rows.append(dict(family=family,quadrature_error=value,gradient_error=gradient))
 return rows
if __name__=='__main__':
 rows=controls();Path(__file__).with_name('GAUSSIAN_CP_DERIVATIVE_GRAM_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
