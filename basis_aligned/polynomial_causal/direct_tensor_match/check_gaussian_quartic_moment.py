"""Gauss-Hermite quadrature independently integrates degree-eight squared error."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from gaussian_quartic_moment import squared_error_by_degree
torch.set_num_threads(2);torch.manual_seed(534)
nodes,weights=np.polynomial.hermite.hermgauss(5)
records=[]
for d in [2,3]:
 indices=list(itertools.product(range(5),repeat=d))
 g=torch.tensor([[nodes[i]*2**.5 for i in row] for row in indices],dtype=torch.float64)
 w=torch.tensor([np.prod([weights[i]/np.pi**.5 for i in row]) for row in indices],dtype=torch.float64)
 x=torch.cat([g,torch.ones(len(g),1,dtype=g.dtype)],1)
 for trial in range(3):
  mats=[];parameters=[]
  for j in range(4):
   raw=torch.randn(d+1,d+1,dtype=torch.float64).requires_grad_(j>=2)
   if j>=2:parameters.append(raw)
   mats.append((raw+raw.T)/2)
  A,B,C,D=mats
  pred=lambda M,N:((x@M)*x).sum(1)*((x@N)*x).sum(1)
  exact=((pred(A,B)-pred(C,D)).square()*w).sum()
  degrees=squared_error_by_degree(A,B,C,D);estimate=degrees.sum()
  grad1=torch.autograd.grad(exact,parameters,retain_graph=True);grad2=torch.autograd.grad(estimate,parameters)
  error=float((exact-estimate).detach().abs()/exact.detach())
  gradient=max(float((a-b).norm()/a.norm()) for a,b in zip(grad1,grad2))
  assert error<1e-12 and gradient<1e-12 and degrees.min()>-1e-10
  records.append(dict(dimension=d,trial=trial,value_relative_error=error,gradient_relative_error=gradient))
out=dict(passed=True,records=records,scope='Exact five-node-per-axis quadrature for a degree-eight integrand, with final coordinate fixed to one. Verifies values and gradients, including all Hermite degrees.')
Path(__file__).with_name('GAUSSIAN_QUARTIC_MOMENT_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
