import itertools,json
from pathlib import Path
import torch
from core import metric,coefficients_from_dense,inner,multiply,evaluate,terms

def main():
 torch.set_num_threads(1);torch.manual_seed(831);torch.set_default_dtype(torch.float64);reports=[]
 for degree in [2,4]:
  d=3;t=torch.randn((2,)+(d,)*degree);c=coefficients_from_dense(t)
  sym=sum(t.permute((0,)+tuple(i+1 for i in p)) for p in itertools.permutations(range(degree)))/__import__('math').factorial(degree)
  mf=metric(d,degree,'frobenius');err=float(abs(inner(c,c,mf)-sym.square().sum()));assert err<1e-10
  # Independent exact Gauss-Hermite integration, not training data.
  import numpy as np
  nodes,weights=np.polynomial.hermite.hermgauss(degree+1);nodes=nodes*2**.5;weights=weights/np.pi**.5
  grid=list(itertools.product(range(degree+1),repeat=d));x=torch.tensor([[nodes[i] for i in row] for row in grid]);w=torch.tensor([np.prod([weights[i] for i in row]) for row in grid])
  direct=(evaluate(c,x,degree).square().sum(-1)*w).sum();gaussian=inner(c,c,metric(d,degree));ge=float(abs(direct-gaussian)/gaussian);assert ge<1e-12
  z=c.clone().requires_grad_();assert torch.autograd.gradcheck(lambda v:inner(v-c,v-c,metric(d,degree)),(z,))
  reports.append(dict(degree=degree,frobenius_absolute_error=err,gaussian_quadrature_relative_error=ge))
 # Repeated-input cancellation: (x0²+x1²)²-(x0²-x1²)² = (2x0x1)².
 q=torch.tensor([[1.,0,1.],[1.,0,-1.],[0.,2.,0.]])
 c=multiply(q,q,2,2,2);assert (c[0]-c[1]-c[2]).abs().max()==0
 # Cosine alone cannot distinguish global scale, difference norm can.
 assert inner(c,c,metric(2,4))>0
 return dict(checks=reports,cancellation_exact=True,gradient_check=True)
if __name__=='__main__':
 r=main();Path(__file__).with_name('CORE_CHECK_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
