import itertools,json
from pathlib import Path
import numpy as np
import torch
from noncentral_quadratic import gaussian_inner

torch.set_num_threads(1);torch.manual_seed(925);torch.set_default_dtype(torch.float64)
c,a,b,d,l,r=[torch.randn(*s,requires_grad=True) for s in [(2,3),(3,3),(3,3),(2,4),(4,3),(4,3)]]
mean=torch.randn(3);nodes,weights=np.polynomial.hermite.hermgauss(3)
xx=torch.tensor(list(itertools.product(nodes,nodes,nodes)))*2**.5+mean
ww=torch.tensor([np.prod(v) for v in itertools.product(weights,weights,weights)])/np.pi**1.5
f=((xx@a.T)*(xx@b.T))@c.T;g=((xx@l.T)*(xx@r.T))@d.T
ref=((f*g).sum(-1)*ww).sum();got=gaussian_inner(c,a,b,d,l,r,mean)
err=float((ref-got).abs().detach());grad=max(float((u-v).abs().max()) for u,v in zip(torch.autograd.grad(ref,(c,a,b,d,l,r),retain_graph=True),torch.autograd.grad(got,(c,a,b,d,l,r))))
assert err<1e-10 and grad<1e-10
out=dict(quadrature_absolute_error=err,gradient_max_error=grad,scope='Exact noncentral Gaussian degree-four moments; 27-point quadrature')
Path(__file__).with_name('NONCENTRAL_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
