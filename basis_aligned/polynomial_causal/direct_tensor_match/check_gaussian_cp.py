import itertools,json
from pathlib import Path
import numpy as np
import torch
from gaussian_cp import gaussian_cp_gram,PAIRINGS
P=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1710);a=[torch.randn(2,3,requires_grad=True) for _ in range(4)];b=[torch.randn(3,3) for _ in range(4)];mat=torch.randn(3,3);cov=mat@mat.T+.2*torch.eye(3);nodes,weights=np.polynomial.hermite.hermgauss(5);inds=torch.tensor(list(itertools.product(range(5),repeat=3)));z=torch.tensor(nodes)[inds]*2**.5;w=torch.tensor(weights/np.sqrt(np.pi))[inds].prod(1);rows=[]
for name,M in [('isotropic',torch.eye(3)),('covariance',cov)]:
 x=z@torch.linalg.cholesky(M).T;fa=torch.ones(len(x),2);fb=torch.ones(len(x),3)
 for v in a:fa=fa*(x@v.T)
 for v in b:fb=fb*(x@v.T)
 ref=fa.T@(w[:,None]*fb);got=gaussian_cp_gram(a,b,M);err=float(((got-ref).norm()/ref.norm()).detach());g1=torch.autograd.grad(got.square().sum(),a,retain_graph=True);g2=torch.autograd.grad(ref.square().sum(),a);grad=max(float((v-u).norm()/u.norm()) for v,u in zip(g1,g2));whiten=gaussian_cp_gram([v@torch.linalg.cholesky(M) for v in a],[v@torch.linalg.cholesky(M) for v in b]);we=float(((got-whiten).norm()/got.norm()).detach());assert max(err,grad,we)<1e-11;rows.append(dict(metric=name,value_relative_error=err,gradient_relative_error=grad,whitening_relative_error=we))
assert len(PAIRINGS)==105
out=dict(pairings=len(PAIRINGS),checks=rows,scope='Exact zero-mean Gaussian eighth-moment CP Gram independently validated by5-point-per-axis quadrature and gradients, isotropic and full covariance. Not empirical eighth moments or noncentral Gaussian.')
(P/'GAUSSIAN_CP_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
