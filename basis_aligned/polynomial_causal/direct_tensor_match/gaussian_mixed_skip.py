"""Moments of existing linear readers and quadratic products, jointly centered."""
import torch
from gaussian_quartic_mean import quadratic_moments

def moments(A,B,mu,M):
 H=torch.cat([A,B]);mp,Gp=quadratic_moments(A,B,mu,M)
 expected_gradient=(B@mu)[:,None]*A+(A@mu)[:,None]*B
 cross=H@M@expected_gradient.T
 G=torch.cat([torch.cat([H@M@H.T,cross],1),torch.cat([cross.T,Gp],1)],0)
 return torch.cat([H@mu,mp]),G

def check():
 import numpy as np
 torch.set_num_threads(2);torch.manual_seed(260932);dtype=torch.float64
 A,B=[torch.randn(2,5,dtype=dtype) for _ in range(2)];mu=torch.randn(5,dtype=dtype);raw=torch.randn(5,5,dtype=dtype);M=raw@raw.T+.2*torch.eye(5,dtype=dtype)
 n,w=np.polynomial.hermite.hermgauss(3);n=torch.tensor(n*2**.5);w=torch.tensor(w/np.pi**.5);idx=torch.cartesian_prod(*[torch.arange(3)]*5);x=n[idx]@torch.linalg.cholesky(M).T+mu;weights=w[idx].prod(1);a,b=x@A.T,x@B.T;p=torch.cat([a,b,a*b],1);mean,G=moments(A,B,mu,M);emp=(weights[:,None]*p).sum(0);center=p-emp;expected=center.T@(weights[:,None]*center)
 errors=dict(mean_error=float((mean-emp).norm()/emp.norm()),covariance_error=float((G-expected).norm()/expected.norm()),minimum_eigenvalue=float(torch.linalg.eigvalsh(G).min()))
 assert max(errors['mean_error'],errors['covariance_error'])<1e-12 and errors['minimum_eigenvalue']>0
 return errors
if __name__=='__main__':
 import json
 from pathlib import Path
 r=check();Path(__file__).with_name('MIXED_SKIP_MOMENTS_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
