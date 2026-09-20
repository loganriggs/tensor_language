"""Exact covariance of a quartic teacher with reusable quadratic product nodes."""
import torch
from gaussian_quartic_mean import native_mean,quadratic_moments

def cross_covariance(teacher,mu,M,A,B):
    mean=lambda center:native_mean(teacher,center,M)
    columns=[]
    for a,b in zip(A,B):
        va,vb=M@a,M@b
        da=torch.func.jvp(mean,(mu,),(va,))[1]
        db=torch.func.jvp(mean,(mu,),(vb,))[1]
        dab=torch.func.jvp(lambda center:torch.func.jvp(mean,(center,),(va,))[1],(mu,),(vb,))[1]
        columns.append((a@mu)*db+(b@mu)*da+dab)
    return torch.stack(columns,1)

def check():
    import numpy as np
    torch.set_num_threads(2);torch.manual_seed(260929);dtype=torch.float64
    teacher=[torch.randn(*s,dtype=dtype) for s in [(2,3),(3,4),(3,4),(4,5),(5,3),(5,3)]]
    mu=torch.randn(3,dtype=dtype);raw=torch.randn(3,3,dtype=dtype);M=raw@raw.T+.2*torch.eye(3,dtype=dtype)
    A,B=[torch.randn(4,3,dtype=dtype) for _ in range(2)]
    nodes,weights=np.polynomial.hermite.hermgauss(4);nodes=torch.tensor(nodes*2**.5);weights=torch.tensor(weights/np.pi**.5);idx=torch.cartesian_prod(*[torch.arange(4)]*3);x=nodes[idx]@torch.linalg.cholesky(M).T+mu;w=weights[idx].prod(1)
    C,L,R,D,l,r=teacher;h=((x@l.T)*(x@r.T))@D.T;y=((h@L.T)*(h@R.T))@C.T;p=(x@A.T)*(x@B.T)
    m,G=quadratic_moments(A,B,mu,M);expected=(w[:,None]*y).T@(p-m);actual=cross_covariance(teacher,mu,M,A,B)
    error=float((actual-expected).norm()/expected.norm());assert error<1e-12
    W=torch.linalg.solve(G,actual.T).T;residual=y-(p-m)@W.T
    orthogonality=float(((w[:,None]*residual).T@(p-m)).norm()/expected.norm());assert orthogonality<1e-12
    return dict(cross_covariance_relative_error=error,corrected_residual_covariance_relative_error=orthogonality,scope='Exact noncentral Gaussian cross moments checked against independent degree-six quadrature; no native fit yet.')

if __name__=='__main__':
    import json
    from pathlib import Path
    r=check();Path(__file__).with_name('GAUSSIAN_QUADRATIC_SKIP_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
