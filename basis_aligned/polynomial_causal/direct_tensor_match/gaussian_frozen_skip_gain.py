"""Gain of a frozen, possibly miscentered correction under a Gaussian law."""
import torch

def mode_gain(K,G,W,residual_mean,feature_mean_offset,U):
    k=U.T@K;w=U.T@W;b=U.T@residual_mean;delta=w@feature_mean_offset
    return 2*(w*k).sum(1)-((w@G)*w).sum(1)+2*b*delta-delta.square()

def check():
    import numpy as np
    from gaussian_quartic_mean import native_mean,quadratic_moments
    from gaussian_quadratic_skip import cross_covariance
    from frozen_program_evaluation import native
    torch.set_num_threads(2);torch.manual_seed(260933);dtype=torch.float64
    teacher=[torch.randn(*s,dtype=dtype) for s in [(2,3),(3,4),(3,4),(4,5),(5,3),(5,3)]]
    base=[t*.8 for t in teacher];mu=torch.randn(3,dtype=dtype);raw=torch.randn(3,3,dtype=dtype);M=raw@raw.T+.2*torch.eye(3,dtype=dtype);A,B=[torch.randn(2,3,dtype=dtype) for _ in range(2)];W=torch.randn(2,2,dtype=dtype);constant=torch.randn(2,dtype=dtype);c=torch.randn(2,dtype=dtype);U=torch.randn(2,3,dtype=dtype)
    mp,G=quadratic_moments(A,B,mu,M);K=cross_covariance(teacher,mu,M,A,B)-cross_covariance(base,mu,M,A,B);b=native_mean(teacher,mu,M)-native_mean(base,mu,M)-constant;actual=mode_gain(K,G,W,b,mp-c,U)
    n,w=np.polynomial.hermite.hermgauss(4);n=torch.tensor(n*2**.5);w=torch.tensor(w/np.pi**.5);idx=torch.cartesian_prod(*[torch.arange(4)]*3);x=n[idx]@torch.linalg.cholesky(M).T+mu;weights=w[idx].prod(1);e=(native(teacher,x)-native(base,x)-constant)@U;delta=(((x@A.T)*(x@B.T)-c)@W.T)@U;expected=(weights[:,None]*(2*e*delta-delta.square())).sum(0)
    error=float((actual-expected).norm()/expected.norm());assert error<1e-10
    return dict(shifted_centering_relative_error=error,scope='Independent degree-six quadrature with nonzero input mean, residual constant, fixed mismatched correction centering and nonorthogonal output projections.')
if __name__=='__main__':
    import json
    from pathlib import Path
    r=check();Path(__file__).with_name('FROZEN_SKIP_GAIN_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
