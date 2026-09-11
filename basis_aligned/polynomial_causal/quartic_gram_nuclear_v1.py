"""ADMM for minimum nuclear norm Gram representation of a fixed quartic.
Signed eigenvalues allowed. Feasible affine projection, explicit dual lower bound.
This convex surrogate is not a guarantee of minimum arithmetic rank.
"""
import time
import torch
from quartic_gram_map_v1 import coefficients,canonical


def solve(target,mapping,iterations=2000,tolerance=1e-7,rho=1.):
    scale=target.norm();assert scale>0;c=target/scale
    z=canonical(c,mapping);u=torch.zeros_like(z);history=[];start=time.perf_counter()
    for iteration in range(iterations):
        v=z-u;x=v+canonical(c-coefficients(v,mapping),mapping)
        value,vector=torch.linalg.eigh((x+u+(x+u).T)/2)
        shrunk=value.sign()*(value.abs()-1/rho).clamp_min(0)
        previous=z;z=(vector*shrunk)@vector.T;u=u+x-z
        if iteration%10==0 or iteration==iterations-1:
            primal=float(torch.linalg.eigvalsh(x).abs().sum())
            dual_coefficient=rho*coefficients(u,mapping);dual_matrix=canonical(dual_coefficient,mapping)
            spectral=float(torch.linalg.eigvalsh(dual_matrix).abs().max());dual_coefficient/=max(1.,spectral)
            dual=float(c@dual_coefficient);gap=(primal-dual)/max(primal,1e-30)
            residual=float((x-z).norm());dual_residual=float(rho*(z-previous).norm())
            history.append(dict(iteration=iteration,primal=primal,dual=dual,relative_gap=gap,primal_residual=residual,dual_residual=dual_residual))
            if gap<=tolerance and gap>=-1e-10 and residual<=tolerance and dual_residual<=tolerance:break
    eig,vec=torch.linalg.eigh(x);keep=eig.abs()>1e-6*eig.abs().max();truncated=(vec[:,keep]*eig[keep])@vec[:,keep].T
    return x*scale,dict(converged=gap<=tolerance and gap>=-1e-10 and residual<=tolerance and dual_residual<=tolerance,
        iterations=iteration+1,relative_gap=gap,primal_residual=residual,dual_residual=dual_residual,
        coefficient_error=float((coefficients(x,mapping)-c).norm()),effective_rank=int(keep.sum()),
        truncated_coefficient_error=float((coefficients(truncated,mapping)-c).norm()),
        nuclear_norm=float(scale)*primal,dual_bound=float(scale)*dual,seconds=time.perf_counter()-start,history=history)
