"""PSD implementation of joint_quadratic_fit_v1.optimal_writers, with diagnostics."""
import torch
from joint_quadratic_fit_v1 import product_cross

def solve(d,l,r,a,b,rtol=1e-10):
    cross=product_cross(l,r,a,b)
    gram=product_cross(a,b,a,b)
    values,vectors=torch.linalg.eigh(gram)
    cutoff=values[-1]*rtol
    assert values[-1]>0
    assert values[0]>=-values[-1]*1e-10
    keep=values>cutoff
    basis=vectors[:,keep]
    rhs=d@cross
    writer=((rhs@basis)/values[keep])@basis.T
    normal=float((writer@gram-rhs).norm()/rhs.norm().clamp_min(1e-30))
    diagnostics=dict(rank=int(keep.sum()),size=gram.shape[0],rtol=rtol,
        minimum_eigenvalue=float(values[0]),maximum_eigenvalue=float(values[-1]),
        retained_condition=float(values[-1]/values[keep][0]),normal_residual=normal)
    return writer,cross,gram,diagnostics
