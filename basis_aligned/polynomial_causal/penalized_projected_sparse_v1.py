"""Component-penalized variable projection on unit dictionary/code rows.

Same fixed sparse graph; outputs absorb reader-row scales. Positive penalty
permits an exact normalized-Gram Cholesky solve. No output truncation/ridge
substitution: the penalty is explicitly sum of individual tensor energies.
"""
import torch
from folded_sparse_dictionary_v1 import decode
from joint_quadratic_fit_v1 import product_cross
from chunked_bilinear_coefficient_v1 import value_gradient as cp_gradient


def output_solve(d,l,r,a,b,penalty):
    if penalty<=0:raise ValueError('Positive component penalty required')
    gram=product_cross(a,b,a,b);norms=gram.diag().sqrt()
    if not bool((norms>1e-12).all()):raise ArithmeticError('Vanishing product')
    normalized=gram/norms[:,None]/norms[None,:]
    regularized=(normalized+normalized.T)/2+penalty*torch.eye(len(a),device=a.device,dtype=a.dtype)
    chol=torch.linalg.cholesky(regularized)
    rhs=d@product_cross(l,r,a,b)
    scaled=torch.cholesky_solve((rhs/norms[None,:]).T,chol).T
    writer=scaled/norms[None,:]
    normal=float((scaled@regularized-rhs/norms[None,:]).norm()/(rhs/norms[None,:]).norm().clamp_min(1e-30))
    return writer,dict(normal_residual=normal,product_norm_min=float(norms.min()),product_norm_max=float(norms.max()))


@torch.no_grad()
def value_gradient(native,whitener,raw_basis,indices,raw_values,total,penalty,chunk=256):
    lengths=raw_values.norm(dim=1,keepdim=True)
    if not bool((lengths>1e-12).all()):raise ArithmeticError('Vanishing code row')
    values=raw_values/lengths
    readers,basis,code,bnorms=decode(raw_basis,indices,values,torch.ones(len(values),device=values.device,dtype=values.dtype))
    a,b=readers.chunk(2);l,r,d=native
    writer,solver=output_solve(d,l,r,a,b,penalty)
    loss,(ga,gb,_),details=cp_gradient(l,r,whitener@d,a,b,whitener@writer,total,penalty=penalty,chunk=chunk)
    gr=torch.cat((ga,gb))
    gc=(gr@basis.T).gather(1,indices);ct=gc-(gc*values).sum(1,keepdim=True)*values
    gbasis=torch.sparse.mm(code.transpose(0,1),gr)
    bt=gbasis-(gbasis*basis).sum(1,keepdim=True)*basis
    denominator=abs(float(loss))
    details.update(dictionary_relative_stationarity=float(bt.norm()*len(basis)**.5/max(denominator,1e-12)),
        codes_relative_stationarity=float(ct.norm()*len(values)**.5/max(denominator,1e-12)),
        output_solve=solver,penalty=penalty,parameterization='unit dictionary rows and unit code rows')
    return loss,(bt/bnorms,ct/lengths),writer,details
