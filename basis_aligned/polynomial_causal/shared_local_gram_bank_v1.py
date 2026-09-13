"""Exact leading right subspace via the smaller Gram; direct SVD fallback.

Gram squaring can lose tiny singular values. This is for leading subspaces only;
ill-conditioned requested eigenvalues and undersized groups use direct SVD.
"""
import torch
from shared_local_subspaces_v1 import bank as svd_bank


def gram_bank(x, rank):
    n,d=x.shape
    if min(n,d)<rank:
        return svd_bank(x,rank)
    small = x.T@x if n>=d else x@x.T
    values,vectors=torch.linalg.eigh((small+small.T)/2)
    values=values.flip(0)[:rank];vectors=vectors.flip(1)[:,:rank]
    threshold=torch.finfo(x.dtype).eps*max(n,d)*values[0].abs()
    if bool(values[-1]<=threshold):
        return svd_bank(x,rank)
    q=vectors.T if n>=d else (vectors.T@x)/values.sqrt()[:,None]
    return torch.linalg.qr(q.T,mode='reduced')[0].T
