"""Sparse fixed-support least squares with exact action on one input direction."""
import torch

def fit(matrix,direction,scalar_budget=589824):
    v=direction.to(matrix);v=v/v.norm()
    residual=matrix-(matrix@v)[:,None]*v[None,:]
    bitmap_bytes=(matrix.numel()+7)//8
    count=scalar_budget-bitmap_bytes//4-2*matrix.shape[0]
    indices=residual.abs().flatten().topk(count).indices
    mask=torch.zeros_like(matrix,dtype=torch.bool).flatten();mask[indices]=True
    mask=mask.reshape_as(matrix);raw=residual*mask
    denominator=1-(mask*v.square()).sum(-1)
    assert bool((denominator>0).all())
    sparse=(raw+(raw@v/denominator)[:,None]*v[None,:])*mask
    correction=(matrix-sparse)@v
    return sparse+correction[:,None]*v[None,:]
