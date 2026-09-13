"""Compute a shared projection once across group choices, same joint LS fit.

Requires row-orthonormal global/private banks. Near-dependent joint spans use
the original encoder so coefficient gauge and numerical-rank semantics survive.
"""
import torch
from shared_local_subspaces_v1 import encode as reference_encode


def reused_encode(x, p, locals_):
    global_read=x@p.T
    decompositions=[];scores=[]
    for q in locals_:
        overlap=q@p.T
        h=q-overlap@p
        u,s,vh=torch.linalg.svd(h,full_matrices=False)
        if bool(s[-1]<1e-7):
            return reference_encode(x,p,locals_)
        read=x@vh.T
        scores.append(read.square().sum(1))
        decompositions.append((u,s,vh,overlap))
    labels=torch.stack(scores,1).argmax(1)
    ac=global_read.clone();bc=x.new_zeros(len(x),len(locals_[0]))
    for k,(u,s,vh,overlap) in enumerate(decompositions):
        ix=labels==k
        code=((x[ix]@vh.T)/s)@u.T
        bc[ix]=code;ac[ix]-=code@overlap
    return dict(global_bank=p,local_banks=locals_,global_codes=ac,local_codes=bc,labels=labels)
