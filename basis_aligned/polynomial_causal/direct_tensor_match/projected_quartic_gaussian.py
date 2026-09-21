"""Exact degree-two Gaussian Hermite coefficient in a requested input subspace.

Teacher is C[(l h)(r h)], h=D[(A x)(B x)], x~N(0,I).
Return Z.T (6 Tr H_v) Z, without expanding H or the full dxd output forms.
The second MLP channels are batched to bound temporary storage.
"""
import torch
from quartic_gaussian import mean as teacher_mean


def projected_second(teacher,Z,channel_batch=16):
    if channel_batch<1:raise ValueError('channel_batch must be positive')
    C,l,r,D,A,B=teacher
    left,right=l@D,r@D
    tau=(A*B).sum(-1);ta,tb=left@tau,right@tau
    az,bz=A@Z,B@Z
    result=C.new_zeros((C.shape[0],Z.shape[1]*Z.shape[1]))
    for start in range(0,len(left),channel_batch):
        stop=min(start+channel_batch,len(left));a,b=left[start:stop],right[start:stop]
        # A_t Z and B_t Z, where A_t/B_t are the hidden quadratic forms.
        qa=.5*(A.T@(a[:,:,None]*bz)+B.T@(a[:,:,None]*az))
        qb=.5*(A.T@(b[:,:,None]*bz)+B.T@(b[:,:,None]*az))
        pair=qa.transpose(-1,-2)@qb
        projected=ta[start:stop,None,None]*(Z.T@qb)+tb[start:stop,None,None]*(Z.T@qa)+2*(pair+pair.transpose(-1,-2))
        result.add_(C[:,start:stop]@projected.flatten(1))
    return result.reshape(C.shape[0],Z.shape[1],Z.shape[1])


def functional_cross(teacher,U,V,metadata,pair_batch=8,channel_batch=16):
    from batched_quartic_cross import cross
    fourth=24*cross(teacher,U,V,pair_batch)
    second=2*(projected_second(teacher,metadata['basis'],channel_batch).flatten(1)@metadata['second'].flatten(1).T)
    constant=teacher_mean(*teacher)[:,None]*metadata['mean'][None,:]
    return fourth+second+constant,dict(degree0=constant,degree2=second,degree4=fourth)
