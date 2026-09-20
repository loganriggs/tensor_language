"""Empirical trial/test spaces balancing response and finite-reader snapshots.

This is a snapshot construction, not a stability/error certificate for the model.
"""
import torch


def balanced_frames(responses, readers, rank):
    """Rows are examples; return decoder V and encoder W with W^T V = I.

    Cross-snapshot coupling is truncated jointly. CPU float64 is recommended.
    These need not be orthogonal; downstream code must retain V^T V in RMS.
    """
    X,Y=responses.T,readers.T
    u,s,vh=torch.linalg.svd(Y.T@X,full_matrices=False)
    if rank>len(s) or float(s[rank-1])<=float(s[0])*1e-12:
        raise ValueError('requested balanced rank exceeds numerical coupling rank')
    scale=s[:rank].rsqrt()
    V=(X@vh[:rank].T)*scale
    W=(Y@u[:,:rank])*scale
    return V,W,s
