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


def balanced_frames_qr(responses, readers, rank):
    """Equivalent snapshot balancing without forming the sample-by-sample matrix.

    Reduced QR bounds the central SVD by residual width on both axes. This is
    exact linear algebra, not a randomized approximation or nonlinear certificate.
    """
    rx=torch.linalg.qr(responses,mode='reduced').R
    ry=torch.linalg.qr(readers,mode='reduced').R
    u,s,vh=torch.linalg.svd(ry@rx.T,full_matrices=False)
    if rank>len(s) or float(s[rank-1])<=float(s[0])*1e-12:
        raise ValueError('requested balanced rank exceeds numerical coupling rank')
    scale=s[:rank].rsqrt()
    return (rx.T@vh[:rank].T)*scale,(ry.T@u[:,:rank])*scale,s
