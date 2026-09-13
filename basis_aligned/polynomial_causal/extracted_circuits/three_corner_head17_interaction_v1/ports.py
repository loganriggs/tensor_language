"""Normalized final-query attention ports from linear raw-state projections."""
import torch
from .rotary import rotary

EPS = torch.finfo(torch.float32).eps


def project(raw, matrices):
    return tuple(raw @ matrix.T for matrix in matrices)


def from_projections(projections, rho_squared, first_values, mixture):
    """Shapes: five [...,T,128] projections; rho [...,T]; first [...,T,128]."""
    q1, k1, q2, k2, value = projections
    def normalize(x):
        return x/(x.square().mean(-1)+EPS*rho_squared).sqrt()[..., None]
    width = q1.shape[-1]
    rotations = torch.stack([rotary(s, width) for s in range(q1.shape[-2])]).to(q1)
    def score(q, k):
        qr = normalize(q)[..., -1, :] @ rotations[-1].T
        kr = torch.einsum('...td,ted->...te', normalize(k), rotations)
        return (qr[..., None, :]*kr).sum(-1)/width
    a, b = score(q1, k1), score(q2, k2)
    values = (1-mixture)*value/rho_squared.sqrt()[..., None]+mixture*first_values
    return a, b, values


def additive(native, child, remainder):
    return tuple(c+r-n for n, c, r in zip(native, child, remainder))
