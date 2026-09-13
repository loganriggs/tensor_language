"""Complete-even coefficient loss for an arbitrary full-rank sparse basis.

Original restricted objective assumes a subspace of B; this one does not.
Shared query and shared key slots are both symmetrized, before RMS/value.
"""
import torch


def reflect(k, s):
    return k-2*(k@s)@torch.linalg.solve(s.T@s, s.T)


def query_grams(l1, l2):
    return l1.transpose(-1,-2)@l1, l2.transpose(-1,-2)@l2, l1.transpose(-1,-2)@l2


def inner(grams, a1, a2, b1, b2):
    g1,g2,c=grams
    x11,x12,x21,x22=a1@b1.T,a1@b2.T,a2@b1.T,a2@b2.T
    dot=lambda g,x:(g*x).sum(dim=(-1,-2))
    tr=lambda x:x.diagonal(dim1=-2,dim2=-1).sum(-1)
    return (dot(g1,x11)*dot(g2,x22)+dot(c,x12)*dot(c.transpose(-1,-2),x21)
            +tr(c.transpose(-1,-2)@x11@c@x22)
            +tr(g1@x12@g2@x21))/4


def loss(s, grams, k1, k2, original1, original2, norm):
    r1,r2=reflect(k1,s),reflect(k2,s)
    # Reflections are orthogonal, so both product norms equal norm.
    return ((norm-inner(grams,original1,original2,r1,r2))/2).sum()
