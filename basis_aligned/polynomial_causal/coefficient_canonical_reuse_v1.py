"""Uncentered coefficient-space CCA; fixed spectral rank cutoff, no labels."""
import torch

def fit_pair(c11,c22,c12,rank=8,relative_cutoff=1e-8):
    def whiten(c):
        lam,v=torch.linalg.eigh((c+c.T)/2);keep=lam>lam.max()*relative_cutoff
        if int(keep.sum())<rank:raise ValueError('Insufficient coefficient rank')
        return v[:,keep]/lam[keep].sqrt()
    a,b=whiten(c11),whiten(c22);u,s,vh=torch.linalg.svd(a.T@c12@b,full_matrices=False)
    return a@u[:,:rank],b@vh.T[:,:rank],s[:rank]

def cosines(a,b):
    return (a*b).sum(0)/(a.square().sum(0)*b.square().sum(0)).sqrt().clamp_min(1e-30)
