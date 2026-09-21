"""Exact first two moments of D[(Lz)*(Rz)] for z uniform on sqrt(d) sphere."""
import torch

def moments(left, right, down):
    d=left.shape[1]
    tau=(left*right).sum(1)
    lr=left@right.T
    covariance=d/(d+2)*((left@left.T)*(right@right.T)+lr*lr.T)-2/(d+2)*torch.outer(tau,tau)
    return down@tau, down@covariance@down.T
