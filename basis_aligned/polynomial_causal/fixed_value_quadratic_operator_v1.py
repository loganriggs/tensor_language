"""Exact cubic coefficient metric for a fixed linear value times a quadratic.
Sym(v tensor X) has Gram M(X)=||v||² X/3 + ((Xv)vT+v(Xv)T)/3.
M powers have three eigenspaces: perpendicular, cross, and parallel to v.
"""
import torch

def metric_power(x,v,power):
 norm2=v.square().sum();u=v/norm2.sqrt();proj=u[:,None]*u[None,:]
 parallel=torch.einsum('i,...ij,j->...',u,x,u)[...,None,None]*proj
 pu=torch.einsum('ij,...jk->...ik',proj,x);up=pu.transpose(-1,-2)
 cross=pu+up-2*parallel;perp=x-cross-parallel
 return (norm2/3)**power*perp+(2*norm2/3)**power*cross+norm2**power*parallel

def apply(x,a,b,v):
 """Map whitened source quadratic coordinates to query quadratic coefficients."""
 y=metric_power(x,v,.5);z=a@y@b.T
 return (z+z.T)/2

def adjoint(y,a,b,v):
 z=a.T@((y+y.T)/2)@b;z=(z+z.T)/2
 return metric_power(z,v,.5)
