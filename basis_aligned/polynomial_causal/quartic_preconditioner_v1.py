"""Regularized pullback of symmetric-quadratic Frobenius geometry."""
import torch

def precondition(b,n,gb,gn,ridge_factor=.01):
 omega=b.transpose(-1,-2)@gb;omega=(omega-omega.transpose(-1,-2))/2
 perpendicular=gb-b@omega;ridge=ridge_factor/n.shape[-1]
 gap=n[:,:,None]-n[:,None,:]
 return b@(omega/(gap.square()+ridge))+perpendicular/(2*n.square()[:,None,:]+ridge),gn

def metric(b,n,db,dn,eb,en,ridge_factor=.01):
 omega=b.transpose(-1,-2)@db;theta=b.transpose(-1,-2)@eb;k=db-b@omega;l=eb-b@theta
 gap=n[:,:,None]-n[:,None,:];ridge=ridge_factor/n.shape[-1]
 return (dn*en).sum()+((gap.square()+ridge)*omega*theta).sum()+((2*n.square()[:,None,:]+ridge)*k*l).sum()
