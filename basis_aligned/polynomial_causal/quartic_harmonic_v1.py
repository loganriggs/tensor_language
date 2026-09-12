"""Canonical radial/quadratic/harmonic-quartic decomposition from tensor traces."""
import torch

def lower(trace):
 d=trace.shape[-1];t=trace.diagonal(dim1=-2,dim2=-1).sum(-1)
 matrix=6*(trace-t[...,None,None]*torch.eye(d,device=trace.device,dtype=trace.dtype)/d)/(d+4)
 constant=3*t/(d*(d+2))
 return matrix,constant

def evaluate_lower(trace,x):
 matrix,constant=lower(trace);radius2=x.square().sum(-1)
 quadratic=torch.einsum('nd,ode,ne->no',x,matrix,x)*radius2[:,None]
 radial=radius2[:,None].square()*constant
 return radial,quadratic
