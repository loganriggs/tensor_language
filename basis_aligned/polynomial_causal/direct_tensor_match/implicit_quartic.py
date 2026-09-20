"""Exact queries of fully input-symmetrized two-bilinear-layer coefficient entries.
No expanded order-five tensor is constructed. Inputs are ordered index tuples.
"""
import torch

def entries(C,L2,R2,D1,L1,R1,indices):
 def quadratic_pair(i,j):
  return .5*((L1[:,i]*R1[:,j]+L1[:,j]*R1[:,i]).T@D1.T)
 def outer(s,t):
  return .5*((s@L2.T)*(t@R2.T)+(t@L2.T)*(s@R2.T))@C.T
 i,j,k,l=indices.T
 return (outer(quadratic_pair(i,j),quadratic_pair(k,l))+outer(quadratic_pair(i,k),quadratic_pair(j,l))+outer(quadratic_pair(i,l),quadratic_pair(j,k)))/3
