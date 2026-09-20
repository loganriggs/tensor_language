"""Exact isotropic Gaussian mean of a two-bilinear-layer homogeneous quartic."""
def mean(C,L2,R2,D1,L1,R1):
 left=L2@D1;right=R2@D1
 tau=(L1*R1).sum(-1)
 gram=.5*((L1@L1.T)*(R1@R1.T)+(L1@R1.T)*(R1@L1.T))
 return C@((left@tau)*(right@tau)+2*((left@gram)*right).sum(-1))
