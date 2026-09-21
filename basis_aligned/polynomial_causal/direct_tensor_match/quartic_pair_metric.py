"""Exact symmetric coefficient inner product for products of quadratic forms.

For symmetric A,B,C,D, f=(x.T A x)(x.T B x) and g=(x.T C x)(x.T D x),
this contracts their fully input-symmetrized order-four coefficient tensors.
It is coefficient Frobenius geometry, not Gaussian/probe output MSE.
No degree-four tensor is materialized. Dense cost O(d^3), workspace O(d^2).
"""
import torch

def inner(A,B,C,D):
    return ((A*C).sum()*(B*D).sum()+(A*D).sum()*(B*C).sum()
            +4*torch.trace(A@C@B@D))/6

def squared_error(A,B,C,D):
    return inner(A,B,A,B)+inner(C,D,C,D)-2*inner(A,B,C,D)

def numerator_forms(Qa,Qb,alpha,beta):
    """Homogenize (t-.5qa-alpha*s)(qb-beta*s) with coordinate u=1.

Coordinates [z...,t,s,u]. Both returned matrices are symmetric. The full
normalized feature divides the numerator by s^2; normalization stays explicit.
"""
    d=len(Qa);A=Qa.new_zeros(d+3,d+3);B=Qb.new_zeros(d+3,d+3)
    A[:d,:d]=-.5*Qa;B[:d,:d]=Qb
    A[d,d+2]=A[d+2,d]=.5
    A[d+1,d+2]=A[d+2,d+1]=-.5*alpha
    B[d+1,d+2]=B[d+2,d+1]=-.5*beta
    return A,B
