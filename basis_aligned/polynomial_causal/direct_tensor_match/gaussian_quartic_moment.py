"""Exact Gaussian numerator error with a deterministic final coordinate.
Input is (g,1), g~N(0,I). A/B are augmented symmetric quadratic matrices.
Uses orthogonal Gaussian Hermite degrees, avoiding an explicit order-eight M.
This is NOT a Gaussian expectation of division by the sampled RMS coordinate.
"""
import torch
from quartic_pair_metric import squared_error as quartic_error,inner as quartic_inner

def features(A,B):
 C=A[:-1,:-1];D=B[:-1,:-1]
 u=2*A[:-1,-1];v=2*B[:-1,-1]
 a=A[-1,-1];b=B[-1,-1]
 tc=C.trace();td=D.trace();CD=C@D
 quadratic=(a+tc)*D+(b+td)*C+(torch.outer(u,v)+torch.outer(v,u))/2+2*(CD+CD.T)
 linear=(a+tc)*v+(b+td)*u+2*(C@v+D@u)
 constant=(a+tc)*(b+td)+u@v+2*(C*D).sum()
 return dict(quartic=(C,D),cubic=[(C,v),(D,u)],quadratic=quadratic,linear=linear,constant=constant)

def cubic_inner(A,u,B,v):
 return ((A*B).sum()*(u@v)+2*(u@(B@(A@v))))/3

def squared_error_by_degree(A,B,C,D):
 teacher=features(A,B);student=features(C,D)
 q4=24*quartic_error(*teacher['quartic'],*student['quartic'])
 terms=[(M,u,1.) for M,u in teacher['cubic']]+[(M,u,-1.) for M,u in student['cubic']]
 q3=6*sum(sa*sb*cubic_inner(M,u,N,v) for M,u,sa in terms for N,v,sb in terms)
 q2=2*(teacher['quadratic']-student['quadratic']).square().sum()
 q1=(teacher['linear']-student['linear']).square().sum()
 q0=(teacher['constant']-student['constant']).square()
 return torch.stack([q0,q1,q2,q3,q4])

def teacher_norm_by_degree(A,B):
 f=features(A,B);q4=24*quartic_inner(*f['quartic'],*f['quartic'])
 q3=6*sum(cubic_inner(M,u,N,v) for M,u in f['cubic'] for N,v in f['cubic'])
 return torch.stack([f['constant'].square(),f['linear'].square().sum(),2*f['quadratic'].square().sum(),q3,q4])
