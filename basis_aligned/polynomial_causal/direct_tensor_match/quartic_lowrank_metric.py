"""Exact quartic coefficient loss: dense fixed teacher, low-rank student factors.
C=X diag(a) X.T, D=Y diag(b) Y.T. Per-step dense work O(n^2 r), not O(n^3).
Teacher norm is precomputed. All matrices represent symmetric quadratic forms.
"""
import torch

def student_norm(X,a,Y,b):
    xx=X.T@X;yy=Y.T@Y;xy=X.T@Y
    cc=(xx.square()*a[:,None]*a[None,:]).sum()
    dd=(yy.square()*b[:,None]*b[None,:]).sum()
    cd=(xy.square()*a[:,None]*b[None,:]).sum()
    middle=a[:,None]*xy*b[None,:]
    ccdd=(middle*(xx@middle@yy)).sum()  # trace(C^2 D^2)=||CD||F^2
    return (cc*dd+cd.square()+4*ccdd)/6

def teacher_cross(A,B,X,a,Y,b):
    AX=A@X;BX=B@X;AY=A@Y;BY=B@Y
    ac=((X*AX).sum(0)*a).sum();bc=((X*BX).sum(0)*a).sum()
    ad=((Y*AY).sum(0)*b).sum();bd=((Y*BY).sum(0)*b).sum()
    acbd=((X.T@AY)*(X.T@BY)*a[:,None]*b[None,:]).sum()
    return (ac*bd+ad*bc+4*acbd)/6

def squared_error(A,B,teacher_norm,X,a,Y,b):
    return teacher_norm+student_norm(X,a,Y,b)-2*teacher_cross(A,B,X,a,Y,b)
