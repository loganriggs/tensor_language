"""Exact Gaussian numerator loss with cached dense teacher and lowrank student.
No O(n^3) student multiplication; matrices C,D are formed for degree<=3 terms.
"""
import torch
from gaussian_quartic_moment import features,cubic_inner
from quartic_pair_metric import inner
from quartic_lowrank_metric import squared_error as quartic_error

def prepare_teacher(A,B):
 f=features(A,B)
 f['quartic_norm']=inner(*f['quartic'],*f['quartic'])
 f['cubic_norm']=sum(cubic_inner(M,u,N,v) for M,u in f['cubic'] for N,v in f['cubic'])
 f['variance']=24*f['quartic_norm']+6*f['cubic_norm']+2*f['quadratic'].square().sum()+f['linear'].square().sum()
 return f

def squared_error_by_degree(teacher,X,a,Y,b):
 xp=X[:-1];yp=Y[:-1]
 C=(xp*a)@xp.T;D=(yp*b)@yp.T
 u=2*xp@(a*X[-1]);v=2*yp@(b*Y[-1])
 ca=(a*X[-1].square()).sum();cb=(b*Y[-1].square()).sum()
 tc=(xp.square().sum(0)*a).sum();td=(yp.square().sum(0)*b).sum()
 CD=((xp*a)@(xp.T@yp*b))@yp.T
 quadratic=(ca+tc)*D+(cb+td)*C+(torch.outer(u,v)+torch.outer(v,u))/2+2*(CD+CD.T)
 linear=(ca+tc)*v+(cb+td)*u+2*(C@v+D@u)
 constant=(ca+tc)*(cb+td)+u@v+2*(C*D).sum()
 terms=[(C,v),(D,u)]
 cubic_norm=sum(cubic_inner(M,l,N,r) for M,l in terms for N,r in terms)
 cubic_cross=sum(cubic_inner(M,l,N,r) for M,l in teacher['cubic'] for N,r in terms)
 q4=24*quartic_error(*teacher['quartic'],teacher['quartic_norm'],xp,a,yp,b)
 q3=6*(teacher['cubic_norm']+cubic_norm-2*cubic_cross)
 q2=2*(teacher['quadratic']-quadratic).square().sum()
 q1=(teacher['linear']-linear).square().sum()
 q0=(teacher['constant']-constant).square()
 return torch.stack([q0,q1,q2,q3,q4])
