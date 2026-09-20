"""Symmetric CP quadratic Frobenius least-squares input block operators."""
import torch

def input_operator(C,B):
 G=C.T@C;GB=G*(B@B.T)
 def apply(A):return .5*(GB@A+(G*(B@A.T))@B)
 diagonal=.5*G.diag()[:,None]*(B.square().sum(-1)[:,None]+B.square())
 return apply,diagonal

def input_rhs(C,B,teacher_C,teacher_A,teacher_B):
 cross=C.T@teacher_C
 return .5*((cross*(B@teacher_B.T))@teacher_A+(cross*(B@teacher_A.T))@teacher_B)

def conjugate_gradient(apply,rhs,initial,diagonal,tolerance=1e-10,max_steps=500):
 x=initial.clone();residual=rhs-apply(x);den=rhs.norm().clamp_min(1e-30);z=residual/diagonal;p=z.clone();rho=(residual*z).sum();history=[]
 for step in range(max_steps):
  relative=float(residual.norm()/den);history.append(relative)
  if relative<tolerance:break
  hp=apply(p);curvature=(p*hp).sum()
  if float(curvature)<=0:raise RuntimeError('Nonpositive CG curvature')
  alpha=rho/curvature;x=x+alpha*p;residual=residual-alpha*hp;z=residual/diagonal;next_rho=(residual*z).sum();p=z+(next_rho/rho)*p;rho=next_rho
 return x,dict(iterations=step+1,recurrence_relative_residual=history[-1],true_relative_residual=float((rhs-apply(x)).norm()/den))
