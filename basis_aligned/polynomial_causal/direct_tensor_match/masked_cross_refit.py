"""Conjugate-gradient coefficient refit for a fixed mask in bilinear coordinates."""
import torch

def solve(left,right,target,mask,ridge=1e-10,steps=1000,tolerance=1e-8):
 A=left.T@left;B=right.T@right;rhs=(left.T@target@right)*mask
 def multiply(x):return (A@x@B+ridge*x)*mask
 x=torch.zeros_like(rhs);residual=rhs.clone();direction=residual.clone();rr=residual.square().sum();initial=rr.sqrt();iterations=0
 if float(initial)==0:return x,dict(iterations=0,relative_normal_residual=0.)
 for iteration in range(steps):
  Hd=multiply(direction);den=(direction*Hd).sum()
  if float(den)<=0:break
  alpha=rr/den;x=x+alpha*direction;residual=residual-alpha*Hd;next_rr=residual.square().sum();iterations=iteration+1
  if float(next_rr.sqrt()/initial)<=tolerance:break
  direction=residual+(next_rr/rr)*direction;rr=next_rr
 actual=float((multiply(x)-rhs).norm()/initial)
 return x,dict(iterations=iterations,relative_normal_residual=actual,converged=actual<=tolerance)
