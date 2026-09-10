"""Implicit exact one-reader least squares for weight-only symmetric products.

T_o = sum_j W_oj sym(a_j b_j^T), output metric M. A is variable,
B and W fixed. No explicit d^2 tensor or (r*d)^2 normal matrix required.
This is an ALS building block, not a full native optimizer or Gauss-Newton.
"""
import torch

def normal_operator(a,b,output_gram,damping=0.):
    bb=b@b.T
    return .5*((output_gram*bb)@a+(output_gram*(b@a.T))@b)+damping*a

def native_rhs(left,right,down,b,writer,metric):
    cross=down.T@metric@writer
    return .5*((cross*(right@b.T)).T@left+(cross*(left@b.T)).T@right)

def block_precondition(v,b,output_gram,damping=0.):
    beta=.5*output_gram.diag()[:,None];bn=b.square().sum(1,keepdim=True)
    alpha=beta*bn+damping
    if bool((alpha<=0).any()):raise ArithmeticError('Nonpositive block preconditioner')
    return v/alpha-beta*b*(b*v).sum(1,keepdim=True)/(alpha*(alpha+beta*bn))

def pcg(operator,rhs,precondition,initial=None,tolerance=1e-10,max_iterations=1000):
    x=torch.zeros_like(rhs) if initial is None else initial.clone()
    r=rhs-operator(x);scale=rhs.norm().clamp_min(torch.finfo(rhs.dtype).tiny)
    z=precondition(r);p=z.clone();rz=(r*z).sum();iterations=0
    while float(r.norm()/scale)>tolerance and iterations<max_iterations:
        ap=operator(p);den=(p*ap).sum()
        if not torch.isfinite(den) or float(den)<=0:raise ArithmeticError('Nonpositive CG curvature')
        step=rz/den;x=x+step*p;r=r-step*ap;z=precondition(r);newrz=(r*z).sum();p=z+(newrz/rz)*p;rz=newrz;iterations+=1
    true_relative_residual=float((rhs-operator(x)).norm()/scale)
    return x,dict(iterations=iterations,true_relative_residual=true_relative_residual,converged=true_relative_residual<=tolerance)
