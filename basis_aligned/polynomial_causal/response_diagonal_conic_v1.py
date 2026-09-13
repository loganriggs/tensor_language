"""Exact five-vector bank for K(Delta(a),Delta(a)), not arbitrary cross terms."""
import torch

def prepare_diagonal(products,context):
    # Standard upper-triangle order: 00,01,02,11,12,22.
    p00,p01,p02,p11,p12,p22=products.unbind(-2)
    beta=context['cross_rms'];gamma=context['writer_rms']
    rho0=context['perpendicular_rms']+gamma*context['parallel'].square()
    return torch.stack([p00,p02,p11+2*rho0*p01,p12+4*beta*p01,p22+8*gamma*p01],-2)

def diagonal_coefficients(u):
    u0,u1,u2=u.unbind(-1)
    return torch.stack([u0.square(),2*u0*u2,u1.square(),2*u1*u2,u2.square()],-1)

def execute(u,bank):
    return torch.einsum('...k,...kd->...d',diagonal_coefficients(u),bank)

def prepare_direct(basis,left,right,down,context):
    l0,l1,l2=(basis@left.T).unbind(-2)
    r0,r1,r2=(basis@right.T).unbind(-2)
    beta=context['cross_rms'];gamma=context['writer_rms']
    rho0=context['perpendicular_rms']+gamma*context['parallel'].square()
    h01=l0*r1+l1*r0
    hidden=torch.stack([2*l0*r0,l0*r2+l2*r0,2*l1*r1+2*rho0*h01,
                        l1*r2+l2*r1+4*beta*h01,2*l2*r2+8*gamma*h01],-2)
    return hidden@down.T
