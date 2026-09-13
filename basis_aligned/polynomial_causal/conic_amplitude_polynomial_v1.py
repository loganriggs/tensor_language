"""Exact degree-four numerator of the producer-constrained quadratic response."""
import torch


def compile_degree2(left_basis, right_basis, down, context):
    """Prepare only three output vectors, without preparing the exact five-bank."""
    beta=context['cross_rms'];gamma=context['writer_rms']
    rho0=context['perpendicular_rms']+gamma*context['parallel'].square()
    scale=(rho0/gamma).sqrt()
    l0,l1,l2=left_basis.unbind(-2);r0,r1,r2=right_basis.unbind(-2)
    a0=-rho0*l0-l1;b0=-rho0*r0-r1
    a1=2*beta*l0+l2/2;b1=2*beta*r0+r2/2
    a2=-gamma*l0;b2=-gamma*r0
    hidden=torch.stack([2*a0*b0,2*(a0*b1+a1*b0)*scale,
        (2*a1*b1+2*(a0*b2+a2*b0))*scale.square()],-2)
    return hidden@down.T,scale


def compile_numerator(bank, context):
    """Return [..,5,d] coefficients in dimensionless t=a/sqrt(rho0/gamma)."""
    q0,q1,q2,q3,q4=bank.unbind(-2)
    beta=context['cross_rms'];gamma=context['writer_rms']
    rho0=context['perpendicular_rms']+gamma*context['parallel'].square()
    scale=(rho0/gamma).sqrt()
    coefficients=torch.stack([
        rho0.square()*q0+q2,
        -4*rho0*beta*q0-rho0*q1-q3,
        (4*beta.square()+2*rho0*gamma)*q0+2*beta*q1+q4/4,
        -4*beta*gamma*q0-gamma*q1,
        gamma.square()*q0],-2)
    powers=torch.arange(5,device=bank.device)
    return coefficients*scale.unsqueeze(-1).pow(powers[:,None]),scale


def evaluate(t, coefficients, scale, context, degree=4):
    """t ends in a singleton dimension; preserve the exact rational denominator."""
    value=coefficients[...,degree,:]
    for k in range(degree-1,-1,-1):value=value*t+coefficients[...,k,:]
    a=t*scale
    rho=context['perpendicular_rms']+context['writer_rms']*(a-context['parallel']).square()
    return a.square()/rho.square()*value


def absolute_bound(t, coefficients, scale, context, degree):
    """Triangle-inequality bound on the Euclidean norm of omitted terms."""
    a=t*scale
    rho=context['perpendicular_rms']+context['writer_rms']*(a-context['parallel']).square()
    tail=sum(t.abs().pow(k)*coefficients[...,k,:].norm(dim=-1,keepdim=True)
             for k in range(degree+1,5))
    return a.square()/rho.square()*tail
