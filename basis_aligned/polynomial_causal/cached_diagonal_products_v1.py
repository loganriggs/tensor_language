"""Fold fixed-writer reuse directly into five-bank preparation."""
import torch

def prepare_split(basis,left,right,down,context,global_state):
    assert torch.equal(basis[...,0,:],global_state['writer'].expand_as(basis[...,0,:]))
    private=basis[...,1:,:].contiguous()
    l1,l2=(private@left.T).unbind(-2);r1,r2=(private@right.T).unbind(-2)
    l0=global_state['left_writer'];r0=global_state['right_writer']
    beta=context['cross_rms'];gamma=context['writer_rms']
    rho0=context['perpendicular_rms']+gamma*context['parallel'].square()
    h01=l0*r1+l1*r0
    hidden=torch.stack([l0*r2+l2*r0,2*l1*r1+2*rho0*h01,
                        l1*r2+l2*r1+4*beta*h01,2*l2*r2+8*gamma*h01],-2)
    varying=hidden@down.T
    return dict(fixed=global_state['self_product'],varying=varying)

def prepare(basis,left,right,down,context,global_state):
    bank=prepare_split(basis,left,right,down,context,global_state)
    fixed=bank['fixed'].expand(*basis.shape[:-2],1,basis.shape[-1])
    return torch.cat([fixed,bank['varying']],-2)

def execute_split(u,bank):
    from response_diagonal_conic_v1 import diagonal_coefficients
    weights=diagonal_coefficients(u)
    return weights[...,:1]*bank['fixed']+torch.einsum('...k,...kd->...d',weights[...,1:],bank['varying'])
