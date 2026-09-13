"""Exact three-vector fixed-writer response and six-product conditional basis."""
import torch
from directional_mlp_response_context_v1 import prepare


def prepare_basis(z,baseline,program,reentry_scale):
    context=prepare(z,baseline,program)
    basis=torch.stack([context['direction'].expand_as(z),
        context['Jz']-2*context['cross_rms']*context['baseline'],
        context['Jw'].expand_as(z)-2*context['writer_rms']*context['baseline']],-2)*reentry_scale
    return context,basis


def coefficients(amplitude,context):
    a=amplitude.to(context['baseline'])
    rho=context['perpendicular_rms']+context['writer_rms']*(a-context['parallel']).square()
    return torch.cat([-a,-a/rho,a.square()/(2*rho)],-1)


def prepare_products(basis,left,right,down):
    l=basis@left.T;r=basis@right.T
    pairs=torch.triu_indices(basis.shape[-2],basis.shape[-2],device=basis.device)
    products=(l[...,pairs[0],:]*r[...,pairs[1],:]+l[...,pairs[1],:]*r[...,pairs[0],:])@down.T
    return pairs,products


def combine(u,v,pairs,products):
    i,j=pairs
    weights=u[...,i]*v[...,j]+torch.where(i==j,0,u[...,j]*v[...,i])
    return torch.einsum('...k,...kd->...d',weights,products)
