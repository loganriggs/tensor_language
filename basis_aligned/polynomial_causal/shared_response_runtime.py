"""Portable conditional response DAG with shared unordered product banks.

Context generation remains a separate, charged dependency. No model is accessed.
"""
import torch
from projected_bilinear_response import readout_prepared


def compile_runtime(programs, scales, final_readout):
    if len(programs)!=len(scales):raise ValueError('one residual scale per block required')
    blocks=[]
    for p,scale in zip(programs,scales):
        n=p['core'].shape[-1]
        i,j=torch.triu_indices(n,n,device=p['core'].device)
        coefficients=p['core'][:,i,j]*torch.where(i==j,1,2)
        blocks.append(dict(coefficients=coefficients.clone(),carry=p['carry'].clone(),
            geometry=p['geometry'].clone(),width=p['width'],inputs=n,scale=scale.clone()))
    return dict(blocks=blocks,readout=final_readout)


def step(block, z, context, innovation=None):
    """One normalized bilinear response with a pre-MLP additive write."""
    b,c=block,context
    z=z*b['scale']
    if innovation is not None:z=z+innovation
    i,j=torch.triu_indices(b['inputs'],b['inputs'],device=z.device)
    products=z[...,i]*z[...,j]
    quadratic=products@b['coefficients'].T
    mixed=torch.einsum('...ap,...p->...a',c['linear'],z)
    shift=2*(c['overlap']*z).sum(-1,keepdim=True)
    shift+=torch.einsum('...p,pq,...q->...',z,b['geometry'],z)[...,None]
    s1=c['s0']+shift/b['width']
    return z@b['carry'].T+(mixed+quadratic)/s1+c['baseline_write']*(c['s0']/s1-1)


def execute(runtime, initial_coordinates, contexts, final_context, innovations=None):
    """Apply optional response writes after residual scaling, before each MLP.

    Innovations use that block's input coordinates. Their generators and any
    projection error are separate charged dependencies.
    """
    if len(runtime['blocks'])!=len(contexts):raise ValueError('one context per block required')
    if innovations is None:innovations=[None]*len(contexts)
    if len(innovations)!=len(contexts):raise ValueError('one innovation slot per block required')
    z=initial_coordinates
    for b,c,innovation in zip(runtime['blocks'],contexts,innovations):
        z=step(b,z,c,innovation)
    return readout_prepared(runtime['readout'],z,final_context)
