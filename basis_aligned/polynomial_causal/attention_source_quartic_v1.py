"""Exact attention-gain response of a shared-parent quadratic program.

The downstream denominator remains an explicit fixed port. This is not a full
model attention ablation. Source coefficients retain signs and cross terms.
"""
import torch

def unpack(program, like):
    d=program['dimension'];ij=torch.triu_indices(d,d,device=like.device)
    packed=program['upper_coefficients'].to(device=like.device,dtype=like.dtype)
    values=packed/torch.where(ij[0]==ij[1],1.,2.)[None,:]
    forms=like.new_zeros(len(packed),d,d)
    forms[:,ij[0],ij[1]]=values;forms[:,ij[1],ij[0]]=values
    return forms

def compile_sources(program, residual, attention):
    forms=unpack(program,residual)
    rr=torch.einsum('ni,kij,nj->nk',residual,forms,residual)
    ra=2*torch.einsum('ni,kij,nj->nk',residual,forms,attention)
    aa=torch.einsum('ni,kij,nj->nk',attention,forms,attention)
    q=torch.stack([rr,ra,aa],-1)
    sectors=residual.new_zeros(len(residual),len(program['branches']),5)
    for i in range(3):
        for j in range(3):sectors[:,:,i+j]+=q[:,0,i,None]*q[:,1:,j]
    norm=torch.stack([residual.square().mean(-1)+torch.finfo(torch.float32).eps,
                      2*(residual*attention).mean(-1),attention.square().mean(-1)],-1)
    return dict(sectors=sectors,norm=norm,quadratic_sources=q)

def evaluate_gain(program, compiled, gain, downstream_denominator):
    sectors=compiled['sectors'];powers=sectors.new_tensor([gain**k for k in range(5)])
    numerator=sectors@powers;norm=compiled['norm']@powers[:3]
    assert bool((norm>0).all())
    alpha=numerator/(norm.square()*downstream_denominator.to(norm))[:,None]
    writers=program['writers'].to(device=sectors.device,dtype=sectors.dtype)
    return {branch:alpha[:,j,None]*writers[:,j][None,:] for j,branch in enumerate(program['branches'])}
