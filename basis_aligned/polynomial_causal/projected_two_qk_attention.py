"""Exact conditional attention contraction for x=background+Pz.

Both QK products share z, and values use the same input. The input RMS and head
Q/K RMS denominators remain explicit; RoPE is applied before score contraction.
Contexts are large, charged dependencies. This module does not discover a basis.
"""
import torch


def _rotate_base(x,cos,sin):
    a,b=x.chunk(2,dim=-1)
    c,s=cos[None,:,None,:],sin[None,:,None,:]
    return torch.cat([a*c+b*s,-a*s+b*c],dim=-1)


def _rotate_basis(x,cos,sin):
    a,b=x.chunk(2,dim=-2)
    c,s=cos[:,None,:,None],sin[:,None,:,None]
    return torch.cat([a*c+b*s,-a*s+b*c],dim=-2)


def compile_attention(weights,background,P,Q,first_value,mixture,heads,cos,sin,
                      input_eps,head_eps):
    """weights q/k/q2/k2/v/o use torch Linear's [out,in] convention.

    P[d,r] decodes shared input coordinates. Q[d,a] reads projected output.
    first_value[B,T,H,Dh] is fixed at this conditional boundary.
    """
    batch,length,width=background.shape;hd=width//heads;r=P.shape[1]
    global_norm=dict(base=background.square().mean(-1)+input_eps,
        linear=2*(background@P)/width,gram=(P.T@P)/width)
    def branch(weight):
        base=(background@weight.T).reshape(batch,length,heads,hd)
        basis=(weight@P).reshape(heads,hd,r)
        norm=dict(base=base.square().mean(-1)+head_eps*global_norm['base'][...,None],
            linear=2*torch.einsum('bthk,hkp->bthp',base,basis)/hd+head_eps*global_norm['linear'][:,:,None,:],
            gram=torch.einsum('hkp,hkq->hpq',basis,basis)/hd+head_eps*global_norm['gram'][None])
        rotated_base=_rotate_base(base,cos,sin)
        rotated_basis=_rotate_basis(basis[None].expand(length,-1,-1,-1),cos,sin)
        return rotated_base,rotated_basis,norm
    def score(qweight,kweight):
        q,qb,qn=branch(qweight);k,kb,kn=branch(kweight)
        return dict(base=torch.einsum('bthk,bshk->bhts',q,k),
            query=torch.einsum('thkp,bshk->bhtsp',qb,k),
            key=torch.einsum('bthk,shkq->bhtsq',q,kb),
            joint=torch.einsum('thkp,shkq->htspq',qb,kb),qnorm=qn,knorm=kn)
    output=torch.einsum('da,dhk->hak',Q,weights['o'].reshape(width,heads,hd))
    raw_value=(background@weights['v'].T).reshape(batch,length,heads,hd)
    value_basis=(weights['v']@P).reshape(heads,hd,r)
    return dict(score1=score(weights['q'],weights['k']),score2=score(weights['q2'],weights['k2']),
        value_base=(1-mixture)*torch.einsum('bthk,hak->btha',raw_value,output),
        value_basis=(1-mixture)*torch.einsum('hkp,hak->hap',value_basis,output),
        cached_value=mixture*torch.einsum('bthk,hak->btha',first_value,output),
        global_norm=global_norm,head_width=hd,length=length)


def _head_norm(norm,z):
    return (norm['base']+torch.einsum('bthp,btp->bth',norm['linear'],z)
        +torch.einsum('hpq,btp,btq->bth',norm['gram'],z,z)).sqrt().transpose(1,2)


def _score(program,z,head_width):
    numerator=program['base']+torch.einsum('bhtsp,btp->bhts',program['query'],z)
    numerator=numerator+torch.einsum('bhtsq,bsq->bhts',program['key'],z)
    numerator=numerator+torch.einsum('htspq,btp,bsq->bhts',program['joint'],z,z)
    return numerator/(_head_norm(program['qnorm'],z)[...,None]
                     *_head_norm(program['knorm'],z)[:,:,None,:]*head_width)


def execute(program,z):
    """Return Q^T attention(RMS(background+Pz)), not a softmax attention."""
    first=_score(program['score1'],z,program['head_width'])
    second=_score(program['score2'],z,program['head_width'])
    pattern=first*second
    mask=torch.ones(program['length'],program['length'],device=z.device,dtype=torch.bool).tril()
    pattern=pattern.masked_fill(~mask,0)
    g=program['global_norm']
    scale=(g['base']+(g['linear']*z).sum(-1)+torch.einsum('pq,btp,btq->bt',g['gram'],z,z)).sqrt()
    value=(program['value_base']+torch.einsum('hap,btp->btha',program['value_basis'],z))/scale[:,:,None,None]
    value=value+program['cached_value']
    return torch.einsum('bhts,bsha->bta',pattern,value)
