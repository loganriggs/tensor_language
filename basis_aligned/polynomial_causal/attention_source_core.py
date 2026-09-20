"""Exact readout-contracted two-QK attention for x[b,t]=h[b,t]+K[b,t] a[b].

Retains input/head normalization, rounded RoPE, causal masking and first values.
No native weights are needed by execute; compiled contexts remain dependencies.
"""
def compile_core(weights,background,directions,readers,first,mixture,heads,cos,sin,input_eps,head_eps):
    import torch
    b,t,d=background.shape;hd=d//heads;p=directions.shape[-1]
    global_norm=dict(base=background.square().mean(-1)+input_eps,linear=2*torch.einsum('btd,btdp->btp',background,directions)/d,gram=torch.einsum('btdp,btdq->btpq',directions,directions)/d)
    def branch(weight):
        base=(background@weight.T).reshape(b,t,heads,hd);basis=torch.einsum('vd,btdp->btvp',weight,directions).reshape(b,t,heads,hd,p)
        norm=dict(base=base.square().mean(-1)+head_eps*global_norm['base'][...,None],linear=2*torch.einsum('bthd,bthdp->bthp',base,basis)/hd+head_eps*global_norm['linear'][:,:,None,:],gram=torch.einsum('bthdp,bthdq->bthpq',basis,basis)/hd+head_eps*global_norm['gram'][:,:,None,:,:])
        x,y=base.chunk(2,dim=-1);c=cos[None,:,None,:];s=sin[None,:,None,:];base=torch.cat([x*c+y*s,-x*s+y*c],dim=-1)
        x,y=basis.chunk(2,dim=-2);c=c[...,None];s=s[...,None];basis=torch.cat([x*c+y*s,-x*s+y*c],dim=-2)
        return base,basis,norm
    def score(qname,kname):
        q,qp,qn=branch(weights[qname]);k,kp,kn=branch(weights[kname])
        gram=torch.einsum('bthdp,bshdq->bhtspq',qp,kp);gram=(gram+gram.transpose(-1,-2))/2
        return dict(base=torch.einsum('bthd,bshd->bhts',q,k),linear=torch.einsum('bthdp,bshd->bhtsp',qp,k)+torch.einsum('bthd,bshdp->bhtsp',q,kp),gram=gram,qnorm=qn,knorm=kn)
    output=torch.einsum('btod,dhk->btohk',readers,weights['o'].reshape(d,heads,hd))
    vb=(background@weights['v'].T).reshape(b,t,heads,hd);vk=torch.einsum('vd,btdp->btvp',weights['v'],directions).reshape(b,t,heads,hd,p)
    return dict(score1=score('q','k'),score2=score('q2','k2'),global_norm=global_norm,value_base=(1-mixture)*torch.einsum('btohd,bshd->bohts',output,vb),value_linear=(1-mixture)*torch.einsum('btohd,bshdp->bohtsp',output,vk),cached_value=mixture*torch.einsum('btohd,bshd->bohts',output,first.to(vb).reshape_as(vb)),head_width=hd,length=t)

def polynomial(core,a):
    import torch
    return core['base']+torch.einsum('b...p,bp->b...',core['linear'],a)+torch.einsum('b...pq,bp,bq->b...',core['gram'],a,a)

def execute(core,a):
    import torch
    def score(s):
        q=polynomial(s['qnorm'],a).sqrt().transpose(1,2);k=polynomial(s['knorm'],a).sqrt().transpose(1,2)
        return polynomial(s,a)/(q[...,None]*k[:,:,None,:]*core['head_width'])
    pattern=score(core['score1'])*score(core['score2'])
    mask=torch.ones(core['length'],core['length'],device=a.device,dtype=torch.bool).tril();pattern=pattern.masked_fill(~mask,0.)
    value=(core['value_base']+torch.einsum('bohtsp,bp->bohts',core['value_linear'],a))/polynomial(core['global_norm'],a).sqrt()[:,None,None,None,:]+core['cached_value']
    return (pattern[:,None]*value).sum((-1,-2,-3))
