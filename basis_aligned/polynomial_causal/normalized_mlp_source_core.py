"""Exact weight-contracted bilinear MLP readouts on an affine source interface.

RMS denominator remains explicit. Readers are fixed for this local operation;
variation of downstream readers belongs to separate chain-rule terms.
"""
def compile_core(left,right,down,bias,background,directions,readers,eps):
    import torch
    d=background.shape[-1]
    l=background@left.T;r=background@right.T
    lk=torch.einsum('md,...dp->...mp',left,directions)
    rk=torch.einsum('md,...dp->...mp',right,directions)
    c=torch.einsum('...od,dm->...om',readers,down)
    n0=torch.einsum('...m,...om,...m->...o',l,c,r)
    n1=torch.einsum('...mp,...om,...m->...op',lk,c,r)+torch.einsum('...m,...om,...mp->...op',l,c,rk)
    n2=torch.einsum('...mp,...om,...mq->...opq',lk,c,rk);n2=(n2+n2.transpose(-1,-2))/2
    return dict(n0=n0,n1=n1,n2=n2,bias=torch.einsum('...od,d->...o',readers,bias),s0=background.square().mean(-1)+eps,s1=2*torch.einsum('...d,...dp->...p',background,directions)/d,s2=torch.einsum('...dp,...dq->...pq',directions,directions)/d)

def evaluate(core,amplitudes):
    import torch
    a=amplitudes
    numerator=core['n0']+torch.einsum('...op,...p->...o',core['n1'],a)+torch.einsum('...p,...opq,...q->...o',a,core['n2'],a)
    denominator=core['s0']+torch.einsum('...p,...p->...',core['s1'],a)+torch.einsum('...p,...pq,...q->...',a,core['s2'],a)
    return numerator/denominator[...,None]+core['bias']

def derivatives_at_zero(core):
    n0,n1,n2,s0,s1,s2=(core[k] for k in ['n0','n1','n2','s0','s1','s2'])
    gradient=n1/s0[...,None,None]-n0[...,None]*s1[...,None,:]/s0[...,None,None]**2
    hessian=2*n2/s0[...,None,None,None]
    hessian=hessian-(n1[..., :, :,None]*s1[...,None,None,:]+s1[...,None,:,None]*n1[..., :,None,:])/s0[...,None,None,None]**2
    hessian=hessian-2*n0[...,None,None]*s2[...,None,:,:]/s0[...,None,None,None]**2
    hessian=hessian+2*n0[...,None,None]*s1[...,None,:,None]*s1[...,None,None,:]/s0[...,None,None,None]**3
    return gradient,hessian
