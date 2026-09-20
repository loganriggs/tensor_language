"""Exact two-endpoint scalar-observable pullback through two-QK attention.

Requires both endpoints; this is an attribution/calibration operator, never an
edited-state predictor. Cached first-layer values are held fixed at this boundary.
"""
import torch


def rms(x,eps):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()


def rms_pullback(x0,x1,g,eps):
    r0=(x0.square().mean(-1,keepdim=True)+eps).sqrt()
    r1=(x1.square().mean(-1,keepdim=True)+eps).sqrt()
    mid=(x0+x1)/2;d=x0.shape[-1]
    return .5*(1/r0+1/r1)*g-2*mid*(mid*g).sum(-1,keepdim=True)/(d*r0*r1*(r0+r1))


def rotate(x,cos,sin,transpose=False):
    a,b=x.chunk(2,dim=-1);c=cos[None,:,None,:];s=sin[None,:,None,:]
    if transpose:return torch.cat([a*c-b*s,a*s+b*c],dim=-1)
    return torch.cat([a*c+b*s,-a*s+b*c],dim=-1)


def forward(weights,x,first_value,mixture,heads,cos,sin,input_eps,head_eps):
    batch,length,width=x.shape;hd=width//heads
    n=rms(x,input_eps);raw={};branches={}
    for name in ['q','k','q2','k2']:
        raw[name]=(n@weights[name].T).reshape(batch,length,heads,hd)
        branches[name]=rotate(rms(raw[name],head_eps),cos,sin)
    def score(q,k):return torch.einsum('bthd,bshd->bhts',q,k)/hd
    s1=score(branches['q'],branches['k']);s2=score(branches['q2'],branches['k2'])
    mask=torch.ones(length,length,device=x.device,dtype=torch.bool).tril()
    pattern=(s1*s2).masked_fill(~mask,0)
    value=(1-mixture)*(n@weights['v'].T).reshape(batch,length,heads,hd)+mixture*first_value
    joined=torch.einsum('bhts,bshd->bthd',pattern,value).reshape(batch,length,width)
    return joined@weights['o'].T,dict(raw=raw,branches=branches,score1=s1,score2=s2,
                                    pattern=pattern,value=value,mask=mask)


def pullback(weights,x0,x1,first_value,mixture,heads,cos,sin,input_eps,head_eps,g):
    """Return q with <q,x1-x0> = <g,Attention(x1)-Attention(x0)>.

    The equality sums over sequence positions/features independently per example.
    No softmax, row normalization, frozen RMS scale or perfect-RoPE assumption.
    """
    _,a=forward(weights,x0,first_value,mixture,heads,cos,sin,input_eps,head_eps)
    _,b=forward(weights,x1,first_value,mixture,heads,cos,sin,input_eps,head_eps)
    batch,length,width=x0.shape;hd=width//heads
    single=g.ndim==3
    if single:g=g[None]
    readers=g.shape[0]
    go=(g@weights['o']).reshape(readers,batch,length,heads,hd)
    gp=torch.einsum('rbthd,bshd->rbhts',go,(a['value']+b['value'])/2)
    gp=gp.masked_fill(~a['mask'],0)
    gv=torch.einsum('bhts,rbthd->rbshd',(a['pattern']+b['pattern'])/2,go)
    gn=(1-mixture)*gv.reshape(readers,batch,length,width)@weights['v']
    for qname,kname,other in [('q','k','score2'),('q2','k2','score1')]:
        gs=gp*(a[other]+b[other])/2
        qm=(a['branches'][qname]+b['branches'][qname])/2
        km=(a['branches'][kname]+b['branches'][kname])/2
        gq=torch.einsum('rbhts,bshd->rbthd',gs,km)/hd
        gk=torch.einsum('rbhts,bthd->rbshd',gs,qm)/hd
        for name,adjoint in [(qname,gq),(kname,gk)]:
            # Merge reader/example axes only for the position-aware rotation.
            unrotated=rotate(adjoint.reshape(readers*batch,length,heads,hd),cos,sin,transpose=True)
            unrotated=unrotated.reshape(readers,batch,length,heads,hd)
            raw_adjoint=rms_pullback(a['raw'][name],b['raw'][name],unrotated,head_eps)
            gn=gn+raw_adjoint.reshape(readers,batch,length,width)@weights[name]
    result=rms_pullback(x0,x1,gn,input_eps)
    return result[0] if single else result
