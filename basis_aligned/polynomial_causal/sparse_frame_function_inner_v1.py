"""Exact function inner product between orthogonal quadratic edge programs."""
import torch


def coefficients(q,l,r,w,edges):
    a,b=l@q,r@q;i,j=edges
    norm=torch.where(i==j,2.,2.**.5)
    return w@((a[:,i]*b[:,j]+a[:,j]*b[:,i])/norm)


@torch.no_grad()
def inner(q,e,c,qq,ee,cc,chunk=256):
    overlap=q.T@qq;i,j=e;ii,jj=ee
    n=torch.where(i==j,2.,2.**.5);nn=torch.where(ii==jj,2.,2.**.5)
    value=q.new_zeros(())
    for start in range(0,len(i),chunk):
        a=i[start:start+chunk,None];b=j[start:start+chunk,None]
        features=2*(overlap[a,ii]*overlap[b,jj]+overlap[a,jj]*overlap[b,ii])/(n[start:start+chunk,None]*nn)
        value+=((c[:,start:start+chunk].T@cc)*features).sum()
    return value


def control():
    torch.set_default_dtype(torch.float64);g=torch.Generator().manual_seed(120461)
    q,qq=[torch.linalg.qr(torch.randn(6,6,generator=g)).Q for _ in range(2)]
    e=torch.tensor([[0,1,2,3],[0,2,4,5]]);ee=torch.tensor([[0,1,2],[3,1,5]])
    c=torch.randn(5,4,generator=g);cc=torch.randn(5,3,generator=g)
    def dense(q,e,c):
        i,j=e;n=torch.where(i==j,2.,2.**.5)
        return torch.einsum('oe,ae,be,e->oab',c,q[:,i],q[:,j],1/n)+torch.einsum('oe,ae,be,e->oab',c,q[:,j],q[:,i],1/n)
    s,t=dense(q,e,c),dense(qq,ee,cc)
    error=float(abs(inner(q,e,c,qq,ee,cc)-(s*t).sum())/(s.norm()*t.norm()))
    return dict(error=error,passed=error<1e-12)


if __name__=='__main__':
    import json
    print(json.dumps(control()))
