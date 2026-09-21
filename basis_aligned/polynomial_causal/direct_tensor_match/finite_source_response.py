"""Exact finite response of a quadratic, using its midpoint polarization identity."""
import torch
from centered_product_response import features

def normalized_secant(h,delta,mean,eps):
    def norm(t):return t/(t.square().mean(-1,keepdim=True)+eps).sqrt()
    x0=norm(h);x1=norm(h+delta)
    return (x0+x1)/2-mean,x1-x0

def controls():
    rows=[]
    for dimension in range(3,8):
        torch.manual_seed(903+dimension)
        rand=lambda *s:torch.randn(*s,dtype=torch.float64)
        h,delta,mu=rand(23,dimension),rand(23,dimension),rand(dimension)
        a,b,c=rand(5,dimension),rand(5,dimension),rand(4,5)
        a.requires_grad_();b.requires_grad_();eps=torch.finfo(torch.float32).eps
        midpoint,step=normalized_secant(h,delta,mu,eps)
        norm=lambda t:t/(t.square().mean(-1,keepdim=True)+eps).sqrt()
        x0,x1=norm(h)-mu,norm(h+delta)-mu
        direct=(((x1@a.T)*(x1@b.T))-((x0@a.T)*(x0@b.T)))@c.T
        implicit=features(midpoint,step,a,b)@c.T
        error=float(((direct-implicit).norm()/direct.norm()).detach())
        gd=torch.autograd.grad(direct.square().sum(),(a,b),retain_graph=True)
        gi=torch.autograd.grad(implicit.square().sum(),(a,b))
        grad=max(float((x-y).norm()/x.norm()) for x,y in zip(gd,gi))
        rows.append(dict(dimension=dimension,finite_difference_error=error,gradient_error=grad))
    assert max(max(r['finite_difference_error'],r['gradient_error']) for r in rows)<1e-12
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    rows=controls();Path(__file__).with_name('FINITE_SOURCE_RESPONSE_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
