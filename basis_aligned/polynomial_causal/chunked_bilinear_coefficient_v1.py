"""Exact CP quadratic loss and analytic factor gradients in bounded-size chunks.

Writers must already be in Euclidean output coordinates (e.g. whitened by U).
No full token tensor or coefficient Jacobian is materialized.
"""
import json
from pathlib import Path
import torch


@torch.no_grad()
def value_gradient(native_a,native_b,native_w,a,b,w,total,penalty=0.,chunk=256):
    ga=torch.zeros_like(a);gb=torch.zeros_like(b);gw=torch.zeros_like(w)
    norm=a.new_zeros(());overlap=norm.clone()
    for start in range(0,len(a),chunk):
        stop=min(start+chunk,len(a));sl=slice(start,stop)
        aa=a[sl]@a.T;bb=b[sl]@b.T;ab=a[sl]@b.T;ba=b[sl]@a.T
        output=w[:,sl].T@w
        gram=(aa*bb+ab*ba)/2
        norm+=(output*gram).sum()
        ga[sl]=(output*bb)@a+(output*ba)@b
        gb[sl]=(output*aa)@b+(output*ab)@a
        gw[:,sl]=2*w@gram.T
        la=native_a@a[sl].T;lb=native_a@b[sl].T
        ra=native_b@a[sl].T;rb=native_b@b[sl].T
        out_cross=native_w.T@w[:,sl]
        cross=(la*rb+lb*ra)/2
        overlap+=(out_cross*cross).sum()
        ga[sl]-=(out_cross*rb).T@native_a+(out_cross*lb).T@native_b
        gb[sl]-=(out_cross*la).T@native_b+(out_cross*ra).T@native_a
        gw[:,sl]-=2*native_w@cross
    na=a.square().sum(1);nb=b.square().sum(1);dot=(a*b).sum(1)
    nw=w.square().sum(0);self_gram=(na*nb+dot.square())/2
    energy=(nw*self_gram).sum()
    ga+=penalty*nw[:,None]*(nb[:,None]*a+dot[:,None]*b)
    gb+=penalty*nw[:,None]*(na[:,None]*b+dot[:,None]*a)
    gw+=2*penalty*w*self_gram
    residual=(total+norm-2*overlap)/total
    return residual+penalty*energy/total,(ga/total,gb/total,gw/total),dict(
        residual=float(residual),component_energy=float(energy/total),
        fitted_energy=float(norm/total),cross_energy=float(overlap/total))


def dense(a,b,w):
    q=torch.einsum('ok,ki,kj->oij',w,a,b)
    return (q+q.transpose(-1,-2))/2


def control():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(618)
    na,nb,nw=torch.randn(9,7),torch.randn(9,7),torch.randn(5,9)
    target=dense(na,nb,nw);total=target.square().sum();rows=[]
    for penalty in (0.,.01):
        a,b,w=[x.requires_grad_() for x in (torch.randn(6,7),torch.randn(6,7),torch.randn(5,6))]
        prediction=dense(a,b,w)
        energy=(w.square().sum(0)*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())/2).sum()
        exact=((prediction-target).square().sum()+penalty*energy)/total
        gradients=torch.autograd.grad(exact,(a,b,w))
        for chunk in (1,4,32):
            actual,g,details=value_gradient(na,nb,nw,a,b,w,total,penalty,chunk)
            errors=[float((x-y).norm()/y.norm()) for x,y in zip(g,gradients)]
            rows.append(dict(penalty=penalty,chunk=chunk,loss_error=abs(float(actual-exact)),gradient_errors=errors))
    result=dict(rows=rows,maximum_error=max(max(row['loss_error'],*row['gradient_errors']) for row in rows),
                scope='Exact dense/analytic gradient comparison; no optimizer or native recovery claim.')
    with Path(__file__).with_name('CHUNKED_BILINEAR_COEFFICIENT_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['maximum_error']<1e-10


if __name__=='__main__':control()
