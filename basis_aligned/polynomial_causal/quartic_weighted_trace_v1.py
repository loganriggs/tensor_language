"""Matrix-free quartic flattening action on symmetric matrices."""
import torch

def producer_core(d0,l1,r1,output_coefficients,scale=1.):
    matrix=l1.T@(output_coefficients[:,None]*r1)
    matrix=(matrix+matrix.T)/2
    return scale**2*(d0.T@matrix@d0)

def native_action(l,r,h,q):
    lq=l@q;rq=r@q
    ll=lq@l.T;lr=lq@r.T;rr=rq@r.T
    trace=(lq*r).sum(-1)
    weights=h@trace
    first=l.T@(weights[:,None]*r);first=(first+first.T)/2
    product=(l.T@((h*lr.T)@r)+l.T@((h*rr)@l)+r.T@((h*ll)@r)+r.T@((h*lr)@l))/4
    return (first+2*product)/3

def fitted_action(b,n,mixing,q):
    result=torch.zeros_like(q)
    for bank,weights,coefficient in zip(b,n,mixing):
        inner=bank.T@q@bank
        trace=(weights*inner.diagonal()).sum()
        core=torch.diag(trace*weights)+2*weights[:,None]*inner*weights[None,:]
        result+=coefficient*(bank@core@bank.T)/3
    return result
