"""Compare linear reader spaces after folding through a quadratic producer.

Uses Frobenius polynomial coefficient geometry, not an activation distribution.
Whiten in supported function spans; invariance is to invertible reader re-basing.
"""
import json
from pathlib import Path
import torch


def product_gram(left, right):
    return ((left@left.T)*(right@right.T)+(left@right.T)*(right@left.T))/2


def support_whitener(gram, tolerance=1e-10):
    values, vectors = torch.linalg.eigh((gram+gram.T)/2)
    scale = float(values.abs().max())
    if scale == 0: return vectors[:, :0]
    if float(values.min()) < -tolerance*scale:
        raise ValueError('Indefinite Gram: coefficient metric invalid')
    keep = values > tolerance*scale
    return vectors[:, keep]/values[keep].sqrt()[None, :]


def overlap(a, b, metric):
    aa, bb, ab = a@metric@a.T, b@metric@b.T, a@metric@b.T
    wa, wb = support_whitener(aa), support_whitener(bb)
    if wa.shape[1] == 0 or wb.shape[1] == 0:
        return dict(ranks=[wa.shape[1], wb.shape[1]], principal_cosines=[],
                    mean_squared_cosine=None)
    singular = torch.linalg.svdvals(wa.T@ab@wb)
    return dict(ranks=[wa.shape[1], wb.shape[1]], principal_cosines=singular.tolist(),
                mean_squared_cosine=float(singular.square().mean()))


def control():
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64); torch.manual_seed(1343)
    # Residual coordinates0/1 contain identical producer functions despite
    # orthogonal raw read vectors; coordinate2 carries an independent monomial.
    l=torch.tensor([[1.,0.,0.],[0.,0.,1.]])
    rr=torch.tensor([[0.,1.,0.],[0.,0.,1.]])
    down=torch.tensor([[1.,0.],[1.,0.],[0.,1.]])
    kg=product_gram(l,rr); metric=down@kg@down.T
    a=torch.tensor([[1.,0.,0.]]); b=torch.tensor([[0.,1.,0.]])
    other=torch.tensor([[0.,0.,1.]])
    raw=overlap(a,b,torch.eye(3)); shared=overlap(a,b,metric); null=overlap(a,other,metric)
    reader=torch.randn(2,3); partner=torch.randn(2,3)
    base=overlap(reader,partner,metric)
    change=torch.tensor([[3.,.4],[.2,.7]])
    rebase=overlap(change@reader,torch.linalg.inv(change)@partner,metric)
    gauge=max(abs(x-y) for x,y in zip(base['principal_cosines'],rebase['principal_cosines']))
    zero=overlap(torch.tensor([[1.,-1.,0.]]),other,metric)
    # Independent dense polynomial coefficients audit the implicit Gram.
    forms=(l[:,:,None]*rr[:,None,:]+rr[:,:,None]*l[:,None,:])/2
    out=torch.einsum('oi,ijk->ojk',down,forms)
    direct=out.flatten(1)@out.flatten(1).T
    gram_error=float((metric-direct).norm()/direct.norm())
    x=torch.randn(23,3)
    native=((x@l.T)*(x@rr.T))@down.T
    folded=((x@l.T)*(x@rr.T))@(reader@down).T
    replay=float((native@reader.T-folded).norm()/folded.norm())
    result=dict(instrument_passed=raw['mean_squared_cosine']==0 and
                abs(shared['mean_squared_cosine']-1)<1e-10 and null['mean_squared_cosine']==0
                and zero['ranks'][0]==0 and max(gauge,gram_error,replay)<1e-10,
                raw_reader_overlap=raw,shared_producer_function_overlap=shared,
                independent_function_overlap=null,zero_function_handled=zero['ranks'][0]==0,
                invertible_reader_gauge_error=gauge,dense_gram_error=gram_error,
                quadratic_pullback_replay_error=replay,
                scope='Planted coefficient-function sharing only. Native MLP16 comparison not run; final normalization, residual and base branches are outside this local numerator control.')
    with Path(__file__).with_name('PRODUCER_FUNCTION_OVERLAP_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
