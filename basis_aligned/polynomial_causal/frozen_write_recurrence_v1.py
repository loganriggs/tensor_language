"""Replay native ordered residual arithmetic with supplied frozen writes."""
import torch


def replay(entry,embedding,attention,mlp,lambdas):
    assert len(attention)==len(mlp)==len(lambdas)
    state=entry
    for a,m,coeff in zip(attention,mlp,lambdas):
        live=coeff[0]*state+coeff[1]*embedding
        state=live+a
        state=state+m
    return state


def controls():
    rng=torch.Generator().manual_seed(910833)
    entry=torch.randn(16,32,generator=rng)*1e5;e=torch.randn(16,32,generator=rng)
    attention=[torch.randn(16,32,generator=rng)*1e3 for _ in range(10)]
    mlps=[torch.randn(16,32,generator=rng)*1e3 for _ in range(10)]
    scales=[torch.tensor([.7+.08*i,.13]) for i in range(10)]
    delta=torch.randn(16,32,generator=rng)*.1
    edited=[(mlps[0].double()-delta.double()).to(mlps[0])]+mlps[1:]
    actual=entry.clone()
    for i in range(10):
        actual=torch.add(torch.mul(scales[i][0],actual),torch.mul(scales[i][1],e))
        actual=torch.add(torch.add(actual,attention[i]),edited[i])
    predicted=replay(entry,e,attention,edited,scales)
    assert torch.equal(predicted,actual)
    native=replay(entry,e,attention,mlps,scales)
    carry=1.
    for c in scales[1:]:carry*=float(c[0])
    closed=(native.double()-carry*delta.double()).to(native)
    discrepancy=float((closed-actual).abs().max());assert discrepancy>1e-3
    return {'passed':True,'ordered_recurrence_bitwise':True,'closed_form_absolute_discrepancy':discrepancy,
            'scope':'Native operation order is replayed; real-arithmetic transport is not a bitwise floating-point identity.'}
