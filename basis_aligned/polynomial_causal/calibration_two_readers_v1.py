"""Weight folding and exact two-consumer readout, no rank approximation."""
import torch

EPS32 = torch.finfo(torch.float32).eps


def fold(mlp, w):
    w=w.double();c=mlp.Down.weight.double().T@w/(w@w)
    a=(mlp.Left.weight.double().T*c)@mlp.Right.weight.double()
    return (a+a.T)/2, (mlp.Down_bias.double()@w)/(w@w)


def scalar(u, Q, beta):
    u=u.double()
    return ((u@Q)*u).sum(-1)+beta


def read(h,q,w,U,numerator,denominator):
    h=h.double();q=q.double();w=w.double();U=U.double()
    norm_state=h-q[...,None]*w if denominator else h
    rho=(norm_state.square().mean(-1,keepdim=True)+EPS32).sqrt()
    scores=h@U.T
    if numerator:scores=scores-q[...,None]*(U@w)
    return 30*torch.tanh(scores/(30*rho))


def controls():
    from types import SimpleNamespace
    g=torch.Generator().manual_seed(9111130)
    rand=lambda *shape:torch.randn(*shape,generator=g,dtype=torch.float64)
    m=SimpleNamespace(Left=SimpleNamespace(weight=rand(7,4)),Right=SimpleNamespace(weight=rand(7,4)),
                      Down=SimpleNamespace(weight=rand(4,7)),Down_bias=rand(4))
    u=rand(9,4);w=rand(4);U=rand(11,4);out=(u@m.Left.weight.T)*(u@m.Right.weight.T)
    out=out@m.Down.weight.T+m.Down_bias
    Q,beta=fold(m,w);q=scalar(u,Q,beta);direct=out@w/(w@w)
    h=rand(9,4)+out;changed=h-q[:,None]*w
    expected=30*torch.tanh((changed@U.T)/(30*(changed.square().mean(-1,keepdim=True)+EPS32).sqrt()))
    arms={(n,d):read(h,q,w,U,n,d) for n in (0,1) for d in (0,1)}
    tiny=h*1e-6
    correct=read(tiny,q*1e-6,w,U,True,True)
    omitted=30*torch.tanh(((tiny-q[:,None]*1e-6*w)@U.T)/(30*(tiny-q[:,None]*1e-6*w).square().mean(-1,keepdim=True).sqrt()))
    errors=dict(producer=float((q-direct).abs().max()),joint=float((arms[1,1]-expected).abs().max()))
    assert max(errors.values())<1e-10
    assert float((arms[1,0]-arms[0,1]).norm())>.01
    assert float((correct-omitted).norm())>.01
    return dict(passed=True,errors=errors,consumer_distinction=True,epsilon_falsifier=True)
