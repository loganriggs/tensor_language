"""Native vector-valued bilinear cross term between two frozen input readers."""
import torch


def writer(mlp,e1,e2):
    L=mlp.Left.weight.double();R=mlp.Right.weight.double();D=mlp.Down.weight.double()
    return D@((L@e1)*(R@e2)+(L@e2)*(R@e1))


def coefficient(u,e1,e2):
    return (u.double()@e1)*(u.double()@e2)


def controls():
    from types import SimpleNamespace
    gen=torch.Generator().manual_seed(9111236)
    rand=lambda *s:torch.randn(*s,generator=gen,dtype=torch.float64)
    m=SimpleNamespace(Left=SimpleNamespace(weight=rand(9,6)),Right=SimpleNamespace(weight=rand(9,6)),Down=SimpleNamespace(weight=rand(6,9)),Down_bias=rand(6))
    E=torch.linalg.qr(rand(6,2)).Q;e1,e2=E.T;v=writer(m,e1,e2)
    f=lambda u:((u@m.Left.weight.T)*(u@m.Right.weight.T))@m.Down.weight.T+m.Down_bias
    polarization=(f(e1+e2)-f(e1-e2))/2
    u=rand(12,6);a=u@e1;b=u@e2;c=u-a[:,None]*e1-b[:,None]*e2
    # Mixed finite difference keeps the complement c fixed, cancels bias and
    # all terms other than the e1/e2 interaction.
    mixed=f(u)-f(c+a[:,None]*e1)-f(c+b[:,None]*e2)+f(c)
    expected=(a*b)[:,None]*v
    errors=dict(writer=float((polarization-v).abs().max()),mixed=float((mixed-expected).abs().max()),sign_gauge=float((coefficient(u,-e1,e2)[:,None]*writer(m,-e1,e2)-expected).abs().max()))
    assert max(errors.values())<1e-10 and float(expected.norm())>.1
    return dict(passed=True,errors=errors,scope='Complete vector writer, no output-axis fit; input basis sign changes cancel in the full term')
