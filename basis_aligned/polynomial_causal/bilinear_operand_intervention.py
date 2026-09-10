"""Scoped native operand edits and an explicitly symmetric mixed-output extension."""
from contextlib import contextmanager
import torch


def replace_valid(output, replacement, stops):
    changed = output.clone()
    assert output.shape == replacement.shape
    for i, stop in enumerate(stops):
        assert 0 < stop <= output.shape[1]
        changed[i, :stop] = replacement[i, :stop].to(output)
    return changed


@contextmanager
def scoped(module, left=None, right=None, output=None, stops=()):
    handles = []
    try:
        for target, replacement in ((module.Left, left), (module.Right, right), (module, output)):
            if replacement is not None:
                def hook(_m, _args, value, replacement=replacement):
                    return replace_valid(value, replacement, stops)
                handles.append(target.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def controls():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.Left = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
            self.Right = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.Left.weight.copy_(torch.eye(2, dtype=torch.float64))
                self.Right.weight.copy_(torch.tensor([[0.,1.],[1.,0.]], dtype=torch.float64))
        def forward(self,x):
            return (self.Left(x)*self.Right(x)) @ torch.tensor([[1.],[-1.]], dtype=x.dtype)
    m=Toy();b=torch.tensor([[[1.,0.],[2.,1.]]],dtype=torch.float64);d=b.flip(-1);stops=[1]
    with torch.no_grad():
        base=m(b);donor=m(d);ld=m.Left(d);rd=m.Right(d)
        with scoped(m,left=ld,stops=stops):l=m(b)
        with scoped(m,right=rd,stops=stops):r=m(b)
        with scoped(m,left=ld,right=rd,stops=stops):both=m(b)
        with scoped(m,output=donor,stops=stops):direct=m(b)
        symmetric=(l+r)/2;skew=(l-r)/2
        try:
            with scoped(m,left=ld,stops=stops):raise RuntimeError('fixture')
        except RuntimeError:pass
        checks={
            'diagonal_skew_vanishes':bool(torch.equal(base,torch.zeros_like(base))),
            'independent_operands_reveal_skew':float((l-r).abs().max())==2.,
            'symmetric_extension_loses_operand_identity':bool(torch.equal(symmetric,base)),
            'symmetric_plus_skew_recovers_left':bool(torch.equal(symmetric+skew,l)),
            'symmetric_minus_skew_recovers_right':bool(torch.equal(symmetric-skew,r)),
            'both_equals_direct_donor':bool(torch.equal(both,direct)),
            'unselected_positions_unchanged':bool(torch.equal(l[:,1:],base[:,1:])),
            'normal_and_exception_hooks_restored':all(not q._forward_hooks for q in (m,m.Left,m.Right)) and bool(torch.equal(m(b),base)),
        }
    return {'passed':all(checks.values()),'checks':checks}
