"""Scoped all-position attention-write subtraction preserving other tuple fields."""
from contextlib import contextmanager
import torch


@contextmanager
def subtract(module, expected, delta, audit):
    def hook(_module, _args, output):
        assert torch.equal(output[0], expected)
        changed = (output[0].double() - delta.double()).to(output[0])
        actual = output[0].double() - changed.double()
        audit.append(actual.detach().clone())
        return (changed, *output[1:])
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def controls():
    class Attention(torch.nn.Module):
        def forward(self, x, first):
            return x * 2, first
    m = Attention()
    x = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    first = x + 7
    native = m(x, first)
    audit = []
    with subtract(m, native[0], torch.ones_like(x), audit):
        changed = m(x, first)
    assert torch.equal(changed[0], native[0] - 1)
    assert changed[1] is first and torch.equal(audit[0], torch.ones_like(x))
    assert not m._forward_hooks
    return {'passed': True, 'all_positions_changed': True, 'first_value_identity': True,
            'hook_cleanup': True}
