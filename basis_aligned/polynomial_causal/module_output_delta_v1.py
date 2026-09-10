"""Scoped tensor-output subtraction with a bitwise incoming-output check."""
from contextlib import contextmanager
import torch

@contextmanager
def subtract(module,expected,delta,audit):
    def hook(_module,_args,output):
        assert torch.equal(output,expected)
        changed=(output.double()-delta.double()).to(output)
        audit.append((output.double()-changed.double()).detach().clone())
        return changed
    handle=module.register_forward_hook(hook)
    try:yield
    finally:handle.remove()

def controls():
    m=torch.nn.Identity();x=torch.arange(24,dtype=torch.float32).view(2,3,4);audit=[]
    with subtract(m,x,torch.zeros_like(x),audit):identity=m(x)
    with subtract(m,x,torch.ones_like(x),audit):changed=m(x)
    assert torch.equal(identity,x) and torch.equal(changed,x-1)
    assert torch.equal(audit[1],torch.ones_like(x)) and not m._forward_hooks
    rejected=False
    try:
        with subtract(m,x+1,torch.zeros_like(x),[]):m(x)
    except AssertionError:rejected=True
    assert rejected and not m._forward_hooks
    return {'passed':True,'identity_bitwise':True,'all_positions_subtracted':True,
            'wrong_incoming_rejected':True,'cleanup_after_exception':True}
