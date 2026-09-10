"""Small output-space interchange primitive for fixed joint reader coordinates."""
from contextlib import contextmanager
import torch


def apply(output,donor,readers,writers,selected,stops):
    changed=output.clone()
    for i,stop in enumerate(stops):
        assert 0<int(stop)<=output.shape[1] and int(stop)<=donor.shape[1]
        x=output[i,:int(stop)].double();target=donor[i,:int(stop)].to(x)
        c=readers[selected].to(x);d=writers[:,selected].to(x)
        changed[i,:int(stop)]=(x+((target-x)@c.T)@d.T).to(output.dtype)
    return changed


@contextmanager
def at_module(module,donor,readers,writers,selected,stops,audit=None):
    def hook(_module,_args,output):
        changed=apply(output,donor,readers,writers,selected,stops)
        if audit is not None:audit.append((output.detach().clone(),changed.detach().clone()))
        return changed
    handle=module.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def controls():
    dtype=torch.float64
    c=torch.tensor([[1.,0.,0.],[.8,.6,0.]],dtype=dtype)
    d=c.T@torch.linalg.inv(c@c.T)
    x=torch.tensor([[[2.,3.,4.],[4.,2.,1.]],[[3.,1.,4.],[2.,4.,1.]]],dtype=dtype)
    target=x.flip(-1);stops=[2,1]
    a=apply(x,target,c,d,[0],stops);b=apply(x,target,c,d,[1],stops)
    ab=apply(a,target,c,d,[1],stops);ba=apply(b,target,c,d,[0],stops)
    joint=apply(x,target,c,d,[0,1],stops)
    positions=torch.tensor([[True,True],[True,False]])
    module=torch.nn.Identity();audit=[]
    with at_module(module,target,c,d,[0],stops,audit):hooked=module(x)
    restored=torch.equal(module(x),x) and len(module._forward_hooks)==0
    try:
        with at_module(module,target,c,d,[0],stops):raise RuntimeError('control')
    except RuntimeError:pass
    checks={'target_reader_interchanged':bool(torch.allclose((a@c[0])[positions],(target@c[0])[positions],atol=1e-12,rtol=1e-12)),
            'other_reader_preserved':bool(torch.allclose(a@c[1],x@c[1],atol=1e-12,rtol=1e-12)),
            'joint_and_both_orders':bool(torch.allclose(ab,joint,atol=1e-12,rtol=1e-12) and torch.allclose(ba,joint,atol=1e-12,rtol=1e-12)),
            'unselected_positions_unchanged':bool(torch.equal(joint[~positions],x[~positions])),
            'native_hook_replay':bool(torch.equal(hooked,a)) and len(audit)==1,
            'hook_restored':restored and len(module._forward_hooks)==0,
            'live_edit':float((a-x).abs().max())>1}
    return {'passed':all(checks.values()),'checks':checks}
