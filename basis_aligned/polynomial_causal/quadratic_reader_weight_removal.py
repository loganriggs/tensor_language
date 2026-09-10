"""Bias-preserving static Down edit for fixed homogeneous quadratic readers."""
from contextlib import contextmanager
import torch


def remove_from_down(down,readers,writers,selected):
    c=readers[selected].to(down);d=writers[:,selected].to(down)
    return down-d@(c@down)


@contextmanager
def at_down(module,edited):
    original=module.Down.weight.detach().clone();bias=module.Down_bias.detach().clone()
    try:
        with torch.no_grad():module.Down.weight.copy_(edited.to(module.Down.weight))
        yield
    finally:
        with torch.no_grad():module.Down.weight.copy_(original)
        assert torch.equal(module.Down.weight,original) and torch.equal(module.Down_bias,bias)


def controls():
    from types import SimpleNamespace
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(60918);dtype=torch.float64
        c=torch.randn(4,12,dtype=dtype);d=c.T@torch.linalg.inv(c@c.T)
        w=torch.randn(12,20,dtype=dtype);phi=torch.randn(17,20,dtype=dtype);bias=torch.randn(12,dtype=dtype)
    a=[0,1];b=[2,3];all_indices=a+b
    wa=remove_from_down(w,c,d,a);wb=remove_from_down(w,c,d,b);wab=remove_from_down(w,c,d,all_indices)
    original=phi@w.T+bias
    subtract=original-((phi@w.T)@c[a].T)@d[:,a].T
    direct=phi@wa.T+bias
    error=float((direct-subtract).abs().max())
    module=SimpleNamespace(Down=torch.nn.Linear(20,12,bias=False,dtype=dtype),Down_bias=torch.nn.Parameter(bias.clone()))
    with torch.no_grad():module.Down.weight.copy_(w)
    with at_down(module,wa):live=torch.equal(module.Down.weight,wa) and torch.equal(module.Down_bias,bias)
    restored=torch.equal(module.Down.weight,w)
    try:
        with at_down(module,wb):raise RuntimeError('control')
    except RuntimeError:pass
    # Constant reassignment leaves all interchanges unchanged but alters removal.
    g=(phi@w.T)@c[a].T;k=torch.ones(2,dtype=dtype)
    interchange_error=float(((g[1:]-g[:-1])-((g[1:]+k)-(g[:-1]+k))).abs().max())
    removal_shift=float(((g+k)@d[:,a].T-g@d[:,a].T).abs().max())
    checks={'static_weight_equals_output_program':error<1e-10,
            'opposite_reader_preserved':torch.allclose(c[b]@wa,c[b]@w,atol=1e-10,rtol=1e-10),
            'removed_quadratic_reader_zero':float((c[a]@wa).abs().max())<1e-10,
            'joint_edits_commute':torch.allclose(remove_from_down(wa,c,d,b),wab,atol=1e-10,rtol=1e-10) and torch.allclose(remove_from_down(wb,c,d,a),wab,atol=1e-10,rtol=1e-10),
            'bias_unchanged_and_weight_edit_live':live,'weights_restored_on_normal_and_exception':restored and torch.equal(module.Down.weight,w),
            'interchange_does_not_fix_removal_origin':interchange_error<1e-10 and removal_shift>.1}
    checks={k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks,'max_weight_output_identity_error':error,
            'scope':'Fixed homogeneous quadratic component, original bias retained; semantic absence not assumed.'}
