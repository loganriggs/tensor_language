"""Separate raw key projections from source RMS gain, with native query/value."""
import torch
import source_port_interchange_reference as P


def gains(layer, x):
    eps = torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    return torch.rsqrt(x.square().mean(-1, keepdim=True)+eps)


def ports(program, context, write, direction, gain):
    x = context['x']; changed = x-write; layer = program.background.layers[-1]
    ratio = gains(layer, changed)/gains(layer, x)
    assert bool(torch.isfinite(ratio).all() and (ratio > 0).all())
    source = program.features(changed, context['positions']) if direction else context['features']
    factor = (1. if gain else ratio.reciprocal()) if direction else (ratio if gain else 1.)
    if isinstance(factor, torch.Tensor): factor = factor.unsqueeze(-1)
    mixed = {**context['features']}
    for name in ('k1', 'k2'): mixed[name] = source[name]*factor
    return mixed


def native(model, base, write, direction, gain):
    layer = model.layers[-1]; raw = base-write if direction else base
    normalized = raw*gains(layer, base-write if gain else base); handles = []
    try:
        for name in ('k1', 'k2'):
            module = getattr(layer, name); replacement = module(normalized)
            def replace(_module, _args, _output, replacement=replacement): return replacement
            handles.append(module.register_forward_hook(replace))
        return model.head(layer(base))
    finally:
        for h in handles: h.remove()


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42908); model = DeepModel(8,16,4,['attn']*4,32,norm='rms').double().eval()
        tok = torch.randint(8,(2,8)); write = torch.zeros(2,8,16,dtype=torch.float64)
        write[:,2] = torch.randn(2,16,dtype=torch.float64)*.2
    program = SourceEditProgram(model); checks = {}
    with torch.inference_mode():
        context = program.prepare(tok); before = model(tok)
        for d in (False, True):
            for g in (False, True):
                out = P.execute(program,context,ports(program,context,write,d,g))
                checks[f'native_{d}_{g}'] = bool(torch.allclose(out,native(model,context['x'],write,d,g),atol=1e-9,rtol=1e-10))
                zero = P.execute(program,context,ports(program,context,write*0,d,g))
                checks[f'zero_write_{d}_{g}'] = bool(torch.allclose(zero,before,atol=1e-9,rtol=1e-10))
        equal = write*0; equal[:,2,:8] = 2*context['x'][:,2,:8]
        gain_only = ports(program,context,equal,False,True)
        full = ports(program,context,equal,True,True); direction = ports(program,context,equal,True,False)
        checks['equal_norm_gain_identity'] = all(torch.allclose(gain_only[k],context['features'][k],atol=1e-12,rtol=1e-12) for k in ('k1','k2'))
        checks['equal_norm_direction_live'] = float((direction['k1']-context['features']['k1']).abs().max())>1e-3
        checks['equal_norm_direction_full'] = torch.allclose(direction['k1'],full['k1'],atol=1e-12,rtol=1e-12)
        radial = write*0; radial[:,2] = -.5*context['x'][:,2]
        rfull = ports(program,context,radial,True,True); rdir = ports(program,context,radial,True,False)
        checks['radial_full_cancellation'] = torch.allclose(rfull['k1'],context['features']['k1'],atol=1e-12,rtol=1e-12)
        checks['radial_direction_live'] = float((rdir['k1']-context['features']['k1']).abs().max())>1e-3
        checks['hooks_restored'] = torch.equal(before,model(tok))
    checks = {k:bool(v) for k,v in checks.items()}
    return {'passed':all(checks.values()),'checks':checks}
