"""Exact payload producer terms with live normalization and native join routing."""
import torch
from join_contribution_context_reference import contributions, intervene, populations

KINDS = ('E', 'Y0', 'Y1')


def producer_writes(model, tokens, masks, heads):
    captured = {}; handles = []
    for i in range(2):
        layer = model.layers[i]
        assert layer.residual == 'lerp' and layer.scale == .5
        def capture(_module, _args, output, i=i): captured[i] = output
        handles.append(layer.o.register_forward_hook(capture))
    try:
        e = model.embed(tokens); x = e
        for layer in model.layers[:2]: x = layer(x)
    finally:
        for handle in handles: handle.remove()
    parts = {'E': e*.25, 'Y0': captured[0]*.25, 'Y1': captured[1]*.5}
    layer = model.layers[2]
    assert layer.residual == 'lerp' and layer.scale == .5
    assert isinstance(layer.norm, torch.nn.RMSNorm) and not layer.norm.elementwise_affine
    eps = torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    gain = torch.rsqrt(x.square().mean(-1, keepdim=True)+eps)
    p = layer.pattern(x); out = {j: {} for j in masks}
    for kind, part in parts.items():
        v = layer.v(part*gain).reshape(len(tokens), tokens.shape[1], layer.n_head, layer.d_head)
        for j, mask in masks.items():
            h = heads[j]; z = torch.einsum('bts,bsd->btd', p[:, h]*mask, v[:, :, h])
            out[j][kind] = (z @ layer.o.weight[:, h*layer.d_head:(h+1)*layer.d_head].T)*layer.scale
    return out, float((sum(parts.values())-x).abs().max())


def combine(parts, selected=KINDS):
    return {j: sum(v[k] for k in selected) if selected else torch.zeros_like(v['E']) for j, v in parts.items()}


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(18908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
    w = populations()['iid'][0]; tok = w['recipient']; masks, heads = w['masks'], w['heads']
    checks = {}
    with torch.inference_mode():
        parts, error = producer_writes(model, tok, masks, heads)
        original = contributions(model, tok, masks, heads); summed = combine(parts)
        checks['residual_producer_identity'] = error <= 1e-9
        checks['native_write_sum'] = all(torch.allclose(v, original[j], atol=1e-9, rtol=1e-9) for j, v in summed.items())
        checks['all_terms_live'] = all(float(v.abs().max()) > 1e-12 for d in parts.values() for v in d.values())
        native = model(tok)
        with intervene(model, masks, heads, (0, 1), summed): restored = model(tok)
        checks['joint_all_term_replay'] = bool(torch.allclose(native, restored, atol=1e-9, rtol=1e-9))
        with intervene(model, masks, heads, (0, 1), combine(parts, ())): zero = model(tok)
        with intervene(model, masks, heads, (0, 1)): cut = model(tok)
        checks['empty_subset_removal'] = bool(torch.equal(zero, cut))
    return {'passed': all(checks.values()), 'checks': checks}
