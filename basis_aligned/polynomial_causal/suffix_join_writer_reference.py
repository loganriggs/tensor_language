"""Live pair-to-pair upstream edge cuts and explicitly labeled native source ports."""
from contextlib import contextmanager
import types
import torch


def edge_masks(tokens):
    assert tokens.device.type == 'cpu' and tokens.shape[1] == 51
    target = torch.zeros(len(tokens), 51, 51, dtype=torch.bool)
    control = torch.zeros_like(target)
    ports = torch.zeros(len(tokens), 51, dtype=torch.bool)
    orientation = []
    for row, tok in enumerate(tokens):
        keys, values = tok[:48:2].tolist(), tok[1:48:2].tolist()
        fmap = dict(zip(keys, values)); positions = {e: 2*i for i, e in enumerate(keys)}
        e = int(tok[49]); second = fmap[e]; third = fmap[second]
        p2, p3 = positions[second], positions[third]
        assert p2 != p3
        early, late = min(p2, p3), max(p2, p3)
        assert early+3 < late and (early+2)//2 not in (3, 11, 19)
        target[row, late:late+2, early:early+2] = True
        control[row, late:late+2, early+2:early+4] = True
        ports[row, late:late+2] = True
        orientation.append('B2_later' if p2 > p3 else 'B3_later')
    return target, control, ports, orientation


@contextmanager
def modifications(model, mask, layers, ports=None, source=None):
    changed = []; handles = []
    try:
        for index in layers:
            layer = model.layers[index]; original = layer.pattern
            def pattern(_self, x, original=original):
                return original(x).masked_fill(mask[:, None], 0)
            layer.pattern = types.MethodType(pattern, layer); changed.append(layer)
        if source is not None:
            for name in ('k1', 'k2', 'v'):
                def restore(_module, _args, output, name=name):
                    return torch.where(ports[..., None], source[name], output)
                handles.append(getattr(model.layers[-1], name).register_forward_hook(restore))
        yield
    finally:
        for h in handles:
            h.remove()
        for layer in changed:
            del layer.pattern


def native_ports(model, tokens):
    captured = {}; handles = []
    for name in ('k1', 'k2', 'v'):
        def capture(_module, _args, output, name=name):
            captured[name] = output.detach().clone()
        handles.append(getattr(model.layers[-1], name).register_forward_hook(capture))
    try:
        logits = model(tokens)
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def explicit_masked(model, tokens, mask, layers):
    x = model.embed(tokens)
    for i, layer in enumerate(model.layers):
        pattern = layer.pattern(x)
        if i in layers:
            pattern = pattern.masked_fill(mask[:, None], 0)
        v = layer.v(layer.norm(x)).reshape(*tokens.shape, layer.n_head, layer.d_head)
        z = torch.einsum('bhts,bshd->bthd', pattern, v).flatten(-2)
        x = x+layer.o(z) if layer.residual == 'add' else torch.lerp(x, layer.o(z), layer.scale)
    return model.head(x)


def controls():
    from deep_model import DeepModel
    import causal_suffix_join_reference as data
    tokens = data.populations(torch)['iid'][0][:4]
    mask, control, ports, orientation = edge_masks(tokens)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
    checks = {'four_edges_each': bool((mask.sum((1, 2)) == 4).all()),
              'control_disjoint': not bool((mask & control).any()),
              'causal_edges': not bool((mask | control).triu(1).any())}
    with torch.inference_mode():
        base, values = native_ports(model, tokens)
        for layers in ((0,), (1,), (2,), (0, 1, 2)):
            expected = explicit_masked(model, tokens, mask, layers)
            with modifications(model, mask, layers):
                observed = model(tokens)
            checks['explicit_'+str(layers)] = bool(torch.allclose(expected, observed, atol=1e-9, rtol=1e-9))
        with modifications(model, mask, (), ports, values):
            restored = model(tokens)
        checks['identity_rescue'] = bool(torch.allclose(base, restored, atol=1e-9, rtol=1e-9))
        checks['hooks_restored'] = bool(torch.equal(base, model(tokens)))
        checks['live_joint_cut'] = bool((expected-base).abs().max() > 1e-12)
    return {'passed': all(checks.values()), 'checks': checks}
