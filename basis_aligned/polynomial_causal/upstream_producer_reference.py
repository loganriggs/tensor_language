"""Exact live normalized producer contributions and final reader-edge cuts."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

READERS = ('q1', 'k1', 'q2', 'k2', 'v')
PRODUCERS = ('E', 'A', 'M')


def producers(model, tokens):
    assert model.spec == ['attn', 'mlp', 'attn']
    embed = model.embed(tokens)
    first = model.layers[0]
    after_attn = first(embed)
    residual = 1. if first.residual == 'add' else 1-first.scale
    e = residual*embed
    a = after_attn-e
    x = model.layers[1](after_attn)
    m = x-after_attn
    norm = model.layers[-1].norm
    if isinstance(norm, torch.nn.RMSNorm):
        assert not norm.elementwise_affine
        eps = torch.finfo(x.dtype).eps if norm.eps is None else norm.eps
        gain = torch.rsqrt(x.square().mean(-1, keepdim=True)+eps)
    else:
        assert isinstance(norm, torch.nn.Identity)
        gain = torch.ones_like(x[..., :1])
    return x, {'E': gain*e, 'A': gain*a, 'M': gain*m}


@contextmanager
def reader_cuts(layer, components, cuts):
    handles = []
    for reader in READERS:
        selected = [p for p, r in cuts if r == reader]
        if not selected:
            continue
        value = sum(components[p] for p in selected)
        def remove(module, args, output, value=value):
            return output-F.linear(value, module.weight)
        handles.append(getattr(layer, reader).register_forward_hook(remove))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def forward(model, tokens, cuts=()):
    x, components = producers(model, tokens)
    with reader_cuts(model.layers[-1], components, cuts):
        return model.head(model.layers[-1](x))


def controls():
    from deep_model import DeepModel
    from hop_data import sample_docs
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7908)
        model = DeepModel(29, 16, 4, ['attn', 'mlp', 'attn'], 240, norm='rms').double().eval()
        tokens = sample_docs(2, torch.Generator().manual_seed(7908))[0][:, :-1]
    checks = {}
    with torch.inference_mode():
        x, parts = producers(model, tokens)
        layer = model.layers[-1]
        checks['normalized_reassembly'] = bool(torch.allclose(sum(parts.values()), layer.norm(x), atol=1e-9, rtol=1e-9))
        for reader in READERS:
            projection = getattr(layer, reader)
            checks['reassembled_'+reader] = bool(torch.allclose(sum(projection(v) for v in parts.values()), projection(layer.norm(x)), atol=1e-9, rtol=1e-9))
        checks['native_forward'] = bool(torch.allclose(model(tokens), forward(model, tokens), atol=1e-9, rtol=1e-9))
        for p in PRODUCERS:
            for readers in (('q1',), READERS):
                cuts = tuple((p, r) for r in readers)
                observed = forward(model, tokens, cuts)
                # Independent reference recomputes each reader on normalized x minus component.
                handles = []
                for reader in readers:
                    def explicit(module, args, _output, p=p):
                        return F.linear(args[0]-parts[p], module.weight)
                    handles.append(getattr(layer, reader).register_forward_hook(explicit))
                expected = model(tokens)
                for h in handles:
                    h.remove()
                checks[f'cut_{p}_{len(readers)}'] = bool(torch.allclose(observed, expected, atol=1e-9, rtol=1e-9))
        checks['live_producers'] = all(float(v.square().mean()) > 1e-12 for v in parts.values())
        changed = tokens.clone(); changed[:, 120:] = (changed[:, 120:]+1)%29
        checks['causal_cut'] = bool(torch.allclose(forward(model, tokens, (('M', 'q1'),))[:, :120],
                                                   forward(model, changed, (('M', 'q1'),))[:, :120], atol=1e-9, rtol=1e-9))
    return {'passed': all(checks.values()), 'checks': checks}
