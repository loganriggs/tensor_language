"""L2 head-specific causal edge cuts and final source-port subset restoration."""
from contextlib import contextmanager
import types
import torch
from suffix_join_writer_reference import edge_masks, native_ports


@contextmanager
def modify(model, mask, heads, ports=None, source=None):
    layer = model.layers[2]; original = layer.pattern; handles = []
    def pattern(_self, x):
        out = original(x).clone()
        out[:, list(heads)] = out[:, list(heads)].masked_fill(mask[:, None], 0)
        return out
    layer.pattern = types.MethodType(pattern, layer)
    try:
        for name, values in (source or {}).items():
            assert name in ('k1', 'k2', 'v')
            def restore(_module, _args, output, values=values):
                return torch.where(ports[..., None], values, output)
            handles.append(getattr(model.layers[-1], name).register_forward_hook(restore))
        yield
    finally:
        for h in handles:
            h.remove()
        del layer.pattern


def controls():
    from deep_model import DeepModel
    import causal_suffix_join_reference as data
    from suffix_join_writer_reference import explicit_masked
    tokens = data.populations(torch)['iid'][0][:4]
    mask, _, ports, _ = edge_masks(tokens)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
    checks = {}
    with torch.inference_mode():
        base, values = native_ports(model, tokens)
        with modify(model, mask, (), ports, values):
            checks['identity'] = bool(torch.allclose(model(tokens), base, atol=1e-9, rtol=1e-9))
        expected = explicit_masked(model, tokens, mask, (2,))
        with modify(model, mask, (0, 1, 2, 3)):
            checks['all_head_explicit'] = bool(torch.allclose(model(tokens), expected, atol=1e-9, rtol=1e-9))
        with modify(model, mask, (0, 1, 2, 3), ports, values):
            rescued = model(tokens)
            checks['query_port_closure'] = bool(torch.allclose(rescued[:, 50], base[:, 50], atol=1e-9, rtol=1e-9))
        checks['local_queries_not_claimed_restored'] = bool((rescued[ports]-base[ports]).abs().max() > 1e-12)
        layer = model.layers[2]
        x = model.layers[1](model.layers[0](model.embed(tokens)))
        p = layer.pattern(x)
        for head in range(4):
            with modify(model, mask, (head,)):
                edited = layer.pattern(x)
            wanted = p.clone(); wanted[:, head] = wanted[:, head].masked_fill(mask, 0)
            checks['head_'+str(head)] = bool(torch.equal(edited, wanted))
        checks['hooks_restored'] = bool(torch.equal(base, model(tokens)))
    return {'passed': all(checks.values()), 'checks': checks}
