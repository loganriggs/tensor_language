"""Replace current equality contributions while retaining the live remainder."""
import numpy as np
from fp32_add_observation_v1 import observe


def execute(runtime, tokens, specs, target=None):
    torch = runtime.torch
    device = next(runtime.model.parameters()).device
    token_tensor = torch.as_tensor(tokens, dtype=torch.long, device=device)
    b = len(specs)
    terms_array = np.zeros((b, 4, 1152), dtype=np.float32)
    states = {name: np.zeros((b, 3, 1152), dtype=np.float32) for name in ('before', 'total', 'after')}
    calls = []
    def attention(event):
        if target is None or event.site not in (5, 7, 8):
            return event.block.attn(event.state, event.first_value)
        write, first, terms, _ = runtime.r585.factorize_attention_event(
            event, specs, torch=torch, functional=runtime.functional, induction=runtime.induction)
        changed = write.clone()
        indices = [i for i, site in enumerate(runtime.p.SITES) if int(site[1]) == event.site]
        li = (5, 7, 8).index(event.site)
        for local, spec in enumerate(specs):
            q = int(spec['final_position'])
            pieces = []
            for i in indices:
                current = terms[local][runtime.p.SITES[i]]['term'].numpy()
                terms_array[local, i] = current
                pieces.append(np.subtract(target[local, i], current, dtype=np.float32))
            # Match the native aggregate FP32 layer-add convention.
            total = torch.as_tensor(np.stack(pieces), device=device).sum(dim=0)
            states['before'][local, li] = changed[local, q].float().detach().cpu().numpy()
            states['total'][local, li] = total.detach().cpu().numpy()
            changed[local, q] += total.to(changed.dtype)
            states['after'][local, li] = changed[local, q].float().detach().cpu().numpy()
        calls.append(event.site)
        return changed, first
    def mlp(event):
        return event.block.mlp(event.state)
    with torch.inference_mode():
        logits = runtime.facade.forward_with_dispatch(runtime.model, token_tensor, attention, mlp, require_production=False)
    selected = np.stack([logits[i, int(s['final_position'])].float().cpu().numpy() for i, s in enumerate(specs)])
    observation = None
    if target is not None:
        assert calls == [5, 7, 8]
        observation = observe(states['before'], states['total'], states['after'])
        assert observation['passed']
    return selected, terms_array, states, observation


def controls():
    """A nonlinear chain distinguishes live replacement from frozen addition."""
    import torch
    from types import SimpleNamespace
    sites = ('L5H5', 'L7H3', 'L8H3', 'L8H4')
    cache = []
    def term(state, layer, head):
        return state[:, -1] ** 2 * (.01 * (head + 1)) + .1 * layer
    def factor(event, specs, **_):
        ts = []
        for i in range(len(specs)):
            ts.append({s: {'term': term(event.state, event.site, int(s[3]))[i].detach().cpu()}
                       for s in sites if int(s[1]) == event.site})
        return event.block.attn(event.state, None)[0], None, ts, 0.
    class Facade:
        def forward_with_dispatch(self, model, tokens, attention, mlp, **_):
            x = tokens.float()[:, :, None].expand(-1, -1, 1152).clone() * .1
            cache.clear()
            for layer in (5, 7, 8):
                def attn(state, first, l=layer):
                    w = torch.zeros_like(state)
                    for s in sites:
                        if int(s[1]) == l:
                            w[:, -1] += term(state, l, int(s[3]))
                    return w, first
                event = SimpleNamespace(site=layer, state=x, first_value=None,
                                        block=SimpleNamespace(attn=attn))
                cache.extend([term(x, layer, int(s[3])).detach().cpu().numpy().copy()
                              for s in sites if int(s[1]) == layer])
                w, _ = attention(event)
                x = x + w
            return x
    runtime = SimpleNamespace(torch=torch, model=torch.nn.Linear(1, 1), facade=Facade(),
                              r585=SimpleNamespace(factorize_attention_event=factor),
                              functional=None, induction=None, p=SimpleNamespace(SITES=sites))
    tokens = np.array([[1, 2, 3], [4, 5, 6]])
    specs = [dict(final_position=2)] * 2
    native, *_ = execute(runtime, tokens, specs)
    native_terms = np.stack(cache, axis=1)
    identity, _, _, obs = execute(runtime, tokens, specs, native_terms)
    assert np.array_equal(native, identity) and obs['passed']
    target = native_terms * np.float32(1.2)
    changed, current, states, obs = execute(runtime, tokens, specs, target)
    assert np.array_equal(current[:, 0], native_terms[:, 0])
    assert not np.array_equal(current[:, 1:], native_terms[:, 1:])
    assert np.max(np.abs(changed - native)) > .01
    # The applied correction uses the changed current term, not its stale baseline.
    for li, ids in enumerate(((0,), (1,), (2, 3))):
        expected = torch.as_tensor(target[:, ids] - current[:, ids]).sum(1).numpy()
        assert np.array_equal(expected, states['total'][:, li])
    return dict(passed=True, self_donor_exact=True, later_terms_recomputed=True,
                aggregate_delta_exact=True, rounded_transactions_exact=bool(obs['passed']))
