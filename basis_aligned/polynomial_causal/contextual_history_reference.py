"""Complete small-model execution with an explicit final readout source partition.

Prefix and contextual routers remain native, generated from tokens, and priced.
The semantic masks describe source provenance, not an assertion that native keys
implement an exact symbolic equality predicate.
"""
import copy
import types
import torch
from torch import nn


def source_masks(tokens):
    """CPU parser; input includes the final answer, excluded from model inputs."""
    assert tokens.device.type == 'cpu' and tokens.shape[1] == 240
    b, length = len(tokens), 239
    masks = {name: torch.zeros(b, length, length, dtype=torch.bool) for name in ('H', 'B', 'C')}
    repeated = torch.zeros(b, 48, dtype=torch.bool)
    eligible = repeated.clone()
    age_h, age_c = [], []
    for row in range(b):
        bindings = {int(tokens[row, 2*i]): 2*i+1 for i in range(24)}
        history = []
        for j in range(48):
            qpos = 50+4*j
            entity, hop = int(tokens[row, qpos-1]), int(tokens[row, qpos])-25
            assert 0 <= entity < 24 and 0 <= hop <= 3
            hs = [p for e, k, p in history if e == entity and k == hop]
            cs = [p for e, k, p in history if e != entity and k == hop]
            repeated[row, j] = bool(hs)
            masks['H'][row, qpos, hs] = True
            masks['B'][row, qpos, bindings[entity]] = True
            if hs and len(cs) >= len(hs):
                chosen = cs[-len(hs):]
                masks['C'][row, qpos, chosen] = True
                eligible[row, j] = True
                age_h.extend(qpos-p for p in hs)
                age_c.extend(qpos-p for p in chosen)
            history.append((entity, hop, qpos+1))
    assert not (masks['H'] & masks['B']).any()
    assert not (masks['H'] & masks['C']).any()
    assert not (masks['B'] & masks['C']).any()
    for mask in masks.values():
        assert not mask.triu(1).any()
    return masks, repeated, eligible, dict(history_age_tokens=age_h, control_age_tokens=age_c)


def native_forward(model, tokens, kill=None):
    if kill is None:
        return model(tokens)
    layer = model.layers[-1]
    old = layer.pattern
    def pattern(_self, x):
        return old(x).masked_fill(kill[:, None], 0)
    layer.pattern = types.MethodType(pattern, layer)
    try:
        return model(tokens)
    finally:
        del layer.pattern


class HistoryReadout(nn.Module):
    """Actual final W_O removed; prefix, contextual Q/K/V and residual head retained."""
    def __init__(self, model):
        super().__init__()
        self.background = copy.deepcopy(model)
        layer = self.background.layers[-1]
        assert self.background.spec[-1] == 'attn'
        self.register_buffer('folded', (model.head.weight @ model.layers[-1].o.weight).detach().clone()
                             .reshape(model.head.out_features, layer.n_head, layer.d_head))
        del layer.o

    def parts(self, tokens, masks):
        x = self.background.embed(tokens)
        for layer in self.background.layers[:-1]:
            x = layer(x)
        layer = self.background.layers[-1]
        pattern = layer.pattern(x)
        value = layer.v(layer.norm(x)).reshape(*tokens.shape, layer.n_head, layer.d_head)
        payload = torch.einsum('bshp,vhp->bshv', value, self.folded)
        a, r = (1., 1.) if layer.residual == 'add' else (layer.scale, 1-layer.scale)
        parts = {'R': r*self.background.head(x)}
        regions = dict(masks)
        regions['O'] = ~(masks['H'] | masks['B'])
        for name, region in regions.items():
            parts[name] = a*torch.einsum('bhts,bshv->btv', pattern*region[:, None], payload)
        return parts

    def forward(self, tokens, masks, remove=()):
        parts = self.parts(tokens, masks)
        out = parts['R']+parts['H']+parts['B']+parts['O']
        for name in remove:
            out = out-parts[name]
        return out


def controls():
    from deep_model import DeepModel
    from hop_data import sample_docs
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4908)
        model = DeepModel(29, 16, 2, ['attn', 'mlp', 'attn'], 240, norm='rms').double().eval()
        tokens, _, _ = sample_docs(2, torch.Generator().manual_seed(4908))
    # Force a known repeated query without using an output to choose it.
    tokens[:, 56:60] = tokens[:, 48:52]
    masks, repeated, _, _ = source_masks(tokens)
    checks = {'known_history_edge': bool(masks['H'][:, 58, 51].all() and repeated[:, 2].all())}
    program = HistoryReadout(model)
    tok = tokens[:, :-1]
    with torch.inference_mode():
        for names in ((), ('H',), ('B',), ('H', 'B'), ('C',)):
            kill = None if not names else torch.stack([masks[n] for n in names]).any(0)
            checks[str(names)] = bool(torch.allclose(program(tok, masks, names), native_forward(model, tok, kill),
                                                     atol=1e-9, rtol=1e-9))
        parts = program.parts(tok, masks)
        checks['live_history'] = bool(parts['H'].abs().max() > 1e-12)
        checks['live_binding'] = bool(parts['B'].abs().max() > 1e-12)
        changed = tok.clone()
        changed[:, 120:] = (changed[:, 120:]+1)%29
        # Masks are fixed for this numerical causality control; modified suffix need not be a legal task doc.
        checks['causal'] = bool(torch.allclose(program(tok, masks)[:, :120], program(changed, masks)[:, :120],
                                               atol=1e-9, rtol=1e-9))
    return dict(passed=all(checks.values()), checks=checks)
