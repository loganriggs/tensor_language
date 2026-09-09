"""Literal two-shift first-layer program, preserving native coupled weight maps."""
import copy
import torch
from torch import nn
import types


def dense_forward(model, tokens, remove=(), restrict=False):
    first = model.layers[0]
    old = first.pattern
    def pattern(_self, x):
        p = old(x)
        t = torch.arange(x.shape[1], device=x.device)
        lag = t[:, None]-t[None, :]
        kill = (lag > 1) if restrict else torch.zeros_like(lag, dtype=torch.bool)
        for d in remove:
            kill |= lag == d
        return p.masked_fill(kill[None, None], 0)
    first.pattern = types.MethodType(pattern, first)
    try:
        return model(tokens)
    finally:
        del first.pattern


class LocalTransport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.background = copy.deepcopy(model)

    def forward(self, tokens, remove=()):
        x = self.background.embed(tokens)
        layer = self.background.layers[0]
        n = layer.norm(x)
        shape = (*tokens.shape, layer.n_head, layer.d_head)
        vectors = {name: layer.rotary(getattr(layer, name)(n).reshape(shape))
                   for name in ('q1', 'q2', 'k1', 'k2')}
        value = layer.v(n).reshape(shape)
        result = torch.zeros_like(value)
        for d in (0, 1):
            if d in remove:
                continue
            # Slicing prevents cyclic wraparound at the first token.
            count = tokens.shape[1]-d
            score = ((vectors['q1'][:, d:]*vectors['k1'][:, :count]).sum(-1)
                     *(vectors['q2'][:, d:]*vectors['k2'][:, :count]).sum(-1))/(layer.d_head**2)
            result[:, d:] += score[..., None]*value[:, :count]
        update = layer.o(result.flatten(2))
        x = x+update if layer.residual == 'add' else torch.lerp(x, update, layer.scale)
        for layer in self.background.layers[1:]:
            x = layer(x)
        return self.background.head(x)


def controls():
    from deep_model import DeepModel
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3908)
        model = DeepModel(29, 16, 2, ['attn', 'mlp', 'attn'], 16, norm='rms').double().eval()
        tok = torch.randint(29, (3, 12))
    program = LocalTransport(model)
    checks = {}
    with torch.inference_mode():
        for remove in ((), (0,), (1,), (0, 1)):
            actual = program(tok, remove)
            reference = dense_forward(model, tok, remove, True)
            checks[str(remove)] = bool(torch.allclose(actual, reference, atol=1e-9, rtol=1e-9))
        modified = tok.clone()
        modified[:, 6:] = (modified[:, 6:]+1)%29
        checks['causal'] = bool(torch.allclose(program(tok)[:, :6], program(modified)[:, :6],
                                               atol=1e-9, rtol=1e-9))
        effect = float((program(tok)-program(tok, (1,))).abs().max())
        checks['live_previous_branch'] = effect > 1e-10
    return dict(passed=all(checks.values()), checks=checks, previous_effect_max_abs=effect)
