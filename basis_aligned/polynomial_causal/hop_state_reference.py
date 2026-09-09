"""Shared causal role/hop state substituted for selected final q1/k1 rows."""
import copy
from contextlib import contextmanager
import torch
from torch import nn
from contextual_history_reference import HistoryReadout


def codes(tokens):
    pos = torch.arange(tokens.shape[1], device=tokens.device)
    role = torch.where(pos < 48, pos % 2, 2+(pos-48) % 4)
    marker = (tokens >= 25) & (tokens <= 28)
    last = torch.where(marker, pos+1, 0).cummax(-1).values
    previous = tokens.gather(1, (last-1).clamp_min(0))-24
    state = torch.where(last > 0, previous, 0)
    return role[None]*5+state


class StateProjection(nn.Module):
    def __init__(self, linear, book, seen):
        super().__init__()
        self.d = book.shape[-1]
        self.weight = nn.Parameter(linear.weight.detach().reshape(4, self.d, -1)[[1, 2]].clone())
        self.register_buffer('book', book.detach().clone())
        self.register_buffer('seen', seen.detach().clone())
        self.current_codes = None

    def forward(self, x):
        assert self.current_codes is not None
        assert self.current_codes.shape == x.shape[:2]
        assert bool(self.seen[self.current_codes].all()), 'unseen semantic state'
        native = torch.einsum('bti,hpi->bthp', x, self.weight)
        lookup = self.book[self.current_codes]
        return torch.stack((lookup[..., 0, :], native[..., 0, :],
                            native[..., 1, :], lookup[..., 1, :]), -2).flatten(-2)


class HopStateReadout(HistoryReadout):
    def __init__(self, model, books, seen):
        super().__init__(model)
        layer = self.background.layers[-1]
        for name in ('q1', 'k1'):
            setattr(layer, name, StateProjection(getattr(layer, name), books[name], seen))

    def parts(self, tokens, masks):
        layer = self.background.layers[-1]
        state = codes(tokens)
        layer.q1.current_codes = layer.k1.current_codes = state
        try:
            return super().parts(tokens, masks)
        finally:
            layer.q1.current_codes = layer.k1.current_codes = None


def full(program, tokens):
    empty = torch.zeros(*tokens.shape, tokens.shape[1], dtype=torch.bool, device=tokens.device)
    return program(tokens, {k: empty for k in ('H', 'B', 'C')})


@contextmanager
def remove_heads(model, heads):
    layer = model.background.layers[-1] if isinstance(model, HistoryReadout) else model.layers[-1]
    def cut(_module, _args, out):
        result = out.reshape(*out.shape[:-1], 4, -1).clone()
        result[..., list(heads), :] = 0
        return result.flatten(-2)
    handle = layer.q1.register_forward_hook(cut)
    try:
        yield
    finally:
        handle.remove()


def controls():
    from deep_model import DeepModel
    from hop_data import sample_docs
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6908)
        model = DeepModel(29, 16, 4, ['attn', 'mlp', 'attn'], 240, norm='rms').double().eval()
        tokens = sample_docs(2, torch.Generator().manual_seed(6908))[0][:, :-1]
        books = {n: torch.randn(30, 2, 4, dtype=torch.float64) for n in ('q1', 'k1')}
    seen = torch.ones(30, dtype=torch.bool)
    state = codes(tokens)
    changed = tokens.clone(); changed[:, 120:] = (changed[:, 120:]+1)%29
    checks = {'state_causal': bool(torch.equal(state[:, :120], codes(changed)[:, :120])),
              'binding_role': bool(torch.equal(state[:, :48], (torch.arange(48)%2*5).expand(2, -1))),
              'first_query_has_no_future_hop': bool((state[:, 48:50] == torch.tensor([10, 15])).all()),
              'visible_hop': bool((state[:, 50] == 20+tokens[:, 50]-24).all()),
              'answer_keeps_hop': bool((state[:, 51] == 25+tokens[:, 50]-24).all())}
    compiled = HopStateReadout(model, books, seen)
    direct = copy.deepcopy(model)
    # Independent explicit scatter reference, preserving direct native W_O.
    handles = []
    for name in ('q1', 'k1'):
        def patch(_module, _args, out, name=name):
            out = out.reshape(*tokens.shape, 4, 4).clone()
            out[..., 0, :] = books[name][state][..., 0, :]
            out[..., 3, :] = books[name][state][..., 1, :]
            return out.flatten(-2)
        handles.append(getattr(direct.layers[-1], name).register_forward_hook(patch))
    with torch.inference_mode():
        for heads in ((), (0,), (3,), (0, 3)):
            with remove_heads(direct, heads), remove_heads(compiled, heads):
                checks[f'direct_scatter_fold_{heads}'] = bool(torch.allclose(direct(tokens), full(compiled, tokens), atol=1e-9, rtol=1e-9))
    for h in handles:
        h.remove()
    checks['transient_state_cleared'] = compiled.background.layers[-1].q1.current_codes is None
    return {'passed': all(checks.values()), 'checks': checks}
