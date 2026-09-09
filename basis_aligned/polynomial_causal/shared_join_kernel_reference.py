"""Fixed causal records and small shared middle-equality kernel fit.

This is a proposed whole-head replacement, not a native activation probe. Native
execution integration must physically retain only other heads' Q/K rows.
"""
import copy
from contextlib import contextmanager
import types
import torch
from torch import nn
from contextual_history_reference import HistoryReadout

N_ROLE = 8
N_LAG = 51
N_ENTITY = 24
N_GROUP = 2*N_ROLE*N_ROLE*N_LAG


def records(tokens):
    b, t = tokens.shape
    assert 1 <= t <= 51
    pos = torch.arange(t, device=tokens.device)[None].expand(b, -1)
    previous = torch.cat((torch.full_like(tokens[:, :1], 24), tokens[:, :-1]), 1)
    key = torch.where(pos%2 == 0, tokens, previous)
    value = tokens.clone()
    role = (pos%2).clone()
    key = torch.where(pos == 48, 24, key); value = torch.where(pos == 48, 24, value)
    role = torch.where(pos == 48, 2, role)
    key = torch.where(pos == 49, tokens, key); role = torch.where(pos == 49, 3, role)
    key = torch.where(pos == 50, previous, key); value = torch.where(pos == 50, previous, value)
    role = torch.where(pos == 50, tokens-21, role)
    assert bool(((role >= 0)&(role < N_ROLE)).all())
    key = torch.where(key < 24, key, 24); value = torch.where(value < 24, value, 24)
    return key, value, role


def indices(tokens, head):
    key, value, role = records(tokens)
    q, s = (value, key) if head == 0 else (key, value)
    t = tokens.shape[1]; pos = torch.arange(t, device=tokens.device)
    lag = pos[:, None]-pos[None, :]
    match = (q[:, :, None] == s[:, None, :])&(q[:, :, None] < 24)&(lag[None] >= 0)
    same_binding = (pos[:, None] < 48)&(pos[None, :] < 48)&(pos[:, None]//2 == pos[None, :]//2)
    match &= ~same_binding[None]
    group = ((head*N_ROLE+role[:, :, None])*N_ROLE+role[:, None, :])*N_LAG+lag.clamp_min(0)[None]
    middle = (head*N_ENTITY+q.clamp_max(23))[:, :, None].expand_as(group)
    return group, middle, match


def rms_gain(x, norm):
    assert isinstance(norm, torch.nn.RMSNorm) and not norm.elementwise_affine
    eps = torch.finfo(x.dtype).eps if norm.eps is None else norm.eps
    return torch.rsqrt(x.square().mean(-1)+eps)


def fitted_pattern(tokens, gain, theta, gamma, seen):
    factor = gain.square()[:, :, None]*gain.square()[:, None, :]
    out = []
    for h in range(2):
        group, middle, match = indices(tokens, h)
        assert bool(seen[group[match]].all()), 'unseen matched role/lag cell'
        out.append(match*factor*theta[group]*gamma[middle])
    return torch.stack(out, 1)


def fit(group, middle, feature, target, n_group=N_GROUP, n_middle=48, passes=8):
    theta = torch.zeros(n_group, dtype=target.dtype, device=target.device)
    gamma = torch.ones(n_middle, dtype=target.dtype, device=target.device)
    count = torch.bincount(group, minlength=n_group)
    for _ in range(passes):
        a = feature*gamma[middle]
        num = torch.zeros_like(theta).index_add_(0, group, a*target)
        den = torch.zeros_like(theta).index_add_(0, group, a.square())
        theta = num/den.clamp_min(1e-30)
        a = feature*theta[group]
        num = torch.zeros_like(gamma).index_add_(0, middle, a*target)
        den = torch.zeros_like(gamma).index_add_(0, middle, a.square())
        gamma = (num/den.clamp_min(1e-30)).clamp_min(1e-6)
        for h in range(2):
            start, stop = h*(n_middle//2), (h+1)*(n_middle//2)
            scale = gamma[start:stop].mean()
            gamma[start:stop] /= scale
            theta[h*(n_group//2):(h+1)*(n_group//2)] *= scale
    return theta, gamma, count > 0


class JoinAttention(nn.Module):
    """H1/H2 Q/K weights physically absent; native V/O and H0/H3 retained."""
    def __init__(self, layer, theta, gamma, seen):
        super().__init__()
        self.native = copy.deepcopy(layer)
        d = layer.d_head
        for name in ('q1', 'k1', 'q2', 'k2'):
            weight = getattr(layer, name).weight
            reduced = nn.Linear(weight.shape[1], 2*d, bias=False, device=weight.device, dtype=weight.dtype)
            reduced.weight = nn.Parameter(weight.detach().reshape(4, d, -1)[[0, 3]].flatten(0, 1).clone())
            setattr(self.native, name, reduced)
        self.register_buffer('theta', theta.detach().clone())
        self.register_buffer('gamma', gamma.detach().clone())
        self.register_buffer('seen', seen.detach().clone())
        self.tokens = None

    def pattern(self, x):
        assert self.tokens is not None
        layer = self.native; b, t, _ = x.shape; h = layer.norm(x)
        projected = {n: layer.rotary(getattr(layer, n)(h).reshape(b, t, 2, layer.d_head))
                     for n in ('q1', 'k1', 'q2', 'k2')}
        p1 = torch.einsum('bthd,bshd->bhts', projected['q1'], projected['k1'])
        p2 = torch.einsum('bthd,bshd->bhts', projected['q2'], projected['k2'])
        retained = p1*p2/(layer.d_head**2)*layer.mask[:t, :t]
        shared = fitted_pattern(self.tokens, rms_gain(x, layer.norm), self.theta, self.gamma, self.seen)
        return torch.stack((retained[:, 0], shared[:, 0], shared[:, 1], retained[:, 1]), 1)

    def forward(self, x):
        layer = self.native; b, t, _ = x.shape
        v = layer.v(layer.norm(x)).reshape(b, t, 4, layer.d_head)
        z = torch.einsum('bhts,bshd->bthd', self.pattern(x), v).flatten(-2)
        return x+layer.o(z) if layer.residual == 'add' else torch.lerp(x, layer.o(z), layer.scale)


class JoinKernelProgram(HistoryReadout):
    def __init__(self, model, theta, gamma, seen):
        super().__init__(model)
        self.background.layers[2] = JoinAttention(model.layers[2], theta, gamma, seen)

    def parts(self, tokens, masks):
        layer = self.background.layers[2]; layer.tokens = tokens
        try:
            return super().parts(tokens, masks)
        finally:
            layer.tokens = None


def full(program, tokens):
    empty = torch.zeros(*tokens.shape, tokens.shape[1], dtype=torch.bool, device=tokens.device)
    return program(tokens, {k: empty for k in ('H', 'B', 'C')})


@contextmanager
def cut_heads(model, heads):
    layer = model.background.layers[2] if isinstance(model, HistoryReadout) else model.layers[2]
    previous = layer.__dict__.get('pattern'); original = layer.pattern
    def pattern(_self, x):
        p = original(x).clone(); p[:, list(heads)] = 0
        return p
    layer.pattern = types.MethodType(pattern, layer)
    try:
        yield
    finally:
        if previous is None:
            del layer.pattern
        else:
            layer.pattern = previous


def executor_controls():
    from deep_model import DeepModel
    from hop_data import sample_docs
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(15908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
        tokens = sample_docs(2, torch.Generator().manual_seed(15908))[0][:, :51]
        theta = torch.randn(N_GROUP, dtype=torch.float64)*.1
        gamma = torch.rand(48, dtype=torch.float64)+.5
    seen = torch.ones(N_GROUP, dtype=torch.bool)
    program = JoinKernelProgram(model, theta, gamma, seen)
    layer = model.layers[2]; original = layer.pattern
    def reference(_self, x):
        p = original(x).clone()
        p[:, 1:3] = fitted_pattern(tokens, rms_gain(x, layer.norm), theta, gamma, seen)
        return p
    layer.pattern = types.MethodType(reference, layer)
    checks = {}
    with torch.inference_mode():
        for heads in ((), (1,), (2,), (1, 2)):
            with cut_heads(model, heads), cut_heads(program, heads):
                expected = model(tokens); actual = full(program, tokens)
            checks['nonzero_pattern_scatter_fold_'+str(heads)] = bool(torch.allclose(expected, actual, atol=1e-9, rtol=1e-9))
        del layer.pattern
        zero = JoinKernelProgram(model, theta*0, gamma, seen)
        with cut_heads(model, (1, 2)):
            checks['zero_rule_native_removal'] = bool(torch.allclose(model(tokens), full(zero, tokens), atol=1e-9, rtol=1e-9))
        changed = tokens.clone(); changed[:, 30:48] = (changed[:, 30:48]+1)%24
        checks['executor_causal'] = bool(torch.allclose(full(program, tokens)[:, :30], full(program, changed)[:, :30], atol=1e-9, rtol=1e-9))
    checks['removed_half_qk_rows'] = all(getattr(program.background.layers[2].native, n).weight.numel()*2 == getattr(layer, n).weight.numel()
                                       for n in ('q1', 'k1', 'q2', 'k2'))
    checks['transient_tokens_cleared'] = program.background.layers[2].tokens is None
    return {'passed': all(checks.values()), 'checks': checks}


def controls():
    tokens = torch.arange(51)[None]%24
    tokens[:, 48:] = torch.tensor([24, 7, 28])
    key, value, role = records(tokens)
    checks = {'partial_record': bool(key[0, 0] == 0 and value[0, 0] == 0),
              'complete_record': bool(key[0, 1] == 0 and value[0, 1] == 1),
              'marker_has_no_entity': bool(key[0, 48] == 24 and value[0, 48] == 24),
              'query_record': bool((key[0, 49:] == 7).all() and (value[0, 49:] == 7).all()),
              'roles': bool(torch.equal(role[0, 48:], torch.tensor([2, 3, 7])))}
    changed = tokens.clone(); changed[:, 30:] = 0; changed[:, 48:] = torch.tensor([24, 5, 25])
    checks['parser_causal'] = all(torch.equal(a[:, :30], b[:, :30]) for a, b in zip(records(tokens), records(changed)))
    for h in range(2):
        group, middle, match = indices(tokens, h)
        checks['causal_mask_'+str(h)] = not bool(match.triu(1).any())
        checks['distinct_binding_'+str(h)] = not bool(match[0, :48, :48].diagonal().any())
        checks['index_bounds_'+str(h)] = bool(group.max() < N_GROUP and middle.max() < 48)
    # Complete balanced planted Cartesian design with positive entity gains.
    group = torch.arange(8).repeat_interleave(6)
    middle = torch.arange(6).repeat(8)+6*(group//4)
    feature = torch.ones(48, dtype=torch.float64)
    truth_t = torch.tensor([1., -2., 3., .5, 4., -1., 2., .7], dtype=torch.float64)
    truth_g = torch.tensor([.5, 1., 2., 3., 1.5, .8]*2, dtype=torch.float64)
    target = truth_t[group]*truth_g[middle]
    theta, gamma, seen = fit(group, middle, feature, target, n_group=8, n_middle=12)
    checks['planted_fit'] = bool(torch.allclose(theta[group]*gamma[middle], target, atol=1e-10, rtol=1e-10))
    checks['planted_all_seen'] = bool(seen.all())
    return {'passed': all(checks.values()), 'checks': checks, 'charged_kernel_constants': N_GROUP+48}
