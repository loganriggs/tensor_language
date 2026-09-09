"""Fixed causal records and small shared middle-equality kernel fit.

This is a proposed whole-head replacement, not a native activation probe. Native
execution integration must physically retain only other heads' Q/K rows.
"""
import torch

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
