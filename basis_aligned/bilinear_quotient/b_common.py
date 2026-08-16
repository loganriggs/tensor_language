"""Part B machinery: multiplicative (bilinear) attention toy models.

Architecture under study (doc B0). One head, two QK circuits whose scores or
patterns combine multiplicatively. Both placements are implemented because they
have different theory:

  logit    s(q,k) = (qᵀW₁k)(qᵀW₂k),  A = softmax(s)
           a product of two matching conditions BEFORE the softmax: the key must
           satisfy a condition from set 1 AND one from set 2.
  postsoft A = normalise(A₁ ⊙ A₂) with A_i = softmax(qᵀW_i k)
           a soft intersection of two attention distributions.

plus the standard single-circuit head as the matched-capacity baseline.

The planted DGP (doc B2) gives every token two independent properties in
DISJOINT embedding subspaces, so "which property does this factor read" is a
question with an exact answer: how much of W_i's mass sits in the type block
versus the property block.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ DGP

class ConjunctiveRetrieval:
    """Query names (type t*, timing p*). Exactly one key matches both; `m` keys
    match the type only and `m` the timing only, so any single-property strategy
    spreads over 1+m keys and scores 1/(1+m) on the payload.

    control='type' makes the type alone sufficient (no type-only distractors),
    which is the task where conjunction should NOT be needed.
    """

    def __init__(self, n_types=8, n_props=8, n_payload=16, seq=16, m=3,
                 d_type=8, d_prop=8, d_pay=16, device='cpu', seed=0,
                 control=None, prop_corr=0.0):
        self.T, self.P, self.V, self.L, self.m = n_types, n_props, n_payload, seq, m
        self.dt, self.dp, self.dv = d_type, d_prop, d_pay
        self.d = d_type + d_prop + d_pay + 1          # +1: query marker
        self.device, self.control, self.prop_corr = device, control, prop_corr
        g = torch.Generator().manual_seed(seed)
        # planted, fixed, orthonormal-ish tables in disjoint subspaces
        self.E_t = torch.linalg.qr(torch.randn(max(n_types, d_type), d_type,
                                               generator=g))[0][:n_types].to(device)
        self.E_p = torch.linalg.qr(torch.randn(max(n_props, d_prop), d_prop,
                                               generator=g))[0][:n_props].to(device)
        self.E_v = torch.linalg.qr(torch.randn(max(n_payload, d_pay), d_pay,
                                               generator=g))[0][:n_payload].to(device)
        self.g = torch.Generator(device=device).manual_seed(seed + 1)

    # -- planted subspace projectors, in embedding coordinates
    def slices(self):
        a = self.dt
        b = a + self.dp
        c = b + self.dv
        return {'type': slice(0, a), 'prop': slice(a, b), 'pay': slice(b, c),
                'marker': slice(c, c + 1)}

    def embed(self, t, p, v, is_query):
        n, L = t.shape
        x = torch.zeros(n, L, self.d, device=self.device)
        s = self.slices()
        x[:, :, s['type']] = self.E_t[t]
        x[:, :, s['prop']] = self.E_p[p]
        x[:, :, s['pay']] = self.E_v[v] * (~is_query).unsqueeze(-1)
        x[:, :, s['marker']] = is_query.unsqueeze(-1).to(x.dtype)
        return x

    def batch(self, n):
        L, m, T, P = self.L, self.m, self.T, self.P
        dev, g = self.device, self.g
        t = torch.randint(0, T, (n, L), generator=g, device=dev)
        p = torch.randint(0, P, (n, L), generator=g, device=dev)
        v = torch.randint(0, self.V, (n, L), generator=g, device=dev)
        tq = torch.randint(0, T, (n,), generator=g, device=dev)
        pq = torch.randint(0, P, (n,), generator=g, device=dev)
        if self.prop_corr > 0:               # tie the timing to the type sometimes
            tie = torch.rand(n, generator=g, device=dev) < self.prop_corr
            pq = torch.where(tie, tq % P, pq)

        # choose slots: 0 = the match, 1..m type-only, m+1..2m prop-only
        perm = torch.stack([torch.randperm(L, generator=g, device=dev) for _ in range(n)])
        r = torch.arange(n, device=dev)
        match = perm[:, 0]
        t[r, match], p[r, match] = tq, pq
        n_type_only = 0 if self.control == 'prop' else m
        n_prop_only = 0 if self.control == 'type' else m
        for j in range(1, 1 + n_type_only):          # same type, different timing
            sl = perm[:, j]
            t[r, sl] = tq
            p[r, sl] = (pq + 1 + torch.randint(0, P - 1, (n,), generator=g, device=dev)) % P
        for j in range(1 + n_type_only, 1 + n_type_only + n_prop_only):
            sl = perm[:, j]
            p[r, sl] = pq
            t[r, sl] = (tq + 1 + torch.randint(0, T - 1, (n,), generator=g, device=dev)) % T
        # remaining slots must match neither
        for j in range(1 + n_type_only + n_prop_only, L):
            sl = perm[:, j]
            t[r, sl] = (tq + 1 + torch.randint(0, T - 1, (n,), generator=g, device=dev)) % T
            p[r, sl] = (pq + 1 + torch.randint(0, P - 1, (n,), generator=g, device=dev)) % P

        y = v[r, match]
        tq_ = torch.cat([t, tq[:, None]], 1)
        pq_ = torch.cat([p, pq[:, None]], 1)
        v_ = torch.cat([v, torch.zeros(n, 1, dtype=v.dtype, device=dev)], 1)
        isq = torch.zeros(n, L + 1, dtype=torch.bool, device=dev)
        isq[:, -1] = True
        x = self.embed(tq_, pq_, v_, isq)
        return {'x': x, 'y': y, 'match': match, 't': t, 'p': p, 'v': v,
                'tq': tq, 'pq': pq}

    def single_property_ceiling(self):
        """Accuracy of the best strategy that uses only one of the two properties."""
        n_t = 1 + (0 if self.control == 'prop' else self.m)
        n_p = 1 + (0 if self.control == 'type' else self.m)
        return max(1.0 / n_t, 1.0 / n_p)


# ------------------------------------------------------------------ models

class Head(torch.nn.Module):
    """kind: 'logit' | 'postsoft' | 'standard'. `rank` is the QK rank per factor;
    'standard' is given 2x the rank so parameter counts match."""

    def __init__(self, d, n_out, kind='logit', rank=8, seed=0, device='cpu'):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind, self.d = kind, d
        r = rank * (2 if kind == 'standard' else 1)
        nf = 1 if kind == 'standard' else 2

        def mk(a, b):
            return torch.nn.Parameter((torch.randn(a, b, generator=g) / math.sqrt(b)).to(device))

        self.Wq = torch.nn.ParameterList([mk(r, d) for _ in range(nf)])
        self.Wk = torch.nn.ParameterList([mk(r, d) for _ in range(nf)])
        self.Wv = mk(d, d)
        self.Wo = mk(n_out, d)
        self.scale = 1.0 / math.sqrt(r)

    def qk(self, i):
        """The factor's bilinear form W_i (d x d): score = x_qᵀ W_i x_k."""
        return self.Wq[i].T @ self.Wk[i] * self.scale

    def scores(self, x):
        """Per-factor raw scores (n_factors, batch, keys)."""
        q, ks = x[:, -1:, :], x[:, :-1, :]
        out = []
        for i in range(len(self.Wq)):
            out.append(torch.einsum('bqd,de,bke->bk', q, self.qk(i), ks))
        return torch.stack(out)

    def pattern(self, x):
        s = self.scores(x)
        if self.kind == 'standard':
            return torch.softmax(s[0], -1), s
        if self.kind == 'logit':
            return torch.softmax(s[0] * s[1], -1), s
        if self.kind == 'unnorm':
            # bilin18's actual placement: the product of two score fields, causally
            # masked, with NO softmax and NO normalisation anywhere
            # (jacclust/tt_model.py:134-144). Entries are signed and rows do not sum
            # to one, so every entropy-based statistic is undefined here.
            return s[0] * s[1], s
        a = torch.softmax(s[0], -1) * torch.softmax(s[1], -1)
        return a / a.sum(-1, keepdim=True).clamp_min(1e-30), s

    def forward(self, x):
        A, _ = self.pattern(x)
        vals = x[:, :-1, :] @ self.Wv.T
        return (A.unsqueeze(-1) * vals).sum(1) @ self.Wo.T

    def factor_patterns(self, x):
        """Each factor's own attention distribution, for the entropy census."""
        s = self.scores(x)
        return [torch.softmax(s[i], -1) for i in range(s.shape[0])]


def train(model, dgp, steps=4000, lr=3e-3, batch=256, log_every=0, eval_every=0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for s in range(steps):
        b = dgp.batch(batch)
        loss = F.cross_entropy(model(b['x']), b['y'])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if eval_every and (s % eval_every == 0 or s == steps - 1):
            hist.append({'step': s, 'loss': float(loss.detach()), **evaluate(model, dgp)})
            if log_every:
                print(f"    step {s:5d} loss {hist[-1]['loss']:.4f} "
                      f"acc {hist[-1]['acc']:.4f} attn {hist[-1]['attn_to_match']:.4f}")
    return hist


@torch.no_grad()
def evaluate(model, dgp, n=4096):
    b = dgp.batch(n)
    out = model(b['x'])
    A, _ = model.pattern(b['x'])
    r = torch.arange(n, device=A.device)
    return {'acc': float((out.argmax(1) == b['y']).float().mean()),
            'attn_to_match': float(A[r, b['match']].mean())}


# ------------------------------------------------ B1 census: the three degeneracies

@torch.no_grad()
def census(model, dgp, n=4096):
    """The (a)/(b)/(c) taxonomy of doc B1, one scalar test each.

    (a) factor collapse   — a factor's own pattern is near-uniform, so the head
                            is really using one circuit. H(A_i) ≈ log L.
    (b) factor alignment  — the two circuits are the same up to scale, so the
                            product is a sharpened copy of one. cos(W₁, W₂) ≈ ±1.
    (c) genuine conjunction — both factors individually broad, product sharp.
    """
    b = dgp.batch(n)
    L = b['x'].shape[1] - 1
    logL = math.log(L)
    A, _ = model.pattern(b['x'])

    def H(p):
        return float(-(p.clamp_min(1e-30).log() * p).sum(-1).mean())

    out = {'n_factors': len(model.Wq), 'log_L': logL, 'H_combined': H(A),
           'H_combined_norm': H(A) / logL}
    if len(model.Wq) == 1:
        out['regime'] = 'single-circuit (no factors)'
        return out
    fp = model.factor_patterns(b['x'])
    out['H_factor'] = [H(p) for p in fp]
    out['H_factor_norm'] = [H(p) / logL for p in fp]
    W1, W2 = model.qk(0), model.qk(1)
    cos = float((W1 * W2).sum() / (W1.norm() * W2.norm()).clamp_min(1e-30))
    out['qk_cosine'] = cos
    out['entropy_drop'] = min(out['H_factor_norm']) - out['H_combined_norm']
    # KL of the composite from each factor (how much the product changes each)
    out['kl_from_factor'] = [float((A.clamp_min(1e-30) *
                                    (A.clamp_min(1e-30) / p.clamp_min(1e-30)).log()).sum(-1).mean())
                             for p in fp]
    collapse = max(out['H_factor_norm']) > 0.97
    aligned = abs(cos) > 0.9
    conj = (min(out['H_factor_norm']) > 0.5 and out['H_combined_norm'] < 0.35)
    out['regime'] = ('(a) factor collapse' if collapse else
                     '(b) factor alignment' if aligned else
                     '(c) genuine conjunction' if conj else 'none of (a)/(b)/(c)')
    return out


# --------------------------------------- what each factor reads (planted subspaces)

@torch.no_grad()
def factor_readout(model, dgp):
    """Fraction of each QK form's mass in each planted (query-side, key-side)
    block. The planted answer is a diagonal block: one factor on type, one on
    timing. Factor order is a gauge, so score up to swapping the two."""
    s = dgp.slices()
    names = ['type', 'prop', 'pay', 'marker']
    out = []
    for i in range(len(model.Wq)):
        W = model.qk(i)
        tot = float((W ** 2).sum()) or 1.0
        blocks = {f'{a}->{b}': float((W[s[a], s[b]] ** 2).sum()) / tot
                  for a in names for b in names}
        out.append({'blocks': blocks,
                    'type_diag': blocks['type->type'], 'prop_diag': blocks['prop->prop'],
                    'best': max(blocks, key=blocks.get), 'best_frac': max(blocks.values())})
    return out


@torch.no_grad()
def ablate_factor(model, dgp, which, n=4096):
    """Replace one factor's contribution with a constant (uniform) one and
    re-measure. Prediction: the head falls to the single-property ceiling."""
    b = dgp.batch(n)
    s = model.scores(b['x'])
    keep = 1 - which
    if model.kind == 'logit':
        # the factors multiply BEFORE the softmax, so replacing one by a constant
        # must keep its scale or the ablation silently changes the temperature
        A = torch.softmax(s[keep] * s[which].mean(-1, keepdim=True), -1)
    elif model.kind == 'unnorm':
        # no softmax in this head at all — substituting one would be a distribution
        # shift, not an ablation. Replace the factor by its own mean score.
        A = s[keep] * s[which].mean(-1, keepdim=True)
    else:
        A = torch.softmax(s[keep], -1)
    vals = b['x'][:, :-1, :] @ model.Wv.T
    out = (A.unsqueeze(-1) * vals).sum(1) @ model.Wo.T
    r = torch.arange(n, device=A.device)
    return {'acc': float((out.argmax(1) == b['y']).float().mean()),
            'attn_to_match': float(A[r, b['match']].mean())}


@torch.no_grad()
def causal_query_swap(model, dgp, field, n=2048):
    """Change the query's type (or timing) and check the attention follows to the
    key carrying the NEW value — the causal test that the factor reads it."""
    b = dgp.batch(n)
    r = torch.arange(n, device=b['x'].device)
    sl = dgp.slices()
    A0, _ = model.pattern(b['x'])
    base = float(A0[r, b['match']].mean())
    x = b['x'].clone()
    if field == 'type':
        newv = (b['tq'] + 1 + torch.randint(0, dgp.T - 1, (n,), device=x.device)) % dgp.T
        x[:, -1, sl['type']] = dgp.E_t[newv]
        still = ((b['t'] == newv[:, None]) & (b['p'] == b['pq'][:, None]))
    else:
        newv = (b['pq'] + 1 + torch.randint(0, dgp.P - 1, (n,), device=x.device)) % dgp.P
        x[:, -1, sl['prop']] = dgp.E_p[newv]
        still = ((b['p'] == newv[:, None]) & (b['t'] == b['tq'][:, None]))
    A1, _ = model.pattern(x)
    return {'attn_to_old_match_before': base,
            'attn_to_old_match_after': float(A1[r, b['match']].mean()),
            'attn_to_new_target_after': float((A1 * still.float()).sum(-1).mean()),
            'frac_batches_with_new_target': float((still.sum(-1) > 0).float().mean())}
