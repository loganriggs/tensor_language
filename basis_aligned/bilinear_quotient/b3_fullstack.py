"""B3 — full-stack testbed: a planted quotient upstream of conjunctive attention.

Composes A6 and B2, with two deliberate departures from the plan, both forced by
earlier results:

  * the attention head uses the UNNORMALISED score product, because that is what
    bilin18 does (`jacclust/tt_model.py:134-144`) and the plan's two placements
    are not it;
  * routing is measured by path SENSITIVITY (the expected outer product of the
    input gradient), not by a kernel of a quadratic form, because the composition
    of a bilinear layer with a bilinear head is quartic and has no single form.
    This is A6's instrument.

Planted design. Each token carries four independent properties in disjoint
embedding subspaces — a type A, a timing B, a payload, and a modifier C — plus
dead coordinates nothing should read. A shared bilinear layer maps tokens to a
bus. Downstream:

    attention factor 1  must read A          (which key has the right type)
    attention factor 2  must read B          (which key has the right timing)
    the MLP path        must read C          (the query's own modifier)
    nobody              may read the dead coordinates

and the answer is (payload of the key matching BOTH A and B) + (C of the query),
so all three paths are load-bearing and the routing table is known exactly.

Predictions, registered before running:
 (i)   each path's sensitivity concentrates on its own planted feature;
 (ii)  the end-to-end preserved information is the INTERSECTION of what layer 1
       transmits with what the readers are sensitive to, so it is strictly
       smaller than layer 1's own transmitted rank;
 (iii) the weights-only ledger matches the construction, with patching as audit;
 (iv)  factor 1 and factor 2 split A and B (up to the factor-swap gauge).
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import init_params, forward, interaction, gauge_refactor

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

NA, NB, NV, NC = 6, 6, 8, 4          # cardinalities of type, timing, payload, modifier
DA = DB = DV = DC = 6
DDEAD = 6
D_TOK = DA + DB + DV + DC + DDEAD + 1     # +1 query marker
D_BUS = 24
H1, H2 = 128, 96
SEQ = 12
MDIST = 3
STEPS = 20000
BATCH = 256

REGISTERED = {
    'i': "each path's sensitivity concentrates on its own planted feature",
    'ii': 'end-to-end preserved information is the intersection of what layer 1 transmits '
          'with what the readers read, strictly smaller than layer 1 alone',
    'iii': 'the weights-only ledger matches the construction, audited by patching',
    'iv': 'the two attention factors split A and B, up to the factor-swap gauge',
}


def make_tables(seed=0):
    g = torch.Generator().manual_seed(seed)

    def tab(n, d):
        return torch.linalg.qr(torch.randn(max(n, d), d, generator=g))[0][:n].to(DEV)

    return {'A': tab(NA, DA), 'B': tab(NB, DB), 'V': tab(NV, DV), 'C': tab(NC, DC)}


def slices():
    o = 0
    out = {}
    for k, dd in (('A', DA), ('B', DB), ('V', DV), ('C', DC), ('dead', DDEAD)):
        out[k] = slice(o, o + dd)
        o += dd
    out['marker'] = slice(o, o + 1)
    return out


SL = slices()


def sampler(T, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def gen(n):
        a = torch.randint(0, NA, (n, SEQ), generator=g, device=DEV)
        b = torch.randint(0, NB, (n, SEQ), generator=g, device=DEV)
        v = torch.randint(0, NV, (n, SEQ), generator=g, device=DEV)
        aq = torch.randint(0, NA, (n,), generator=g, device=DEV)
        bq = torch.randint(0, NB, (n,), generator=g, device=DEV)
        cq = torch.randint(0, NC, (n,), generator=g, device=DEV)
        perm = torch.rand(n, SEQ, generator=g, device=DEV).argsort(1)
        r = torch.arange(n, device=DEV)
        match = perm[:, 0]
        a[r, match], b[r, match] = aq, bq
        for j in range(1, 1 + MDIST):                    # type-only distractors
            sl = perm[:, j]
            a[r, sl] = aq
            b[r, sl] = (bq + 1 + torch.randint(0, NB - 1, (n,), generator=g, device=DEV)) % NB
        for j in range(1 + MDIST, 1 + 2 * MDIST):        # timing-only distractors
            sl = perm[:, j]
            b[r, sl] = bq
            a[r, sl] = (aq + 1 + torch.randint(0, NA - 1, (n,), generator=g, device=DEV)) % NA
        for j in range(1 + 2 * MDIST, SEQ):
            sl = perm[:, j]
            a[r, sl] = (aq + 1 + torch.randint(0, NA - 1, (n,), generator=g, device=DEV)) % NA
            b[r, sl] = (bq + 1 + torch.randint(0, NB - 1, (n,), generator=g, device=DEV)) % NB

        x = torch.zeros(n, SEQ + 1, D_TOK, device=DEV)
        x[:, :SEQ, SL['A']] = T['A'][a]
        x[:, :SEQ, SL['B']] = T['B'][b]
        x[:, :SEQ, SL['V']] = T['V'][v]
        x[:, SEQ, SL['A']] = T['A'][aq]
        x[:, SEQ, SL['B']] = T['B'][bq]
        x[:, SEQ, SL['C']] = T['C'][cq]              # the modifier lives on the query
        x[:, SEQ, SL['marker']] = 1.0
        x[:, :, SL['dead']] = torch.randn(n, SEQ + 1, DDEAD, generator=g, device=DEV)
        y = (v[r, match] + cq) % NV                  # needs the retrieval AND the modifier
        return {'x': x, 'y': y, 'match': match, 'a': a, 'b': b, 'v': v,
                'aq': aq, 'bq': bq, 'cq': cq}

    return gen


class Stack(torch.nn.Module):
    """token -> shared bilinear layer -> bus; then an unnormalised two-factor
    attention head plus a bilinear MLP path on the query."""

    def __init__(self, seed=0):
        super().__init__()

        def par(p):
            return torch.nn.ParameterDict(
                {k: torch.nn.Parameter(v.to(torch.get_default_dtype())) for k, v in p.items()})

        self.l1 = par(init_params(D_TOK, H1, D_BUS, seed=seed, device=DEV))
        self.mlp = par(init_params(D_BUS, H2, D_BUS, seed=seed + 1, device=DEV))
        g = torch.Generator().manual_seed(seed + 2)

        def mk(a, b):
            return torch.nn.Parameter((torch.randn(a, b, generator=g) / math.sqrt(b)).to(DEV)
                                      .to(torch.get_default_dtype()))

        r = 8
        self.Wq = torch.nn.ParameterList([mk(r, D_BUS), mk(r, D_BUS)])
        self.Wk = torch.nn.ParameterList([mk(r, D_BUS), mk(r, D_BUS)])
        self.Wv = mk(D_BUS, D_BUS)
        self.Wo = mk(NV, D_BUS)
        self.scale = 1.0 / r

    def bus(self, x):
        n, L, _ = x.shape
        return forward({k: v for k, v in self.l1.items()}, x.reshape(n * L, -1)).reshape(n, L, -1)

    def qk(self, i):
        return self.Wq[i].T @ self.Wk[i] * self.scale

    def parts(self, x):
        h = self.bus(x)
        q, ks = h[:, -1:, :], h[:, :-1, :]
        s = [torch.einsum('bqd,de,bke->bk', q, self.qk(i), ks) for i in (0, 1)]
        pattern = s[0] * s[1]                       # unnormalised, bilin18-style
        attn = (pattern.unsqueeze(-1) * (ks @ self.Wv.T)).sum(1)
        mlp = forward({k: v for k, v in self.mlp.items()}, h[:, -1, :])
        return attn, mlp, s, h

    def forward(self, x):
        attn, mlp, _, _ = self.parts(x)
        return (attn + mlp) @ self.Wo.T


def path_sensitivity(model, x, which, n_probe=4096):
    """E[(d path / d x_token)(.)^T] in TOKEN space, summed over positions."""
    xs = x[:n_probe].detach().requires_grad_(True)
    attn, mlp, s, h = model.parts(xs)
    if which == 'attn':
        out = attn.sum()
    elif which == 'mlp':
        out = mlp.sum()
    elif which == 'factor0':
        out = s[0].sum()
    elif which == 'factor1':
        out = s[1].sum()
    else:
        raise ValueError(which)
    g, = torch.autograd.grad(out, xs)
    g = g.reshape(-1, D_TOK)
    return (g.T @ g) / g.shape[0]


def block_shares(S):
    tot = float(torch.diagonal(S).sum()) or 1.0
    return {k: float(torch.diagonal(S)[v].sum()) / tot for k, v in SL.items()}


@torch.no_grad()
def patch(model, T, gen, feature, n=2048):
    """Resample one planted feature everywhere and measure each path's change."""
    b = gen(n)
    b2 = gen(n)
    x = b['x'].clone()
    x[:, :, SL[feature]] = b2['x'][:, :, SL[feature]]
    a0, m0, s0, _ = model.parts(b['x'])
    a1, m1, s1, _ = model.parts(x)

    def rel(u, v):
        return float((u - v).pow(2).mean().sqrt() / u.std().clamp_min(1e-30))

    return {'attn': rel(a0, a1), 'mlp': rel(m0, m1),
            'factor0': rel(s0[0], s1[0]), 'factor1': rel(s0[1], s1[1])}


def report(model, T, gen, tag, verbose=True):
    b = gen(8192)
    with torch.no_grad():
        acc = float((model(b['x']).argmax(1) == b['y']).double().mean())
    paths = ['factor0', 'factor1', 'mlp']
    led = {p: block_shares(path_sensitivity(model, b['x'], p)) for p in paths}
    pat = {f: patch(model, T, gen, f) for f in ('A', 'B', 'C', 'dead')}
    # layer 1's own transmission
    xs = b['x'][:2048].detach().requires_grad_(True)
    h = model.bus(xs)
    Sl = torch.zeros(D_TOK, D_TOK, device=DEV, dtype=b['x'].dtype)
    for j in range(D_BUS):
        g, = torch.autograd.grad(h[:, :, j].sum(), xs, retain_graph=True)
        gg = g.reshape(-1, D_TOK)
        Sl += (gg.T @ gg) / gg.shape[0]
    lay = block_shares(Sl.detach())
    res = {'tag': tag, 'acc': acc, 'layer1_transmits': lay, 'ledger': led, 'patch': pat}
    if verbose:
        print(f'  [{tag}] accuracy {acc:.4f}')
        print('    layer 1 transmits: ' + ' '.join(f'{k} {v:.3f}' for k, v in lay.items()))
        for p in paths:
            top = max(('A', 'B', 'C'), key=lambda k: led[p][k])
            print(f'    {p:8s} reads: ' + ' '.join(f'{k} {led[p][k]:.3f}'
                                                   for k in ('A', 'B', 'C', 'dead'))
                  + f'   -> "{top}"')
        print('    patching (rows = feature resampled):')
        for f in ('A', 'B', 'C', 'dead'):
            print(f'      {f:5s} -> ' + ' '.join(f'{p} {pat[f][p]:.3f}' for p in
                                                 ('factor0', 'factor1', 'mlp')))
    return res


def main():
    t0 = time.time()
    out = {'registered': REGISTERED, 'config': {'d_tok': D_TOK, 'd_bus': D_BUS, 'seq': SEQ,
           'm_distractors': MDIST, 'steps': STEPS}, 'runs': []}
    for seed in range(2):
        T = make_tables(seed)
        gen = sampler(T, seed)
        model = Stack(seed).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        for s in range(STEPS):
            b = gen(BATCH)
            loss = torch.nn.functional.cross_entropy(model(b['x']), b['y'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        print(f'== seed {seed} ==')
        r = report(model, T, gen, f'trained s{seed}')
        out['runs'].append(r)

        if seed == 0:
            print('  NULL gauge (refactor layer 1, function preserving):')
            pg, resid = gauge_refactor({k: v.detach() for k, v in model.l1.items()}, seed=9)
            import copy
            m2 = copy.deepcopy(model)
            m2.l1 = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in pg.items()})
            bb = gen(2048)
            with torch.no_grad():
                dfn = float((model(bb['x']) - m2(bb['x'])).abs().max())
            rg = report(m2, T, gen, 'gauge-scrambled', verbose=False)
            out['null_gauge'] = {'residual': resid, 'fn_diff': dfn,
                                 'h_before': H1, 'h_after': int(pg['L'].shape[0]),
                                 'ledger_before': r['ledger'], 'ledger_after': rg['ledger']}
            print(f"    residual {resid:.1e}, hidden {H1} -> {pg['L'].shape[0]}, "
                  f"end-to-end function diff {dfn:.1e}")
            for p in ('factor0', 'factor1', 'mlp'):
                print(f"    {p:8s} " + ' '.join(
                    f"{k} {r['ledger'][p][k]:.3f}->{rg['ledger'][p][k]:.3f}"
                    for k in ('A', 'B', 'C')))

    print('== NULL 1: random weights ==')
    T = make_tables(0)
    gen = sampler(T, 0)
    mr = Stack(555).to(DEV)
    out['null_random'] = report(mr, T, gen, 'random')

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/b3_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
