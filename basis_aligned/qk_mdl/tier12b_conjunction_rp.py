"""Tier 1.2 (re-anchored): conjunction test on the GENUINE content-induction
model attn2-s30k-mix50-rp-dense-seed0, copy heads L1H0 + L1H3.

Conventions reconciled with mechdecomp/tier15_induction.py: uniform-random
tokens tiled at period P=96 (copying is the only signal; positional copiers
score chance at P=96), metric = mean softmax P(target) at query positions
[P+2, T-2), guard base > 0.5 and L0H1 causally dominant.

Pre-registered success (spec §3, adapted to the verified circuit): for each
copy head, ONE branch carries the token-identity conjunct — (a) its path-folded
G through the prev-token head shows identity structure, and (b) replacing that
branch's scores with their positional average (token identity destroyed,
Δ-structure kept) collapses P(copy) — while the OTHER branch is positional:
its positional-averaging leaves P(copy) largely intact. Failure reported as
failure.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny

torch.manual_seed(0)
DEV = 'cuda'
RUN = 'attn2-s30k-mix50-rp-dense-seed0'
P, NSEQ = 96, 64
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier12b_conjunction.json'

model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH = cfg['n_head']
DH = cfg['d_model'] // NH
V = cfg['vocab']
T = cfg['n_ctx']

g = torch.Generator(); g.manual_seed(0)
w = torch.randint(V, (NSEQ, P), generator=g)
b = w.repeat(1, (T + 1 + P - 1) // P)[:, :T + 1].to(DEV)
X, Y = b[:, :-1], b[:, 1:]
Q_POS = torch.arange(P + 2, T - 2, device=DEV)


@torch.no_grad()
def forward(tokens, patch=None):
    """patch: dict (layer, head, branch) -> mode; mode in {'pos_avg'} or
    ('ablate_l0_key', h). Returns logits and layer-1 patterns."""
    x = model.embed(tokens)
    l0_writes = {}
    pats1 = None
    for li, layer in enumerate(model.layers):
        h = layer.norm(x)
        Tn = tokens.shape[-1]
        hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
        s = {}
        for br, (wq, wk) in {1: (layer.q1, layer.k1), 2: (layer.q2, layer.k2)}.items():
            q = layer.rotary(hs(wq(h)))
            k = layer.rotary(hs(wk(h)))
            s[br] = torch.einsum('bihd,bjhd->bhij', q, k)
        if patch:
            for (pl, ph, pb), mode in patch.items():
                if pl != li:
                    continue
                sb = s[pb].clone()
                if mode == 'pos_avg':
                    # replace head ph's branch-pb scores by their per-Δ mean
                    sc = sb[:, ph]                                # (B, T, T)
                    d_idx = (torch.arange(Tn, device=DEV)[:, None]
                             - torch.arange(Tn, device=DEV)[None, :]).clamp(min=0)
                    sums = torch.zeros(Tn, device=DEV).index_add_(
                        0, d_idx.flatten(),
                        sc.mean(0).flatten())
                    cnts = torch.zeros(Tn, device=DEV).index_add_(
                        0, d_idx.flatten(), torch.ones(Tn * Tn, device=DEV))
                    sb[:, ph] = (sums / cnts.clamp(min=1))[d_idx][None]
                elif isinstance(mode, tuple) and mode[0] == 'ablate_l0_key':
                    xk = x - l0_writes[mode[1]]
                    hk = layer.norm(xk)
                    wq, wk = ((layer.q1, layer.k1) if pb == 1 else
                              (layer.q2, layer.k2))
                    qf = layer.rotary(hs(wq(h)))
                    kf = layer.rotary(hs(wk(hk)))
                    sb[:, ph] = torch.einsum('bihd,bjhd->bhij', qf, kf)[:, ph]
                s[pb] = sb
        mask = torch.tril(torch.ones(Tn, Tn, device=DEV))
        pat = s[1] * s[2] / DH ** 2 * mask
        if li == 1:
            pats1 = pat
        v = hs(layer.v(layer.norm(x)))
        z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*tokens.shape, -1)
        if li == 0:
            for hh in range(NH):
                zi = torch.zeros_like(z)
                zi[..., hh * DH:(hh + 1) * DH] = z[..., hh * DH:(hh + 1) * DH]
                l0_writes[hh] = layer.scale * layer.o(zi)
        x = torch.lerp(x, layer.o(z), layer.scale)
    return model.head(x), pats1


def pcopy(logits):
    p = torch.softmax(logits.float(), -1)
    return float(p[:, Q_POS, :].gather(2, Y[:, Q_POS].unsqueeze(-1)).mean())


logits, pats = forward(X)
BASE = pcopy(logits)
print(f'base P(copy) = {BASE:.4f} (documented 0.7483)')
assert BASE > 0.5, 'GUARD FAIL: wrong checkpoint/conventions'

js = Q_POS - P + 1
match = {}
for hh in range(NH):
    pp = pats[:, hh].abs()
    pp = pp / pp.sum(-1, keepdim=True).clamp_min(1e-9)
    match[hh] = float((pp[:, Q_POS, :].argmax(-1) == js[None]).float().mean())
print('L1 match argmax-hit:', {f'H{h}': round(v, 3) for h, v in match.items()})
COPY_HEADS = sorted(match, key=match.get, reverse=True)[:2]
print(f'copy heads: {COPY_HEADS}')

results = {'model': RUN, 'base_pcopy': BASE, 'match_screen': match,
           'copy_heads': COPY_HEADS, 'causal': {}, 'structure': {}}

# ---- causal: positional-averaging per (copy head, branch); L0H1 key ablation
for hh in COPY_HEADS:
    for br in (1, 2):
        d = pcopy(forward(X, {(1, hh, br): 'pos_avg'})[0]) - BASE
        results['causal'][f'L1H{hh}_b{br}_posavg'] = d
        print(f'pos-avg L1H{hh} b{br}: dP(copy) {d:+.4f}', flush=True)
        for l0h in range(NH):
            dd = pcopy(forward(X, {(1, hh, br): ('ablate_l0_key', l0h)})[0]) - BASE
            results['causal'][f'L1H{hh}_b{br}_keyablate_L0H{l0h}'] = dd
        top = min(range(NH), key=lambda l0h:
                  results['causal'][f'L1H{hh}_b{br}_keyablate_L0H{l0h}'])
        print(f'  key-path ablations: ' + '  '.join(
            f"L0H{l0h} {results['causal'][f'L1H{hh}_b{br}_keyablate_L0H{l0h}']:+.4f}"
            for l0h in range(NH)), flush=True)

# ---- structure: path-folded identity per (copy head, branch, L0 head)
E = model.embed.weight.detach()
l0, l1 = model.layers[0], model.layers[1]
h_emb = l0.norm(E)
for hh in COPY_HEADS:
    sl = slice(hh * DH, (hh + 1) * DH)
    for br in (1, 2):
        Wq = (l1.q1 if br == 1 else l1.q2).weight.detach()
        Wk = (l1.k1 if br == 1 else l1.k2).weight.detach()
        Qf = (h_emb @ Wq.T)[:, sl]
        for l0h in range(NH):
            Vh = l0.v.weight.detach()[l0h * DH:(l0h + 1) * DH]
            Oh = l0.o.weight.detach()[:, l0h * DH:(l0h + 1) * DH]
            C = (h_emb @ Vh.T) @ Oh.T
            Kf = (C @ Wk.T)[:, sl]
            hits, s1, s2, n = 0, 0.0, 0.0, 0
            for i in range(0, V, 1024):
                Gc = Qf[i:i + 1024] @ Kf.T
                idx = torch.arange(i, min(i + 1024, V), device=DEV)
                hits += int((Gc.argmax(1) == idx).sum())
                s1 += float(Gc.sum()); s2 += float((Gc ** 2).sum()); n += Gc.numel()
            diag = (Qf * Kf).sum(1)
            mu = s1 / n
            z = float((diag.mean() - mu) / max(s2 / n - mu ** 2, 1e-12) ** 0.5)
            results['structure'][f'L1H{hh}_b{br}_via_L0H{l0h}'] = {
                'identity_hit_rate': hits / V, 'diag_zscore': z}
            print(f'G L1H{hh} b{br} via L0H{l0h}: hit {hits / V:.4f} '
                  f'(chance {1 / V:.5f}) z {z:+.2f}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier12b done')
