"""Tier 1.2 follow-up: DATA-CONDITIONED identity metric (tick-6 prediction test).

Generic path-folded weights showed identity structure only in (H3, b2, via
L0H0), while the CAUSAL identity routing is via L0H1 into H0.b1 / H3.b2 — the
generic-vs-conditioned gap mechdecomp documented. Pre-stated prediction
(tick 6): conditioning the structure metric on induction data moves the
identity signal into the causal identity branches, via L0H1.

Method (rp model, tiled random-token data, P=96): collect PRE-ROTARY layer-1
query/key vectors per (copy head, branch). Conditional means:
  qbar_b(t)   = mean q-vector at query positions whose token is t
  kbar_b(s)   = mean k-vector at key positions whose PREVIOUS token is s
  kbar_b,h(s) = same, using only L0 head h's (norm-frozen) contribution to the
                key input (x1 decomposition is additive; the empirical rms of
                x1 is frozen per position so the decomposition stays linear)
Identity metric on covered tokens S (count >= 3): hit rate of
argmax_s qbar(t)·kbar(s) == t, + diagonal z-score. Chance = 1/|S|.
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
P, NSEQ, NBATCH = 96, 64, 20
COPY_HEADS = [0, 3]
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier12c_conditioned.json'

model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH = cfg['n_head']
DH = cfg['d_model'] // NH
V = cfg['vocab']
T = cfg['n_ctx']
SOURCES = ['total', 'direct'] + [f'L0H{h}' for h in range(NH)]

# accumulators: per (copy head, branch, source): sum (V, DH) and counts (V,)
q_sum = {(hh, br): torch.zeros(V, DH, device=DEV) for hh in COPY_HEADS for br in (1, 2)}
q_cnt = torch.zeros(V, device=DEV)
k_sum = {(hh, br, s): torch.zeros(V, DH, device=DEV)
         for hh in COPY_HEADS for br in (1, 2) for s in SOURCES}
k_cnt = torch.zeros(V, device=DEV)

l1 = model.layers[1]
l0 = model.layers[0]


@torch.no_grad()
def accumulate(seed):
    g = torch.Generator(); g.manual_seed(seed)
    w = torch.randint(V, (NSEQ, P), generator=g)
    b = w.repeat(1, (T + P - 1) // P)[:, :T].to(DEV)
    x = model.embed(b)
    # layer 0 forward with per-head write capture
    h = l0.norm(x)
    hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
    q1 = l0.rotary(hs(l0.q1(h))); k1 = l0.rotary(hs(l0.k1(h)))
    q2 = l0.rotary(hs(l0.q2(h))); k2 = l0.rotary(hs(l0.k2(h)))
    s1 = torch.einsum('bihd,bjhd->bhij', q1, k1)
    s2 = torch.einsum('bihd,bjhd->bhij', q2, k2)
    mask = torch.tril(torch.ones(T, T, device=DEV))
    pat = s1 * s2 / DH ** 2 * mask
    v = hs(l0.v(l0.norm(x)))
    z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*b.shape, -1)
    writes = {}
    for hh in range(NH):
        zi = torch.zeros_like(z)
        zi[..., hh * DH:(hh + 1) * DH] = z[..., hh * DH:(hh + 1) * DH]
        writes[f'L0H{hh}'] = l0.scale * l0.o(zi)
    x1 = torch.lerp(x, l0.o(z), l0.scale)
    writes['direct'] = x1 - sum(writes[f'L0H{hh}'] for hh in range(NH))
    # frozen empirical norm denominator of x1 (per position)
    rms = x1.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    comp = {'total': x1 / rms, 'direct': writes['direct'] / rms,
            **{f'L0H{hh}': writes[f'L0H{hh}'] / rms for hh in range(NH)}}
    # sanity: total == norm(x1) up to eps handling
    # pre-rotary q/k per branch for the copy heads
    qpos = torch.arange(P + 2, T, device=DEV)
    kpos = torch.arange(P + 1, T, device=DEV)   # keys in steady state; prev tok exists
    qtok = b[:, qpos].flatten()
    ktok_prev = b[:, kpos - 1].flatten()
    for br, (wq, wk) in {1: (l1.q1, l1.k1), 2: (l1.q2, l1.k2)}.items():
        Qfull = hs(wq(comp['total']))            # (B, T, NH, DH), pre-rotary
        for hh in COPY_HEADS:
            q_sum[(hh, br)].index_add_(0, qtok,
                                       Qfull[:, qpos, hh, :].reshape(-1, DH))
        for s in SOURCES:
            Kc = hs(wk(comp[s]))
            for hh in COPY_HEADS:
                k_sum[(hh, br, s)].index_add_(0, ktok_prev,
                                              Kc[:, kpos, hh, :].reshape(-1, DH))
    q_cnt.index_add_(0, qtok, torch.ones_like(qtok, dtype=torch.float))
    k_cnt.index_add_(0, ktok_prev, torch.ones_like(ktok_prev, dtype=torch.float))


for seed in range(NBATCH):
    accumulate(seed)
print(f'coverage: {int((q_cnt >= 3).sum())} query tokens, '
      f'{int((k_cnt >= 3).sum())} prev-key tokens with >=3 obs')

S = ((q_cnt >= 3) & (k_cnt >= 3)).nonzero().flatten()
NS = len(S)
results = {'model': RUN, 'n_covered': NS, 'chance': 1.0 / NS, 'table': {}}
for hh in COPY_HEADS:
    for br in (1, 2):
        Q = (q_sum[(hh, br)][S] / q_cnt[S, None])
        for s in SOURCES:
            K = (k_sum[(hh, br, s)][S] / k_cnt[S, None])
            G = Q @ K.T                       # (NS, NS) ~ 5k x 5k fine
            hits = float((G.argmax(1) == torch.arange(NS, device=DEV)).float().mean())
            diag = G.diagonal()
            z = float((diag.mean() - G.mean()) / G.std().clamp_min(1e-12))
            results['table'][f'L1H{hh}_b{br}_{s}'] = {
                'identity_hit_rate': hits, 'diag_zscore': z}
            print(f'L1H{hh} b{br} [{s:6s}]: hit {hits:.4f} (chance {1 / NS:.5f})  '
                  f'z {z:+.2f}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier12c done')
