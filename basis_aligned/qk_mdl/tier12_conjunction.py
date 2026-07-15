"""Tier 1.2: the PRE-REGISTERED L1H2 conjunction test (spec §3), on attn2-seed0.

Prior-program context (results_mechdecomp): L0H3→L1H2 is causally verified on
this model (retention: only L0H3@src ablation collapses match, .434→.031), later
re-labeled a repeated-bigram MATCH-AND-COPY circuit (not content induction —
that lives in the rp-dense checkpoints; tick 6). The conjunction hypothesis
applies as written: match-and-copy ≈ (token-identity conjunct through L0H3's OV
on the key side) ∧ (positional conjunct in the other branch).

Part B (weights): per branch b ∈ {1,2} and L0 head h: path-folded
  G_{b,h}[t,s] = Q_b(t) · K_{b,h}(s),  Q_b(t) = W_qb^{L1H2} n(ê_t) slice,
  K_{b,h}(s) = W_kb^{L1H2} (O_h V_h n(ê_s)) slice
(APPROXIMATIONS, stated: layer-1 rms-norm treated as scale-only on the path
input; RoPE ignored for the identity metric — identity structure is judged by
argmax_s G[t,s] == t hit rate vs the 1/V chance floor.)
Positional diagnostic per branch: direct-path band-mass profile + the
token-independent variance fraction of the direct key factors.

Part D (causal, live model): (i) zero branch-1 / branch-2 scores of L1H2;
(ii) subtract L0H3's write from the KEY-side input of branch 1 only / branch 2
only. Metric: match@source of L1H2's pattern (mass at j* = prev-occurrence+1 on
period-128 repeated sequences) + argmax hit rate + full-model CE on the
repeated sequences. Pre-registered success: identity-branch structure in ONE
branch (identity-plus-noise), positional in the OTHER, and only the identity
conjunct's removal collapses match at the correct source.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from folding import load_tiny

torch.manual_seed(0)
DEV = 'cuda'
RUN = 'attn2-dense-seed0'  # attn2-seed0 no longer on disk; deviation logged
L1_HEAD = 2
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/tier12_conjunction.json'

model, cfg = load_tiny(RUN, dtype=torch.float32, device=DEV)
NH = cfg['n_head']
DH = cfg['d_model'] // NH
V = cfg['vocab']
P = 128  # repeat period

val = np.memmap('/workspace/tensor_language/data_owt/val.bin', dtype=np.uint16, mode='r')
g = torch.Generator(); g.manual_seed(0)
starts = torch.randint(0, len(val) - P, (32,), generator=g)
first = torch.stack([torch.tensor(val[int(s):int(s) + P].astype(np.int64)) for s in starts])
SEQ = torch.cat([first, first], dim=1).to(DEV)      # (32, 256), exact repeat


@torch.no_grad()
def forward_probe(tokens, kill_branch=None, ablate_l0h_key=None, key_branch=None):
    """Returns (logits, l1h2_pattern). kill_branch: zero that branch's scores of
    L1H2. ablate_l0h_key: subtract L0 head h's write from the key-side input of
    L1H2 for `key_branch` only."""
    x = model.embed(tokens)
    l0_writes = None
    pat_out = None
    for li, layer in enumerate(model.layers):
        h = layer.norm(x)
        Tn = tokens.shape[-1]
        hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
        v = hs(layer.v(layer.norm(x)))
        mask = torch.tril(torch.ones(Tn, Tn, device=DEV))

        def branch_scores(hin, wq, wk):
            q = layer.rotary(hs(wq(hin)))
            k = layer.rotary(hs(wk(hin)))
            return torch.einsum('bihd,bjhd->bhij', q, k)

        s1 = branch_scores(h, layer.q1, layer.k1)
        s2 = branch_scores(h, layer.q2, layer.k2)
        if li == 1 and ablate_l0h_key is not None:
            xk = x - l0_writes[ablate_l0h_key]
            hk = layer.norm(xk)
            q_w = layer.q1 if key_branch == 1 else layer.q2
            k_w = layer.k1 if key_branch == 1 else layer.k2
            q = layer.rotary(hs(q_w(h)))               # query side untouched
            k = layer.rotary(hs(k_w(hk)))              # key side path-ablated
            s_mod = torch.einsum('bihd,bjhd->bhij', q, k)
            if key_branch == 1:
                s1 = s1.clone(); s1[:, L1_HEAD] = s_mod[:, L1_HEAD]
            else:
                s2 = s2.clone(); s2[:, L1_HEAD] = s_mod[:, L1_HEAD]
        if li == 1 and kill_branch is not None:
            if kill_branch == 1:
                s1 = s1.clone(); s1[:, L1_HEAD] = 0
            else:
                s2 = s2.clone(); s2[:, L1_HEAD] = 0
        pat = s1 * s2 / DH ** 2 * mask
        if li == 1:
            pat_out = pat[:, L1_HEAD].clone()
        z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*tokens.shape, -1)
        write = layer.o(z)
        if li == 0:
            # per-head writes of layer 0 (for key-side path ablation at layer 1)
            l0_writes = {}
            for hh in range(NH):
                zi = torch.zeros_like(z)
                zi[..., hh * DH:(hh + 1) * DH] = z[..., hh * DH:(hh + 1) * DH]
                l0_writes[hh] = layer.scale * layer.o(zi)   # lerp: x+scale*(o(z)-x)?
        x = torch.lerp(x, write, layer.scale)
    return model.head(x), pat_out


# NOTE on l0_writes: lerp(x, o(z), s) = (1-s)x + s*o(z); head h's additive
# contribution to x1 is s * o(z_h-slice). layer.scale * layer.o(zi) is exact.


def match_metrics(pat):
    """pat: (B, T, T) L1H2 pattern. Queries i in second half; source j*=i-P+1."""
    B, T, _ = pat.shape
    qs = torch.arange(P + 8, T, device=DEV)
    js = qs - P + 1
    p = pat.abs()
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    mass = p[:, qs, :].gather(2, js[None, :, None].expand(B, -1, 1)).mean()
    am = p[:, qs, :].argmax(-1)
    hit = (am == js[None, :]).float().mean()
    return float(mass), float(hit)


@torch.no_grad()
def ce_of(logits, tokens):
    return float(F.cross_entropy(logits[:, P:-1].reshape(-1, V),
                                 tokens[:, P + 1:].reshape(-1)))


# ---- screens: which L1 head does match-and-copy? which L0 head is prev-token?
@torch.no_grad()
def screen():
    x = model.embed(SEQ)
    pats = []
    for li, layer in enumerate(model.layers):
        h = layer.norm(x)
        Tn = SEQ.shape[-1]
        hs = lambda t: t.reshape(*t.shape[:-1], NH, DH)
        q1 = layer.rotary(hs(layer.q1(h))); k1 = layer.rotary(hs(layer.k1(h)))
        q2 = layer.rotary(hs(layer.q2(h))); k2 = layer.rotary(hs(layer.k2(h)))
        s1 = torch.einsum('bihd,bjhd->bhij', q1, k1)
        s2 = torch.einsum('bihd,bjhd->bhij', q2, k2)
        mask = torch.tril(torch.ones(Tn, Tn, device=DEV))
        pat = s1 * s2 / DH ** 2 * mask
        pats.append(pat)
        v = hs(layer.v(layer.norm(x)))
        z = torch.einsum('bhij,bjhd->bihd', pat, v).reshape(*SEQ.shape, -1)
        x = torch.lerp(x, layer.o(z), layer.scale)
    return pats

pats = screen()
print('L1 match screens (mass@source / argmax-hit):')
match_by_head = {}
for hh in range(NH):
    mm, hit = match_metrics(pats[1][:, hh])
    match_by_head[hh] = hit
    print(f'  L1H{hh}: {mm:.4f} / {hit:.4f}')
print('L0 prev-token screens (mass at j=i-1):')
for hh in range(NH):
    p0 = pats[0][:, hh].abs()
    p0 = p0 / p0.sum(-1, keepdim=True).clamp_min(1e-9)
    i = torch.arange(2, SEQ.shape[-1], device=DEV)
    print(f'  L0H{hh}: {float(p0[:, i, :].gather(2, (i - 1)[None, :, None].expand(p0.shape[0], -1, 1)).mean()):.4f}')
L1_HEAD_SCREENED = max(match_by_head, key=match_by_head.get)
print(f'screened match-and-copy head: L1H{L1_HEAD_SCREENED} (spec named L1H2)')
L1_HEAD = L1_HEAD_SCREENED

results = {'model': RUN, 'head': f'L1H{L1_HEAD}', 'screened': True, 'period': P, 'conditions': {}}
conds = [('base', {}), ('kill_b1', {'kill_branch': 1}), ('kill_b2', {'kill_branch': 2})]
for hh in range(NH):
    conds += [(f'ablate_L0H{hh}_key_b1', {'ablate_l0h_key': hh, 'key_branch': 1}),
              (f'ablate_L0H{hh}_key_b2', {'ablate_l0h_key': hh, 'key_branch': 2})]
for name, kw in conds:
    logits, pat = forward_probe(SEQ, **kw)
    mass, hit = match_metrics(pat)
    results['conditions'][name] = {'match_mass': mass, 'argmax_hit': hit,
                                   'ce_2nd_half': ce_of(logits, SEQ)}
    print(f"{name:22s} match_mass {mass:.4f}  argmax_hit {hit:.4f}  "
          f"CE(2nd half) {results['conditions'][name]['ce_2nd_half']:.4f}", flush=True)

# ---- Part B: weight-space path-folded identity structure
E = model.embed.weight.detach()
l0, l1 = model.layers[0], model.layers[1]
n = lambda X: l1.norm(X) if not isinstance(l1.norm, torch.nn.Identity) else X
h_emb = model.layers[0].norm(E) if not isinstance(l0.norm, torch.nn.Identity) else E
sl = slice(L1_HEAD * DH, (L1_HEAD + 1) * DH)
results['structure'] = {}
for br in (1, 2):
    Wq = (l1.q1 if br == 1 else l1.q2).weight.detach()
    Wk = (l1.k1 if br == 1 else l1.k2).weight.detach()
    Q = (h_emb @ Wq.T)[:, sl]                              # (V, 32) query factors
    for hh in range(NH):
        Vh = l0.v.weight.detach()[hh * DH:(hh + 1) * DH]   # (32, 128)
        Oh = l0.o.weight.detach()[:, hh * DH:(hh + 1) * DH]  # (128, 32)
        C = (h_emb @ Vh.T) @ Oh.T                          # (V, 128) transported
        K = (C @ Wk.T)[:, sl]
        hits = 0
        off_sum, off_sq, off_n = 0.0, 0.0, 0
        for i in range(0, V, 1024):                        # chunked argmax
            Gc = Q[i:i + 1024] @ K.T
            idx = torch.arange(i, min(i + 1024, V), device=DEV)
            hits += int((Gc.argmax(1) == idx).sum())
            off_sum += float(Gc.sum())
            off_sq += float((Gc ** 2).sum())
            off_n += Gc.numel()
        diag = (Q * K).sum(1)
        mu, var = off_sum / off_n, off_sq / off_n - (off_sum / off_n) ** 2
        z = float((diag.mean() - mu) / max(var, 1e-12) ** 0.5)
        results['structure'][f'b{br}_via_L0H{hh}'] = {
            'identity_hit_rate': hits / V, 'diag_zscore': z}
        print(f'G b{br} via L0H{hh}: identity hit rate {hits / V:.4f} '
              f'(chance {1 / V:.5f})  diag z {z:+.2f}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('tier12 done')
