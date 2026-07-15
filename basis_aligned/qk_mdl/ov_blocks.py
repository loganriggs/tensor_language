"""OV-circuit opener (Logan's steer), two experiments on bilin18:

A. OV sparse-on-its-own: fold each layer-0 head's O∘V into a transported-token
   table c_h(t) = W_o[:, h·128:(h+1)·128] @ (W_v ê_t)[head h]  (V × 1152).
   Compress with {svd-r, vq-k, zero} and audit ΔCE with layer-0 v-side replaced
   by compressed table lookups (pattern and QK untouched).

B. Bilinear-MLP block importance: block-0 MLP input x = n(e_resid + attn_out).
   With the empirical per-position rms FROZEN, the bilinear hidden splits
   EXACTLY into self (e⊙e), cross (e⊙a + a⊙e), source-pair (a⊙a). Drop each
   block; audit ΔCE. Tests Logan's near-one-hot intuition (source-pair ≈
   negligible?) and sizes the V×V cross codebook's target.

Conventions as frozen (ΔCE binding, T=512, audit chunks 4..19 of pile-10k).
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/qk_mdl/ov_blocks.json'

m, cfg = load_elriggs('bilin18')
NH, HD = cfg['n_head'], cfg['n_embd'] // cfg['n_head']
D = cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:]
blk0 = m.transformer.h[0]
a0 = blk0.attn


@torch.no_grad()
def ov_tables():
    E = m.transformer.wte.weight.detach().float()
    h = F.rms_norm(E, (D,))
    Vw = a0.c_v.weight.detach().float()
    Ow = a0.c_proj.weight.detach().float()
    vt = (h @ Vw.T).view(V, NH, HD)                    # v(t) per head
    return vt, Ow                                      # c_h(t) = Ow[:, h] @ v_h(t)


VT, OW = ov_tables()


@torch.no_grad()
def forward(tokens, v_tables=None, mlp0_drop=None):
    """v_tables: (V, NH, HD) replacing layer-0 value vectors (lamb-mixing and
    everything else live). mlp0_drop: 'self'|'cross'|'pair'|None for block-0."""
    x = m.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = tokens.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        cos, sin = rope_tables(T, HD, tokens.device, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]

        def qk(lin):
            z = lin(h).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosr, sinr)

        if li == 0 and v_tables is not None:
            v = v_tables[tokens].to(x.dtype)           # (B, T, NH, HD)
        else:
            v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=tokens.device, dtype=torch.bool))
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        att = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x_resid = x                                   # pre-attn residual
        x = x + att
        if li == 0 and mlp0_drop is not None:
            # exact block split with frozen empirical rms
            rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            e_n = (x_resid.float() / rms).to(x.dtype)
            a_n = (att.float() / rms).to(x.dtype)
            L, R = blk.mlp.Left, blk.mlp.Right
            Le, La = L(e_n), L(a_n)
            Re, Ra = R(e_n), R(a_n)
            blocks = {'self': Le * Re, 'cross': Le * Ra + La * Re,
                      'pair': La * Ra}
            hidden = sum(v_ for k_, v_ in blocks.items() if k_ != mlp0_drop)
            y = blk.mlp.Down(hidden) + blk.mlp.Down_bias
            x = x + y
        else:
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    return 30 * torch.tanh(m.lm_head(x) / 30)


@torch.no_grad()
def ce(**kw):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        logits = forward(b[:, :-1], **kw).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               b[:, 1:].reshape(-1)).item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


CE0 = ce()
# sanity: mlp0 block path with NOTHING dropped must reproduce baseline
ce_nodrop = ce(mlp0_drop='none')
print(f'baseline {CE0:.4f}; block-split-no-drop {ce_nodrop:.4f} '
      f'(gate: must match to ~1e-3; diff {abs(ce_nodrop - CE0):.2e})')

results = {'baseline_ce': CE0, 'block_split_gate_diff': abs(ce_nodrop - CE0),
           'mlp0_blocks': {}, 'ov': {}}

# ---- B: block importance
for drop in ['self', 'cross', 'pair']:
    d = ce(mlp0_drop=drop) - CE0
    results['mlp0_blocks'][f'drop_{drop}'] = d
    print(f'block-0 MLP drop {drop:5s}: dCE {d:+.4f}', flush=True)

# ---- A: OV tables compressed
@torch.no_grad()
def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            xx = X[i:i + 8192]
            assign[i:i + 8192] = ((xx ** 2).sum(1, keepdim=True) - 2 * xx @ C.T
                                  + (C ** 2).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device)
        Cn.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(len(X), device=X.device))
        C = torch.where((cnt == 0)[:, None], C, Cn / cnt.clamp(min=1)[:, None])
    return C, assign


FULL = 32 * V * HD * NH
grid = []
for r in [4, 16, 64]:
    def make_svd(r=r):
        out = torch.empty_like(VT)
        for hh in range(NH):
            U, S, Vt2 = torch.linalg.svd(VT[:, hh], full_matrices=False)
            out[:, hh] = U[:, :r] @ torch.diag(S[:r]) @ Vt2[:r]
        return out, 32 * NH * r * (V + HD + 1)
    grid.append((f'svd{r}', make_svd))
for k in [64, 1024]:
    def make_vq(k=k):
        import math
        out = torch.empty_like(VT)
        for hh in range(NH):
            C, assign = kmeans(VT[:, hh].contiguous(), k)
            out[:, hh] = C[assign]
        return out, NH * (32 * k * HD + V * math.log2(k))
    grid.append((f'vq{k}', make_vq))
grid.append(('zero', lambda: (torch.zeros_like(VT), 0)))

for name, maker in grid:
    tab, dl = maker()
    d = ce(v_tables=tab) - CE0
    results['ov'][name] = {'dce': d, 'dl_bits': dl, 'ratio': dl / FULL}
    print(f'OV {name:7s}: dCE {d:+.4f}  DL ratio {dl / FULL:.3f}', flush=True)

with open(OUT, 'w') as fh:
    json.dump(results, fh, indent=2)
print('ov_blocks done')
