"""OV dictionary variants (Logan 2026-07-21): compare three sparse-coding
schemes for the layer-0 value (OV) content tables, each swept over sparsity.

Schemes:
  A. per-token top-k  — each token = signed top-k of ONE shared dict of n atoms
     (the existing e7 move; anchor: n=512 k=16 -> +0.034 L2-fit).
  B. batch-top-k      — one shared dict; keep the top (k_avg * V) coefficients
     across the WHOLE (V x n) code matrix, so sparsity floats per token (some
     tokens use more atoms, some fewer; k is only the average).
  C. routed / block-sparse — cluster the vocabulary into G groups; each group
     gets its OWN small dict (n_g atoms) and its own k_g. Logan's picture:
     some tokens use 8-of-64, others 8-of-a-different-128.

Per-head dictionaries; ΔCE audit via full value-table reconstruction; bits
(structural) reported per arm (estimation data = 0, dictionaries are fit to the
folded weight tables, not to corpus statistics).
"""
import json
import math
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/ov_dict_variants.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
E = m.transformer.wte.weight.detach().float()
E_hat = F.rms_norm(E, (D,))
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
FULL_FLOATS = 32 * V * HD * NH


def train_dict(X, n, k, mode='token', steps=1200, batch=8192, lr=3e-3, seed=0):
    """Learn a dict on rows of X (N, HD). mode in {'token','batch'} sets the
    encoder used during training. Returns (encoder-fn, atoms Dn, bias b, We)."""
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n]].clone().to(DEV)
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = X.mean(0).clone().to(DEV)
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (batch,), device='cpu')].to(DEV)
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        if mode == 'token':
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
        else:  # batch-top-k within the minibatch
            flat = z.abs().reshape(-1)
            thresh = flat.topk(k * len(x)).values.min()
            zc = z * (z.abs() >= thresh)
            xhat = b + zc @ Dn
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(), b.detach(), We.detach()


@torch.no_grad()
def encode_token(X, Dn, b, We, k):
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
    return xhat, k * len(X)                        # (recon, total nonzeros)


@torch.no_grad()
def encode_batch(X, Dn, b, We, kavg):
    z = (X - b) @ We.T
    nnz = kavg * len(X)
    thresh = z.abs().reshape(-1).topk(nnz).values.min()
    zc = z * (z.abs() >= thresh)
    return b + zc @ Dn, int((zc != 0).sum())


@torch.no_grad()
def ce(v_tab):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        bt = AUDIT[i:i + 4].to(DEV)
        x = m.transformer.wte(bt[:, :-1]); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        B, T = x.shape[0], x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
            if li == 0 and v_tab is not None:
                v = v_tab[bt[:, :-1]].to(x.dtype)
            else:
                v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ce_ = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), bt[:, 1:].reshape(-1))
        tot += ce_.item() * bt[:, 1:].numel(); n += bt[:, 1:].numel()
    return tot / n


import os
if os.path.exists(OUT):
    res = json.load(open(OUT)); CE0 = res['baseline_ce']
    print(f'resumed; {len(res["arms"])} arms done, baseline {CE0:.4f}', flush=True)
else:
    CE0 = ce(None); res = {'baseline_ce': CE0, 'arms': {}}
    print(f'baseline {CE0:.4f}', flush=True)


def bits(n_atoms_total_floats, nnz_total, n_for_index):
    return (n_atoms_total_floats * 32 + nnz_total * (32 + math.log2(max(n_for_index, 2))))


# ---- A: per-token top-k, sweep ----
for k in (4, 8, 16, 32):
    if f'A per-token top-k, n=512 k={k}' in res['arms']: continue
    vt = torch.empty_like(VT); nnz = 0
    for hh in range(NH):
        Dn, b, We = train_dict(VT[:, hh].contiguous(), 512, k, mode='token', seed=hh)
        xhat, nz = encode_token(VT[:, hh].to(DEV), Dn, b, We, k)
        vt[:, hh] = xhat.cpu(); nnz += nz
    bt_ = bits(512 * HD * NH, nnz, 512)
    d = ce(vt) - CE0
    res['arms'][f'A per-token top-k, n=512 k={k}'] = {'dce': round(d, 4), 'Mbits': round(bt_/1e6, 1)}
    print(f'A per-token n=512 k={k}: dCE {d:+.4f}  {bt_/1e6:.0f}Mbits', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)

# ---- B: batch-top-k, sweep average k ----
for k in (4, 8, 16, 32):
    if f'B batch-top-k, n=512 avg-k={k}' in res['arms']: continue
    vt = torch.empty_like(VT); nnz = 0
    for hh in range(NH):
        Dn, b, We = train_dict(VT[:, hh].contiguous(), 512, k, mode='batch', seed=hh)
        xhat, nz = encode_batch(VT[:, hh].to(DEV), Dn, b, We, k)
        vt[:, hh] = xhat.cpu(); nnz += nz
    bt_ = bits(512 * HD * NH, nnz, 512)
    d = ce(vt) - CE0
    res['arms'][f'B batch-top-k, n=512 avg-k={k}'] = {'dce': round(d, 4), 'Mbits': round(bt_/1e6, 1)}
    print(f'B batch-top-k n=512 avg-k={k}: dCE {d:+.4f}  {bt_/1e6:.0f}Mbits', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)

# ---- C: routed / block-sparse (per-group dicts) ----
G = 8
gK = torch.Generator(); gK.manual_seed(1)
C0 = E_hat[torch.randperm(V, generator=gK)[:G]].clone().to(DEV)
for _ in range(10):
    a_ = torch.empty(V, dtype=torch.long, device=DEV)
    for i in range(0, V, 8192):
        xx = E_hat[i:i+8192].to(DEV)
        a_[i:i+8192] = ((xx*xx).sum(1,True) - 2*xx@C0.T + (C0*C0).sum(1)[None]).argmin(1)
    Cn = torch.zeros_like(C0); c2 = torch.zeros(G, device=DEV)
    Cn.index_add_(0, a_, E_hat.to(DEV)); c2.index_add_(0, a_, torch.ones(V, device=DEV))
    nz = c2 > 0; C0[nz] = Cn[nz]/c2[nz][:,None]
GRP = a_  # keep on DEV
gsz = torch.bincount(GRP, minlength=G)
print('routed group sizes:', gsz.tolist(), flush=True)

for tag, n_g_fn, k in [('uniform n_g=128 k=8', lambda s: 128, 8),
                       ('adaptive n_g by size k=8', None, 8)]:
    vt = torch.empty_like(VT); nnz = 0; atomf = 0
    for hh in range(NH):
        for gg in range(G):
            gids = (GRP == gg).nonzero().squeeze(1)
            if len(gids) < 32:
                vt[gids, hh] = VT[gids, hh].to(DEV); continue
            Xg = VT[gids, hh].to(DEV)
            n_g = 128 if n_g_fn else int(64 * (1 + math.log2(max(len(gids)/2000, 1))))
            n_g = max(32, min(n_g, len(gids)))
            Dn, b, We = train_dict(Xg, n_g, k, mode='token', steps=800, seed=hh*G+gg)
            xhat, nz = encode_token(Xg, Dn, b, We, k)
            vt[gids, hh] = xhat; nnz += nz; atomf += n_g * HD
    bt_ = atomf * 32 + nnz * (32 + math.log2(128)) + V * math.log2(G)
    d = ce(vt) - CE0
    res['arms'][f'C routed G={G}, {tag}'] = {'dce': round(d, 4), 'Mbits': round(bt_/1e6, 1)}
    print(f'C routed {tag}: dCE {d:+.4f}  {bt_/1e6:.0f}Mbits', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('ov dict variants done', flush=True)
