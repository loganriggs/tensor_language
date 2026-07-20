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
OUT = f'{QK}/ov_svd_baseline.json'
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
else:
    CE0 = ce(None); res = {'baseline_ce': CE0, 'arms': {}}
print(f'baseline {CE0:.4f} | head_dim={HD}', flush=True)

# SVD (low-rank) baseline: rank r, lossless at r=HD. bits = (r*HD basis + V*r coeffs)*32 per head.
for r in (8, 16, 32, 64, 96, 128):
    vt = torch.empty_like(VT)
    for hh in range(HD and NH):
        X = VT[:, hh].to(DEV).double()
        U, Sg, Vh = torch.linalg.svd(X - X.mean(0), full_matrices=False)
        Xr = (U[:, :r] * Sg[:r]) @ Vh[:r] + X.mean(0)
        vt[:, hh] = Xr.float()
    bt_ = (r * HD + V * r) * 32 * NH
    d = ce(vt) - CE0
    res['arms'][f'SVD rank={r}'] = {'dce': round(d, 4), 'Mbits': round(bt_/1e6, 1)}
    print(f'SVD rank={r}: dCE {d:+.4f}  {bt_/1e6:.0f}Mbits', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('ov svd baseline done', flush=True)
